// AI-vs-AI player used to compare the native C++ inference path with the
// exported ONNX model driven from Python.
//
// The search is a deliberate line-by-line mirror of tools/gomoku.py's MCTS
// (same candidate generation, same tactical shortcuts, same selection formula,
// same tie-breaking, same value convention), so the two drivers search the
// same tree at every position and the only difference left is where the
// network forward pass runs.
//
// Modes:
//   play (default)  self-play a full game, timing every move
//   --moves a,b,c   replay that exact move sequence, still timing every move;
//                   identical positions on both drivers makes the per-move
//                   numbers directly comparable
//   --forward-bench time forward_batch_policy alone at batch 1/8/32
//
// Usage:
//   gomoku_play_benchmark --model DIR [--playout 200] [--c-puct 5]
//                         [--max-candidates 60] [--max-moves 225] [--cpu]
//                         [--moves 112,113,...] [--tag NAME] [--forward-bench]
//
// Output is CSV so both drivers can be diffed and aggregated:
//   meta,<key>,<value>
//   move,<ply>,<player>,<row>,<col>,<ms>,<net_ms>,<forwards>,<chosen>
//   summary,<moves>,<total_ms>,<mean_ms>,<median_ms>,<p90_ms>,<min_ms>,<max_ms>,<net_ms>,<winner>
//   moves,<a>,<b>,...
#include "game.h"
#include "network.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gomoku::BOARD_CELLS;
using gomoku::BOARD_SIZE;
using gomoku::Board;
using gomoku::PureNet;

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// mirrors tools/gomoku.py: makes_five(board, move, player)
bool makes_five(const Board& board, int move, int player) {
    static const int DR[4] = {0, 1, 1, 1};
    static const int DC[4] = {1, 0, 1, -1};
    const int row = move / BOARD_SIZE;
    const int col = move % BOARD_SIZE;
    for (int d = 0; d < 4; ++d) {
        int count = 1;
        for (int sign : {-1, 1}) {
            for (int dist = 1; dist < gomoku::N_IN_ROW; ++dist) {
                const int r = row + sign * DR[d] * dist;
                const int c = col + sign * DC[d] * dist;
                if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) break;
                if (board.at(r, c) != player) break;
                ++count;
            }
        }
        if (count >= gomoku::N_IN_ROW) return true;
    }
    return false;
}

// mirrors tools/gomoku.py: nearby_moves(state, radius=2)
std::vector<int> nearby_moves(const Board& board) {
    std::vector<int> moves;
    if (board.move_count() == 0) {
        moves.push_back(BOARD_CELLS / 2);
        return moves;
    }
    std::vector<char> seen(static_cast<size_t>(BOARD_CELLS), 0);
    for (int move : board.occupied_cells()) {
        const int row = move / BOARD_SIZE;
        const int col = move % BOARD_SIZE;
        for (int r = std::max(0, row - 2); r < std::min(BOARD_SIZE, row + 3); ++r) {
            for (int c = std::max(0, col - 2); c < std::min(BOARD_SIZE, col + 3); ++c) {
                const int cell = r * BOARD_SIZE + c;
                if (board.at(cell) == 0 && seen[static_cast<size_t>(cell)] == 0) {
                    seen[static_cast<size_t>(cell)] = 1;
                    moves.push_back(cell);
                }
            }
        }
    }
    if (moves.empty()) return board.legal_moves();
    std::sort(moves.begin(), moves.end());
    return moves;
}

struct Node {
    Node* parent = nullptr;
    double prior = 0.0;
    double value = 0.0;
    int visits = 0;
    std::vector<int> actions;
    std::vector<std::unique_ptr<Node>> children;

    Node* find(int action) {
        for (size_t i = 0; i < actions.size(); ++i)
            if (actions[i] == action) return children[i].get();
        return nullptr;
    }
};

// mirrors MCTSNode.select: first maximum in insertion order, exploration
// scaled by this node's own visit count.
size_t select_child(const Node& node, double c_puct) {
    const double scale = c_puct * std::sqrt(static_cast<double>(std::max(1, node.visits)));
    size_t best = 0;
    double best_score = 0.0;
    for (size_t i = 0; i < node.children.size(); ++i) {
        const Node& child = *node.children[i];
        const double score =
            child.value + scale * child.prior / (1.0 + child.visits);
        if (i == 0 || score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

void expand(Node& node, const std::vector<std::pair<int, double>>& action_probs) {
    for (const auto& entry : action_probs) {
        if (node.find(entry.first) != nullptr) continue;
        auto child = std::make_unique<Node>();
        child->parent = &node;
        child->prior = entry.second;
        node.actions.push_back(entry.first);
        node.children.push_back(std::move(child));
    }
}

// mirrors MCTSNode.update_recursive
void update_recursive(Node* node, double value) {
    while (node != nullptr) {
        node->visits += 1;
        node->value += (value - node->value) / node->visits;
        value = -value;
        node = node->parent;
    }
}

class MirrorMcts {
public:
    MirrorMcts(const PureNet& net, double c_puct, int n_playout, int max_candidates)
        : net_(net), c_puct_(c_puct), n_playout_(n_playout),
          max_candidates_(max_candidates) {}

    // Returns the chosen cell and reports the time spent inside the network
    // and the number of forward calls for this move.
    int get_move(const Board& board, double* net_seconds, int* forward_calls) {
        net_seconds_ = 0.0;
        forward_calls_ = 0;
        root_ = std::make_unique<Node>();
        for (int i = 0; i < n_playout_; ++i) playout(board);
        // mirrors get_move(): max over (visits, prior, -action), first wins.
        size_t best = 0;
        for (size_t i = 0; i < root_->children.size(); ++i) {
            const Node& candidate = *root_->children[i];
            if (i == 0) {
                best = 0;
                continue;
            }
            const Node& current = *root_->children[best];
            const double key[3] = {static_cast<double>(candidate.visits), candidate.prior,
                                   -static_cast<double>(root_->actions[i])};
            const double best_key[3] = {static_cast<double>(current.visits), current.prior,
                                        -static_cast<double>(root_->actions[best])};
            bool greater = false;
            for (int k = 0; k < 3; ++k) {
                if (key[k] > best_key[k]) { greater = true; break; }
                if (key[k] < best_key[k]) break;
            }
            if (greater) best = i;
        }
        if (net_seconds != nullptr) *net_seconds = net_seconds_;
        if (forward_calls != nullptr) *forward_calls = forward_calls_;
        if (root_->children.empty())
            throw std::logic_error("search produced no move");
        return root_->actions[best];
    }

private:
    void playout(const Board& root_board) {
        Board state = root_board;
        Node* node = root_.get();
        while (!node->children.empty()) {
            const size_t index = select_child(*node, c_puct_);
            state.play(node->actions[index]);
            node = node->children[index].get();
        }
        double value = 0.0;
        if (state.game_over()) {
            const int winner = state.winner();
            value = (winner == 0) ? 0.0
                                  : (winner == state.current_player() ? 1.0 : -1.0);
        } else {
            const std::vector<int> candidates = nearby_moves(state);
            const int me = state.current_player();
            std::vector<int> wins;
            std::vector<int> blocks;
            for (int move : candidates) {
                if (makes_five(state, move, me)) wins.push_back(move);
                else if (makes_five(state, move, 3 - me)) blocks.push_back(move);
            }
            if (!wins.empty()) {
                std::vector<std::pair<int, double>> priors;
                priors.reserve(wins.size());
                for (int move : wins)
                    priors.emplace_back(move, 1.0 / static_cast<double>(wins.size()));
                expand(*node, priors);
                value = 1.0;
            } else {
                const std::vector<int>& allowed =
                    (blocks.size() == 1) ? blocks : candidates;
                float encoded[4 * BOARD_CELLS];
                state.encode_state(encoded);
                std::vector<float> log_policy;
                float net_value = 0.0f;
                const Clock::time_point start = Clock::now();
                net_.forward_one(std::vector<float>(encoded, encoded + 4 * BOARD_CELLS),
                                 log_policy, net_value);
                net_seconds_ += seconds_since(start);
                ++forward_calls_;
                const std::vector<int> legal = state.legal_moves();
                double maximum = -std::numeric_limits<double>::infinity();
                for (int move : legal)
                    maximum = std::max(maximum, static_cast<double>(log_policy[move]));
                double total = 0.0;
                for (int move : legal)
                    total += std::exp(static_cast<double>(log_policy[move]) - maximum);
                std::vector<std::pair<int, double>> filtered;
                for (int move : allowed) {
                    const double probability =
                        std::exp(static_cast<double>(log_policy[move]) - maximum) / total;
                    if (std::isfinite(probability) && probability >= 0.0)
                        filtered.emplace_back(move, probability);
                }
                std::sort(filtered.begin(), filtered.end(),
                          [](const std::pair<int, double>& a,
                             const std::pair<int, double>& b) {
                              if (a.second != b.second) return a.second > b.second;
                              return a.first < b.first;
                          });
                if (static_cast<int>(filtered.size()) > max_candidates_)
                    filtered.resize(static_cast<size_t>(max_candidates_));
                double sum = 0.0;
                for (const auto& entry : filtered) sum += entry.second;
                if (sum > 0.0) {
                    for (auto& entry : filtered) entry.second /= sum;
                } else {
                    filtered.clear();
                    const int take =
                        std::min<int>(max_candidates_, static_cast<int>(allowed.size()));
                    for (int i = 0; i < take; ++i)
                        filtered.emplace_back(allowed[static_cast<size_t>(i)],
                                              1.0 / static_cast<double>(take));
                }
                expand(*node, filtered);
                value = static_cast<double>(net_value);
            }
        }
        update_recursive(node, -value);
    }

    const PureNet& net_;
    double c_puct_;
    int n_playout_;
    int max_candidates_;
    std::unique_ptr<Node> root_;
    double net_seconds_ = 0.0;
    int forward_calls_ = 0;
};

std::vector<int> parse_moves(const std::string& text) {
    std::vector<int> moves;
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t comma = text.find(',', begin);
        const std::string field = text.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!field.empty()) {
            size_t end = 0;
            const int value = std::stoi(field, &end);
            if (end != field.size() || value < 0 || value >= BOARD_CELLS)
                throw std::invalid_argument("invalid move in --moves: " + field);
            moves.push_back(value);
        }
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return moves;
}

// Identical LCG in both drivers so the forward benchmark sees the same
// tensors; convolution cost is data independent but the comparison should
// still be over one input.
uint32_t lcg_next(uint32_t& state) {
    state = state * 1103515245u + 12345u;
    return state >> 1;
}

void forward_benchmark(PureNet& net, const std::string& tag,
                       const std::vector<int>& batches, int steps_override) {
    const int largest = *std::max_element(batches.begin(), batches.end());
    std::vector<float> states(static_cast<size_t>(largest) * 4 * BOARD_CELLS);
    uint32_t seed = 20260814u;
    for (size_t i = 0; i < states.size(); ++i)
        states[i] = static_cast<float>(lcg_next(seed) % 1000u) / 1000.0f;
    std::vector<float> policy(static_cast<size_t>(largest) * BOARD_CELLS);
    std::vector<float> value(static_cast<size_t>(largest));
    for (int batch : batches) {
        for (int i = 0; i < 3; ++i)
            net.forward_batch_policy(states.data(), batch, policy.data(), value.data());
        net.synchronize();
        double best = std::numeric_limits<double>::infinity();
        const int steps = steps_override > 0 ? steps_override
                                             : (batch == 1 ? 30 : 20);
        double total = 0.0;
        for (int step = 0; step < steps; ++step) {
            const Clock::time_point start = Clock::now();
            net.forward_batch_policy(states.data(), batch, policy.data(), value.data());
            net.synchronize();
            const double elapsed = seconds_since(start);
            total += elapsed;
            best = std::min(best, elapsed);
        }
        std::printf("forward,%s,batch=%d,ms_per_call=%.4f,ms_per_position=%.4f,"
                    "positions_per_second=%.1f,best_ms=%.4f\n",
                    tag.c_str(), batch, 1000.0 * total / steps,
                    1000.0 * total / steps / batch,
                    batch * steps / total, 1000.0 * best);
        std::fflush(stdout);
    }
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        fraction * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(index, values.size() - 1)];
}

// One-line board dump so a human opponent can read the position after every
// ply without leaving the CSV log. Rows are printed top (14) to bottom (0),
// same orientation as tools/play_onnx.py.
void print_board(const Board& board, int ply) {
    std::printf("board,%d", ply);
    for (int row = BOARD_SIZE - 1; row >= 0; --row) {
        std::printf(";%2d|", row);
        for (int col = 0; col < BOARD_SIZE; ++col) {
            const int stone = board.at(row, col);
            std::printf("%c", stone == 1 ? 'X' : (stone == 2 ? 'O' : '.'));
        }
    }
    std::printf(";last=%d;player=%d\n", board.last_move(), board.current_player());
}

int run(int argc, char** argv) {
    std::string model;
    std::string tag = "cpp";
    std::string forced;
    std::string my_moves;
    int playout = 200;
    int max_candidates = 60;
    int max_moves = BOARD_CELLS;
    int warmup = 3;
    int inference_threads = 1;
    std::vector<int> bench_batches = {1, 8, 32};
    int bench_steps = 0;
    double c_puct = 5.0;
    bool cpu = false;
    bool bench = false;
    bool show_board = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
            return argv[++i];
        };
        if (key == "--model") model = value();
        else if (key == "--tag") tag = value();
        else if (key == "--moves") forced = value();
        else if (key == "--me") my_moves = value();
        else if (key == "--playout") playout = std::stoi(value());
        else if (key == "--max-candidates") max_candidates = std::stoi(value());
        else if (key == "--max-moves") max_moves = std::stoi(value());
        else if (key == "--warmup") warmup = std::stoi(value());
        else if (key == "--inference-threads") inference_threads = std::stoi(value());
        else if (key == "--forward-steps") bench_steps = std::stoi(value());
        else if (key == "--batches") {
            bench_batches.clear();
            const std::string list = value();
            size_t begin = 0;
            while (begin < list.size()) {
                const size_t comma = list.find(',', begin);
                const std::string field = list.substr(
                    begin, comma == std::string::npos ? std::string::npos : comma - begin);
                if (!field.empty()) bench_batches.push_back(std::stoi(field));
                if (comma == std::string::npos) break;
                begin = comma + 1;
            }
            if (bench_batches.empty()) throw std::invalid_argument("empty --batches");
        }
        else if (key == "--c-puct") c_puct = std::stod(value());
        else if (key == "--cpu") cpu = true;
        else if (key == "--print-board") show_board = true;
        else if (key == "--forward-bench") bench = true;
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (playout < 1 || max_candidates < 1 || max_moves < 1 || warmup < 0 ||
        !(c_puct > 0.0))
        throw std::invalid_argument("playout, candidates, max-moves, warmup and c-puct must be valid");
    if (!forced.empty() && !my_moves.empty())
        throw std::invalid_argument("--moves and --me are mutually exclusive");

    PureNet net;
    net.set_cuda_enabled(!cpu);
    net.set_inference_threads(inference_threads);
    if (!model.empty() && !net.load(model))
        throw std::runtime_error("model load failed: " + model);

    std::printf("meta,tag,%s\n", tag.c_str());
    std::printf("meta,backend,%s\n", net.using_cuda() ? "cuda" : "cpu");
    std::printf("meta,model,%s\n", model.empty() ? "<random>" : model.c_str());
    std::printf("meta,channels,%d\n", gomoku::NetConfig::kChannels);
    std::printf("meta,res_blocks,%d\n", gomoku::NetConfig::kResBlocks);
    std::printf("meta,playout,%d\n", playout);
    std::printf("meta,c_puct,%g\n", c_puct);
    std::printf("meta,max_candidates,%d\n", max_candidates);
    std::fflush(stdout);

    if (bench) {
        forward_benchmark(net, tag, bench_batches, bench_steps);
        return 0;
    }

    // Warm up lazy runtime state (CUDA context, BN folding, cuDNN handles) so
    // the first timed move is not a cold start. The Python driver does the same.
    for (int i = 0; i < warmup; ++i) {
        std::vector<float> blank(static_cast<size_t>(4) * BOARD_CELLS, 0.0f);
        std::vector<float> warm_policy(BOARD_CELLS);
        float warm_value = 0.0f;
        net.forward_batch_policy(blank.data(), 1, warm_policy.data(), &warm_value);
    }
    net.synchronize();

    const std::vector<int> forced_moves = parse_moves(forced);
    const std::vector<int> opponent_moves = parse_moves(my_moves);
    size_t mine = 0;
    Board board;
    MirrorMcts mcts(net, c_puct, playout, max_candidates);
    std::vector<double> move_ms;
    std::vector<double> net_ms;
    std::vector<int> moves;
    int mismatches = 0;
    double total_seconds = 0.0;
    double total_net_seconds = 0.0;
    int forwards = 0;

    while (!board.game_over() &&
           static_cast<int>(moves.size()) < max_moves) {
        const int ply = static_cast<int>(moves.size());
        const int player = board.current_player();
        // --moves replays every ply; --me only pins the black (my) plies and
        // lets the engine answer on white's turn.
        int pinned = -1;
        if (!forced_moves.empty()) {
            if (ply >= static_cast<int>(forced_moves.size())) break;
            pinned = forced_moves[static_cast<size_t>(ply)];
        } else if (!opponent_moves.empty() && player == 1) {
            if (mine >= opponent_moves.size()) break;
            pinned = opponent_moves[mine++];
        }
        double move_net_seconds = 0.0;
        int move_forwards = 0;
        const Clock::time_point start = Clock::now();
        const int chosen = mcts.get_move(board, &move_net_seconds, &move_forwards);
        const double elapsed = seconds_since(start);
        total_seconds += elapsed;
        total_net_seconds += move_net_seconds;
        forwards += move_forwards;
        move_ms.push_back(1000.0 * elapsed);
        net_ms.push_back(1000.0 * move_net_seconds);
        const int played = (pinned >= 0) ? pinned : chosen;
        if (pinned >= 0 && chosen != played) ++mismatches;
        moves.push_back(played);
        char line[128];
        std::snprintf(line, sizeof(line), "move,%d,%d,%d,%d,%.3f,%.3f,%d,%d", ply,
                      player, played / BOARD_SIZE, played % BOARD_SIZE,
                      1000.0 * elapsed, 1000.0 * move_net_seconds, move_forwards,
                      chosen);
        std::printf("%s\n", line);
        if (show_board) print_board(board, ply);
        std::fflush(stdout);
        board.play(played);
    }
    // The position after the last timed ply, so an interactive opponent can
    // read the current board without replaying the move list by hand.
    if (show_board) {
        print_board(board, static_cast<int>(moves.size()));
        std::fflush(stdout);
    }
    std::printf("summary,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d\n",
                tag.c_str(), moves.size(), 1000.0 * total_seconds,
                move_ms.empty() ? 0.0
                                : 1000.0 * total_seconds / move_ms.size(),
                percentile(move_ms, 0.5), percentile(move_ms, 0.9),
                move_ms.empty() ? 0.0 : *std::min_element(move_ms.begin(), move_ms.end()),
                move_ms.empty() ? 0.0 : *std::max_element(move_ms.begin(), move_ms.end()),
                1000.0 * total_net_seconds, forwards, mismatches, board.winner());
    std::printf("moves,%s", tag.c_str());
    for (int move : moves) std::printf(",%d", move);
    std::printf("\n");
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "play_benchmark: %s\n", error.what());
        return 2;
    }
}
