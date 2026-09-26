#include "selfplay.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace gomoku {
namespace {

constexpr int64_t VALUE_SCALE = 1LL << 20;

enum NodeState : uint8_t {
    NODE_UNEXPANDED = 0,
    NODE_EXPANDING = 1,
    NODE_EXPANDED = 2,
};

// Virtual loss booked while a playout is in flight.
//
// Selection maximises -q, where q is the value stored on the *child* (from
// the child's own point of view). Adding a positive amount to the child's
// value_sum therefore lowers the parent's score for that child and pushes
// concurrent threads onto different branches. Without it every thread dives
// into the same child, collides on NODE_EXPANDING and burns the playout
// budget on retries.
//
// The weights are integers because the node counters are lock-free atomics.
struct VirtualLossWeights {
    int32_t visits = 0;
    int64_t value_sum = 0;  // units of VALUE_SCALE
};

// state publishes the immutable child range. Search counters are atomic.
struct alignas(32) GNode {
    std::atomic<int64_t> value_sum{0};
    std::atomic<int32_t> visits{0};
    std::atomic<int32_t> virtual_visits{0};
    std::atomic<uint8_t> state{NODE_UNEXPANDED};
    uint32_t child_begin = 0;
    uint8_t child_count = 0;
    int8_t solved_value = 2;  // 2: unknown, otherwise exact side-to-move value
    int16_t action = -1;
    float prior = 0.0f;
};

static_assert(std::atomic<int64_t>::is_always_lock_free,
              "lock-free MCTS requires lock-free 64-bit atomics");

struct GameState {
    Board board;
    std::unique_ptr<GNode[]> nodes;
    // Ping-pong arena used to compact the chosen branch after each move.
    // Child ranges remain contiguous in the rebuilt arena.
    std::unique_ptr<GNode[]> spare_nodes;
    size_t node_capacity = 0;
    std::atomic<uint32_t> next_node{1};
    std::atomic<int> issued{0};
    std::atomic<int> in_flight{0};
    bool finished = false;
    int winner = 0;
    std::vector<std::array<float, 4 * BOARD_CELLS>> states;
    std::vector<std::array<float, BOARD_CELLS>> probs;
    std::vector<int> players;
    std::vector<std::pair<uint32_t, uint32_t>> remap_queue;
};

struct PathRec {
    std::array<int32_t, BOARD_CELLS + 1> nodes{};
    int len = 0;
};

struct CachedEvaluation {
    std::array<float, BOARD_CELLS> policy{};
    float value = 0.0f;
};

struct LeafTask {
    int game = -1;
    int node = -1;
    int eval = -1;
    uint64_t key = 0;
    int forced_move = -1;
    PathRec path;
    Board board;
    CachedEvaluation evaluation;
};

void initialize_node(GNode& node, int action = -1, float prior = 0.0f) {
    node.value_sum.store(0, std::memory_order_relaxed);
    node.visits.store(0, std::memory_order_relaxed);
    node.virtual_visits.store(0, std::memory_order_relaxed);
    node.child_begin = 0;
    node.child_count = 0;
    node.solved_value = 2;
    node.action = static_cast<int16_t>(action);
    node.prior = prior;
    node.state.store(NODE_UNEXPANDED, std::memory_order_relaxed);
}

void reset_tree(GameState& game) {
    game.next_node.store(1, std::memory_order_relaxed);
    game.issued.store(0, std::memory_order_relaxed);
    game.in_flight.store(0, std::memory_order_relaxed);
    initialize_node(game.nodes[0]);
}

// Undoes a virtual loss booked by select_leaf for one path entry.
void release_virtual_loss(GameState& game, const PathRec& path,
                          const VirtualLossWeights& loss) {
    for (int i = 0; i < path.len; ++i) {
        GNode& node = game.nodes[path.nodes[i]];
        node.virtual_visits.fetch_sub(loss.visits, std::memory_order_relaxed);
        node.value_sum.fetch_sub(loss.value_sum, std::memory_order_relaxed);
    }
}

void backup_path(GameState& game, const PathRec& path, float leaf_value,
                 const VirtualLossWeights& loss) {
    if (!std::isfinite(leaf_value) || leaf_value < -1.0001f || leaf_value > 1.0001f)
        throw std::runtime_error("MCTS received an invalid value evaluation");
    // The path still carries its virtual loss, so each entry must be
    // corrected by that amount before the real value is applied.
    int64_t value = static_cast<int64_t>(std::llround(
        static_cast<double>(leaf_value) * VALUE_SCALE));
    for (int i = path.len - 1; i >= 0; --i) {
        GNode& node = game.nodes[path.nodes[i]];
        node.value_sum.fetch_add(value - loss.value_sum,
                                 std::memory_order_relaxed);
        node.visits.fetch_add(1, std::memory_order_relaxed);
        node.virtual_visits.fetch_sub(loss.visits,
                                      std::memory_order_relaxed);
        value = -value;
    }
}

bool makes_five(const Board& board, int move, int player) {
    const int row = move / BOARD_SIZE;
    const int col = move % BOARD_SIZE;
    static const int DR[4] = {0, 1, 1, 1};
    static const int DC[4] = {1, 0, 1, -1};
    for (int d = 0; d < 4; ++d) {
        int count = 1;
        for (int s = 1; s < N_IN_ROW; ++s) {
            const int nr = row + DR[d] * s;
            const int nc = col + DC[d] * s;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (board.at(nr * BOARD_SIZE + nc) != player) break;
            ++count;
        }
        for (int s = 1; s < N_IN_ROW; ++s) {
            const int nr = row - DR[d] * s;
            const int nc = col - DC[d] * s;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (board.at(nr * BOARD_SIZE + nc) != player) break;
            ++count;
        }
        if (count >= N_IN_ROW) return true;
    }
    return false;
}

uint64_t board_key(const Board& board) {
    return board.position_key();
}

#ifdef __linux__
std::vector<int> search_cpus(int inference_threads) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};
    struct Core { int cpu; int max_frequency; };
    std::map<std::pair<int, int>, Core> physical_cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        auto read_integer = [cpu](const char* leaf, int fallback) {
            std::ifstream file("/sys/devices/system/cpu/cpu" +
                               std::to_string(cpu) + "/" + leaf);
            int value = fallback;
            file >> value;
            return value;
        };
        const int package = read_integer("topology/physical_package_id", 0);
        const int core = read_integer("topology/core_id", cpu);
        const int frequency = read_integer("cpufreq/cpuinfo_max_freq", 0);
        const auto key = std::make_pair(package, core);
        auto [it, inserted] = physical_cores.emplace(key, Core{cpu, frequency});
        if (!inserted && frequency > it->second.max_frequency)
            it->second = {cpu, frequency};
    }
    std::vector<Core> sorted;
    sorted.reserve(physical_cores.size());
    for (const auto& entry : physical_cores) sorted.push_back(entry.second);
    std::sort(sorted.begin(), sorted.end(), [](const Core& left,
                                                const Core& right) {
        if (left.max_frequency != right.max_frequency)
            return left.max_frequency > right.max_frequency;
        return left.cpu < right.cpu;
    });
    const size_t skip = std::min(sorted.size(),
        static_cast<size_t>(std::max(0, inference_threads)));
    std::vector<int> cpus;
    cpus.reserve(sorted.size() - skip);
    for (size_t i = skip; i < sorted.size(); ++i) cpus.push_back(sorted[i].cpu);
    return cpus;
}

void pin_search_worker(int worker, const std::vector<int>& cpus) {
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[static_cast<size_t>(worker) % cpus.size()], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
std::vector<int> search_cpus(int) { return {}; }
void pin_search_worker(int, const std::vector<int>&) {}
#endif

struct SearchWorkerScratch {
    std::vector<int> legal_moves;
    std::vector<std::pair<float, int>> candidates;
};

// A batch of games has many consecutive search rounds. Keep the threads and
// their scratch buffers alive while the inference thread advances each game.
class SearchWorkerPool {
public:
    SearchWorkerPool() = default;
    SearchWorkerPool(const SearchWorkerPool&) = delete;
    SearchWorkerPool& operator=(const SearchWorkerPool&) = delete;
    ~SearchWorkerPool() { stop(); }

    void start(int count, const std::vector<int>& cpus) {
        try {
            threads_.reserve(count);
            for (int index = 0; index < count; ++index) {
                threads_.emplace_back([this, index, &cpus] {
                    pin_search_worker(index, cpus);
                    SearchWorkerScratch scratch;
                    uint64_t seen_round = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex_);
                        start_cv_.wait(lock, [&] {
                            return stopping_ || round_ != seen_round;
                        });
                        if (stopping_) return;
                        seen_round = round_;
                        lock.unlock();
                        try {
                            task_(index, scratch);
                        } catch (...) {
                            std::lock_guard<std::mutex> error_lock(mutex_);
                            if (!error_) error_ = std::current_exception();
                            cancelled_->store(true, std::memory_order_release);
                        }
                        lock.lock();
                        if (--remaining_ == 0) done_cv_.notify_one();
                    }
                });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    void launch(std::function<void(int, SearchWorkerScratch&)> task,
                std::atomic<bool>& cancelled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (remaining_ != 0)
            throw std::logic_error("MCTS worker round still active");
        task_ = std::move(task);
        cancelled_ = &cancelled;
        error_ = nullptr;
        remaining_ = static_cast<int>(threads_.size());
        ++round_;
        start_cv_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return remaining_ == 0; });
        task_ = nullptr;
        cancelled_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            start_cv_.notify_all();
        }
        for (std::thread& thread : threads_)
            if (thread.joinable()) thread.join();
    }

    std::mutex mutex_;
    std::condition_variable start_cv_, done_cv_;
    std::vector<std::thread> threads_;
    std::function<void(int, SearchWorkerScratch&)> task_;
    std::atomic<bool>* cancelled_ = nullptr;
    std::exception_ptr error_;
    uint64_t round_ = 0;
    int remaining_ = 0;
    bool stopping_ = false;
};

enum class SelectionResult { Retry, NeedsInference, Terminal };

SelectionResult select_leaf(GameState& game, float c_puct, LeafTask& task,
                            float& terminal_value,
                            const VirtualLossWeights& loss) {
    task.board = game.board;
    task.forced_move = -1;
    task.path.len = 1;
    task.path.nodes[0] = 0;
    int node_index = 0;
    game.nodes[0].virtual_visits.fetch_add(loss.visits,
                                           std::memory_order_relaxed);
    game.nodes[0].value_sum.fetch_add(loss.value_sum,
                                      std::memory_order_relaxed);

    while (true) {
        GNode& node = game.nodes[node_index];
        const uint8_t state = node.state.load(std::memory_order_acquire);
        if (state == NODE_EXPANDING) {
            release_virtual_loss(game, task.path, loss);
            return SelectionResult::Retry;
        }
        if (state == NODE_UNEXPANDED) {
            uint8_t expected = NODE_UNEXPANDED;
            if (!node.state.compare_exchange_strong(
                    expected, NODE_EXPANDING, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                release_virtual_loss(game, task.path, loss);
                return SelectionResult::Retry;
            }
            task.node = node_index;
            const int winner = task.board.winner();
            if (winner != 0 || task.board.move_count() == BOARD_CELLS) {
                node.child_begin = 0;
                node.child_count = 0;
                node.state.store(NODE_EXPANDED, std::memory_order_release);
                terminal_value = winner == 0 ? 0.0f
                    : (winner == task.board.current_player() ? 1.0f : -1.0f);
                return SelectionResult::Terminal;
            }
            // Exact tactical wins must override an untrained value head.
            // This also skips inference on positions whose outcome is known.
            const int current = task.board.current_player();
            const int opponent = 3 - current;
            int opponent_wins = 0;
            for (int move = 0; move < BOARD_CELLS; ++move) {
                if (!task.board.is_empty(move)) continue;
                if (makes_five(task.board, move, current)) {
                    const uint32_t child = game.next_node.fetch_add(
                        1, std::memory_order_relaxed);
                    if (child >= game.node_capacity)
                        throw std::overflow_error("MCTS node arena exhausted");
                    initialize_node(game.nodes[child], move, 1.0f);
                    node.child_begin = child;
                    node.child_count = 1;
                    node.solved_value = 1;
                    node.state.store(NODE_EXPANDED, std::memory_order_release);
                    terminal_value = 1.0f;
                    return SelectionResult::Terminal;
                }
                if (makes_five(task.board, move, opponent)) {
                    task.forced_move = move;
                    ++opponent_wins;
                }
            }
            if (opponent_wins != 1) task.forced_move = -1;
            task.key = board_key(task.board);
            return SelectionResult::NeedsInference;
        }

        if (node.solved_value != 2) {
            task.node = node_index;
            terminal_value = static_cast<float>(node.solved_value);
            return SelectionResult::Terminal;
        }
        const int child_count = node.child_count;
        if (child_count == 0) {
            task.node = node_index;
            const int winner = task.board.winner();
            terminal_value = winner == 0 ? 0.0f
                : (winner == task.board.current_player() ? 1.0f : -1.0f);
            return SelectionResult::Terminal;
        }
        const int parent_visits = node.visits.load(std::memory_order_relaxed) +
            node.virtual_visits.load(std::memory_order_relaxed);
        const float exploration = c_puct * std::sqrt(
            static_cast<float>(std::max(1, parent_visits)));
        int best_index = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int slot = 0; slot < child_count; ++slot) {
            const int child_index = static_cast<int>(node.child_begin) + slot;
            const GNode& child = game.nodes[child_index];
            if (child.state.load(std::memory_order_acquire) == NODE_EXPANDING)
                continue;
            const int visits = child.visits.load(std::memory_order_relaxed);
            const int virtual_visits = child.virtual_visits.load(
                std::memory_order_relaxed);
            const int effective_visits = visits + virtual_visits;
            const int64_t sum = child.value_sum.load(std::memory_order_relaxed);
            const float q = effective_visits == 0 ? 0.0f
                : (static_cast<float>(sum) *
                   (1.0f / static_cast<float>(VALUE_SCALE))) /
                    static_cast<float>(effective_visits);
            const float score = -q + exploration * child.prior /
                (1.0f + static_cast<float>(effective_visits));
            if (score > best_score) {
                best_score = score;
                best_index = child_index;
            }
        }
        if (best_index < 0) {
            release_virtual_loss(game, task.path, loss);
            return SelectionResult::Retry;
        }
        GNode& child = game.nodes[best_index];
        child.virtual_visits.fetch_add(loss.visits, std::memory_order_relaxed);
        child.value_sum.fetch_add(loss.value_sum, std::memory_order_relaxed);
        if (task.path.len >= static_cast<int>(task.path.nodes.size())) {
            child.virtual_visits.fetch_sub(loss.visits,
                                           std::memory_order_relaxed);
            child.value_sum.fetch_sub(loss.value_sum,
                                      std::memory_order_relaxed);
            release_virtual_loss(game, task.path, loss);
            throw std::logic_error("MCTS path exceeds board size");
        }
        task.path.nodes[task.path.len++] = best_index;
        if (!task.board.is_empty(child.action)) {
            release_virtual_loss(game, task.path, loss);
            throw std::logic_error("MCTS selected an occupied cell");
        }
        task.board.play(child.action);
        node_index = best_index;
    }
}


std::array<float, BOARD_CELLS> legal_priors(
        const Board& board, const CachedEvaluation& evaluation,
        float dirichlet_eps, float dirichlet_alpha, std::mt19937& rng) {
    std::array<float, BOARD_CELLS> priors{};
    int legal_count = 0;
    double prior_sum = 0.0;
    for (int move = 0; move < BOARD_CELLS; ++move) {
        if (!board.is_empty(move)) continue;
        float prior = evaluation.policy[move];
        if (!std::isfinite(prior) || prior < 0.0f) prior = 0.0f;
        priors[move] = prior;
        prior_sum += prior;
        ++legal_count;
    }
    if (legal_count == 0) return priors;
    for (int move = 0; move < BOARD_CELLS; ++move) {
        if (!board.is_empty(move)) continue;
        priors[move] = prior_sum > 0.0
            ? static_cast<float>(priors[move] / prior_sum)
            : 1.0f / legal_count;
    }
    if (dirichlet_eps > 0.0f && legal_count > 1) {
        std::array<float, BOARD_CELLS> noise{};
        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);
        double noise_sum = 0.0;
        for (int move = 0; move < BOARD_CELLS; ++move) {
            if (!board.is_empty(move)) continue;
            noise[move] = gamma(rng);
            noise_sum += noise[move];
        }
        if (noise_sum > 0.0) {
            for (int move = 0; move < BOARD_CELLS; ++move) {
                if (!board.is_empty(move)) continue;
                priors[move] = (1.0f - dirichlet_eps) * priors[move] +
                    static_cast<float>(dirichlet_eps * noise[move] / noise_sum);
            }
        }
    }
    return priors;
}

void expand_leaf(GameState& game, const LeafTask& task,
                 const CachedEvaluation& evaluation,
                 std::vector<int>& legal,
                 std::vector<std::pair<float, int>>& candidates,
                 int max_candidates, float dirichlet_eps,
                 float dirichlet_alpha, std::mt19937& rng) {
    legal.clear();
    for (int move = 0; move < BOARD_CELLS; ++move)
        if (task.board.is_empty(move)) legal.push_back(move);
    candidates.clear();

    // select_leaf already solved immediate wins and identified a unique
    // forced block. Do not repeat the four-direction scan after inference.
    const int forced = task.forced_move;
    if (forced >= 0) {
        candidates.emplace_back(1.0f, forced);
    } else {
        // Root noise is drawn over every legal move before candidate pruning.
        const auto priors = legal_priors(task.board, evaluation,
            task.path.len == 1 ? dirichlet_eps : 0.0f,
            dirichlet_alpha, rng);
        for (int move : legal) candidates.emplace_back(priors[move], move);
    }
    const int keep = std::min<int>(max_candidates, candidates.size());
    if (keep > 0)
        std::partial_sort(candidates.begin(), candidates.begin() + keep,
                          candidates.end(), std::greater<std::pair<float, int>>());
    double prior_sum = 0.0;
    for (int i = 0; i < keep; ++i) prior_sum += candidates[i].first;

    GNode& leaf = game.nodes[task.node];
    if (keep > 0) {
        const uint32_t begin = game.next_node.fetch_add(
            static_cast<uint32_t>(keep), std::memory_order_relaxed);
        if (static_cast<size_t>(begin) + keep > game.node_capacity)
            throw std::overflow_error("MCTS node arena exhausted");
        for (int i = 0; i < keep; ++i) {
            const float prior = prior_sum > 0.0
                ? candidates[i].first / prior_sum : 1.0f / keep;
            initialize_node(game.nodes[begin + i], candidates[i].second, prior);
        }
        leaf.child_begin = begin;
        leaf.child_count = static_cast<uint8_t>(keep);
    } else {
        leaf.child_begin = 0;
        leaf.child_count = 0;
    }
    leaf.state.store(NODE_EXPANDED, std::memory_order_release);
}

enum class PromotionMode { Reset, Keep, Refresh };

// Only completed nodes can be promoted; all search workers have joined before
// this is called. A forced block or a proven win keeps its exact child set.
PromotionMode promotion_mode(const GameState& game, uint32_t chosen) {
    if (chosen == 0 || chosen >= game.next_node.load(std::memory_order_relaxed))
        throw std::logic_error("MCTS promotion selected an invalid node");
    const GNode& root = game.nodes[chosen];
    const uint8_t state = root.state.load(std::memory_order_relaxed);
    const int visits = root.visits.load(std::memory_order_relaxed);
    if (state == NODE_UNEXPANDED || visits == 0) return PromotionMode::Reset;
    if (state != NODE_EXPANDED || root.virtual_visits.load(std::memory_order_relaxed))
        throw std::logic_error("MCTS promotion found an in-flight node");
    if (root.child_count == 0)
        throw std::logic_error("nonterminal promoted root has no children");
    if (root.solved_value != 2) return PromotionMode::Keep;

    const int opponent = 3 - game.board.current_player();
    int forced_move = -1;
    int threats = 0;
    for (int move = 0; move < BOARD_CELLS; ++move) {
        if (game.board.is_empty(move) && makes_five(game.board, move, opponent)) {
            forced_move = move;
            if (++threats > 1) break;
        }
    }
    if (threats == 1) {
        if (root.child_count != 1 ||
            game.nodes[root.child_begin].action != forced_move)
            throw std::logic_error("promoted forced block lost its tactical child");
        return PromotionMode::Keep;
    }
    return PromotionMode::Refresh;
}

void copy_node(const GNode& source, GNode& target) {
    target.value_sum.store(source.value_sum.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    target.visits.store(source.visits.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    target.virtual_visits.store(source.virtual_visits.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
    target.child_begin = source.child_begin;
    target.child_count = source.child_count;
    target.solved_value = source.solved_value;
    target.action = source.action;
    target.prior = source.prior;
    target.state.store(source.state.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
}

// Rebuild only the selected branch into a second arena. A node's descendants
// retain their visits and values, while every child range receives new,
// contiguous indices. Refresh draws noise over every legal move, then keeps
// the fresh top-k plus all previously visited children. This admits moves
// outside the old internal shortlist without discarding search evidence.
uint32_t promote_subtree(GameState& game, uint32_t chosen, PromotionMode mode,
                         const CachedEvaluation* evaluation,
                         int max_candidates, float dirichlet_eps,
                         float dirichlet_alpha,
                         std::mt19937& rng) {
    if (mode == PromotionMode::Reset) {
        reset_tree(game);
        return 0;
    }
    if (game.in_flight.load(std::memory_order_relaxed) != 0)
        throw std::logic_error("MCTS promotion still has in-flight work");
    if (!game.spare_nodes)
        game.spare_nodes = std::make_unique<GNode[]>(game.node_capacity);

    const uint32_t old_count = game.next_node.load(std::memory_order_relaxed);
    const GNode& old_root = game.nodes[chosen];
    GNode* rebuilt = game.spare_nodes.get();
    copy_node(old_root, rebuilt[0]);
    rebuilt[0].action = -1;
    rebuilt[0].prior = 0.0f;
    uint32_t next = 1;
    auto reserve = [&](int count) {
        if (static_cast<size_t>(next) + count > game.node_capacity)
            throw std::overflow_error("MCTS compacted arena exceeds capacity");
        const uint32_t begin = next;
        next += static_cast<uint32_t>(count);
        return begin;
    };
    auto& queue = game.remap_queue;
    queue.clear();
    auto copy_child = [&](uint32_t old_index, uint32_t new_index) {
        if (old_index >= old_count)
            throw std::logic_error("MCTS child range exceeds allocated nodes");
        copy_node(game.nodes[old_index], rebuilt[new_index]);
        if (rebuilt[new_index].virtual_visits.load(std::memory_order_relaxed) ||
            rebuilt[new_index].state.load(std::memory_order_relaxed) == NODE_EXPANDING)
            throw std::logic_error("MCTS promotion found an in-flight descendant");
        queue.emplace_back(old_index, new_index);
    };

    if (mode == PromotionMode::Refresh) {
        if (!evaluation) throw std::logic_error("promoted root lacks its network prior");
        std::array<int32_t, BOARD_CELLS> old_children{};
        old_children.fill(-1);
        for (int i = 0; i < old_root.child_count; ++i) {
            const uint32_t index = old_root.child_begin + i;
            if (index >= old_count) throw std::logic_error("invalid old root children");
            const int action = game.nodes[index].action;
            if (action < 0 || action >= BOARD_CELLS || !game.board.is_empty(action) ||
                old_children[action] >= 0)
                throw std::logic_error("invalid promoted root action");
            old_children[action] = static_cast<int32_t>(index);
        }
        const auto priors = legal_priors(game.board, *evaluation,
                                         dirichlet_eps, dirichlet_alpha, rng);
        std::vector<int> selected;
        selected.reserve(2 * max_candidates);
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(BOARD_CELLS);
        for (int move = 0; move < BOARD_CELLS; ++move) {
            if (game.board.is_empty(move))
                ranked.emplace_back(priors[move], move);
        }
        const int fresh_count = std::min<int>(max_candidates, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + fresh_count,
                          ranked.end(), std::greater<std::pair<float, int>>());
        std::array<bool, BOARD_CELLS> included{};
        for (int i = 0; i < fresh_count; ++i) {
            const int move = ranked[i].second;
            selected.push_back(move);
            included[move] = true;
        }
        // Keep evidence from visited branches even if fresh root noise moves
        // them below the new top-k. Unvisited old candidates are expendable.
        for (int move = 0; move < BOARD_CELLS; ++move) {
            const int old_index = old_children[move];
            if (old_index >= 0 &&
                game.nodes[old_index].visits.load(std::memory_order_relaxed) > 0 &&
                !included[move]) {
                selected.push_back(move);
                included[move] = true;
            }
        }
        std::sort(selected.begin(), selected.end());
        double prior_sum = 0.0;
        for (int move : selected) prior_sum += priors[move];
        if (selected.empty()) throw std::logic_error("nonterminal root has no legal move");
        const uint32_t begin = reserve(static_cast<int>(selected.size()));
        rebuilt[0].child_begin = begin;
        rebuilt[0].child_count = static_cast<uint8_t>(selected.size());
        int slot = 0;
        for (int move : selected) {
            GNode& child = rebuilt[begin + slot++];
            if (old_children[move] >= 0)
                copy_child(static_cast<uint32_t>(old_children[move]), begin + slot - 1);
            else
                initialize_node(child, move);
            child.prior = prior_sum > 0.0
                ? static_cast<float>(priors[move] / prior_sum)
                : 1.0f / selected.size();
        }
    } else {
        const int count = old_root.child_count;
        const uint32_t begin = reserve(count);
        rebuilt[0].child_begin = begin;
        for (int i = 0; i < count; ++i)
            copy_child(old_root.child_begin + i, begin + i);
    }

    for (size_t index = 0; index < queue.size(); ++index) {
        const auto [old_index, new_index] = queue[index];
        const GNode& source = game.nodes[old_index];
        GNode& target = rebuilt[new_index];
        const int count = source.child_count;
        if (count == 0) {
            target.child_begin = 0;
            continue;
        }
        if (static_cast<size_t>(source.child_begin) + count > old_count)
            throw std::logic_error("invalid MCTS descendant range");
        const uint32_t begin = reserve(count);
        target.child_begin = begin;
        for (int child = 0; child < count; ++child)
            copy_child(source.child_begin + child, begin + child);
    }
    game.nodes.swap(game.spare_nodes);
    game.next_node.store(next, std::memory_order_relaxed);
    return next;
}

// Training targets retain search evidence even when actual play is greedy.
// A temperature schedule controls exploration of the trajectory, not how
// strongly the policy is asked to imitate an already noisy, finite search.
std::array<float, BOARD_CELLS> root_policy(const GameState& game) {
    std::array<float, BOARD_CELLS> policy{};
    const GNode& root = game.nodes[0];
    int64_t total_visits = 0;
    for (int i = 0; i < root.child_count; ++i)
        total_visits += game.nodes[root.child_begin + i].visits.load(
            std::memory_order_relaxed);
    for (int i = 0; i < root.child_count; ++i) {
        const GNode& child = game.nodes[root.child_begin + i];
        policy[child.action] = total_visits > 0
            ? static_cast<float>(child.visits.load(std::memory_order_relaxed)) /
                static_cast<float>(total_visits)
            : child.prior;
    }
    return policy;
}

}  // namespace

void validate_search_config(const SearchConfig& config) {
    if (config.n_playout <= 0 || config.batch_games <= 0 ||
        config.max_candidates <= 0 || config.max_candidates > BOARD_CELLS ||
        config.eval_opening_moves < 0 || config.eval_opening_moves > 8 ||
        !std::isfinite(config.c_puct) || config.c_puct < 0.0f ||
        !std::isfinite(config.virtual_loss) || config.virtual_loss < 0.0f ||
        config.virtual_loss > 1024.0f ||
        !std::isfinite(config.dirichlet_eps) || config.dirichlet_eps < 0.0f ||
        config.dirichlet_eps > 1.0f ||
        !std::isfinite(config.dirichlet_alpha) || config.dirichlet_alpha <= 0.0f ||
        !std::isfinite(config.unseen_move_smooth) ||
        config.unseen_move_smooth < 0.0f || config.unseen_move_smooth > 1.0f ||
        !std::isfinite(config.temp_high) || config.temp_high < 0.0f ||
        !std::isfinite(config.temp_low) || config.temp_low < 0.0f ||
        config.temp_switch_move < 0 || config.inference_batch_target < 0)
        throw std::invalid_argument("invalid MCTS configuration");
    if (static_cast<size_t>(config.n_playout) * config.batch_games >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        static_cast<size_t>(config.n_playout) * config.max_candidates >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max()) -
                config.max_candidates - 1)
        throw std::invalid_argument("MCTS configuration exceeds index capacity");
}


BatchedSelfPlay::BatchedSelfPlay(PureNet& net, const SearchConfig& config)
    : BatchedSelfPlay(config,
          [&net](const float* states, int batch, float* policies, float* values) {
              net.forward_batch_policy(states, batch, policies, values);
          }, net.using_cuda()) {
    net.set_inference_threads(inference_threads_);
}

BatchedSelfPlay::BatchedSelfPlay(const SearchConfig& config,
                                 BatchPositionEvaluator evaluate,
                                 bool accelerated)
    : evaluate_(std::move(evaluate)), accelerated_(accelerated), config_(config),
      c_puct_(config.c_puct), n_playout_(config.n_playout),
      batch_games_(config.batch_games) {
    validate_search_config(config_);
    if (!evaluate_) throw std::invalid_argument("self-play requires an evaluator");
    int n_threads = config.n_threads;
    if (n_threads <= 0) {
        const unsigned hardware_threads = std::thread::hardware_concurrency();
        n_threads = hardware_threads == 0 ? 1
            : std::min(8, static_cast<int>(hardware_threads));
    }
    // CUDA needs one inference coordinator; CPU inference uses its worker pool.
    inference_threads_ = accelerated_
        ? 1 : std::max(1, std::min(n_threads, 32));

    const char* leaf_batch_text = std::getenv("GOMOK_LEAF_BATCH");
    if (leaf_batch_text != nullptr) {
        leaf_batch_size_ = std::max(
            1, std::min(n_playout_, std::atoi(leaf_batch_text)));
    } else if (accelerated_) {
        // Independent games supply most of the inference batch. Only allow
        // more simultaneous leaves within one tree when the batch needs them.
        constexpr int kDefaultCudaBatch = 32;
        const int target = config_.inference_batch_target > 0
            ? config_.inference_batch_target : kDefaultCudaBatch;
        leaf_batch_size_ = std::max(1, std::min(
            n_playout_, (target + batch_games_ - 1) / batch_games_));
    } else {
        // Independent games provide the batch; limit stale, simultaneous
        // simulations within one tree to the CPU inference worker count.
        leaf_batch_size_ = std::max(1, std::min(n_playout_,
            (inference_threads_ + batch_games_ - 1) / batch_games_));
    }

    const std::vector<int> cpus = search_cpus(inference_threads_);
    const int default_mcts_threads = cpus.empty() ? 2
        : std::min(8, static_cast<int>(cpus.size()));
    const char* mcts_threads_text = std::getenv("GOMOK_MCTS_THREADS");
    mcts_threads_ = mcts_threads_text != nullptr
        ? std::atoi(mcts_threads_text)
        : std::min(default_mcts_threads, std::max(1, n_threads));
    mcts_threads_ = std::max(1, std::min(mcts_threads_, 32));
}

BatchedSelfPlay::BatchedSelfPlay(PureNet& net, float c_puct, int n_playout,
                                 int batch_games, int n_threads)
    : BatchedSelfPlay(net, [&] {
          SearchConfig config;
          config.c_puct = c_puct;
          config.n_playout = n_playout;
          config.batch_games = batch_games;
          config.n_threads = n_threads;
          // Keep the historical behaviour: no root noise, flat temperature.
          config.dirichlet_eps = 0.0f;
          config.temp_high = 1.0f;
          config.temp_low = 1.0f;
          return config;
      }()) {}

std::vector<std::vector<BatchedSelfPlay::Sample>>
BatchedSelfPlay::run_batch(float fixed_temp) {
    if (!std::isfinite(fixed_temp))
        throw std::invalid_argument("self-play temperature must be finite");
    const char* reuse_setting = std::getenv("GOMOK_TREE_REUSE");
    const bool reuse_tree = reuse_setting == nullptr ||
        std::strcmp(reuse_setting, "0") != 0;
    const int batch = batch_games_;
    std::unique_ptr<GameState[]> games(new GameState[batch]);
    const size_t arena_capacity =
        1 + config_.max_candidates + static_cast<size_t>(n_playout_) *
            config_.max_candidates;

    // Derive the integer virtual-loss weights from the configured strength.
    VirtualLossWeights virtual_loss;
    {
        const float strength = leaf_batch_size_ > 1 ? config_.virtual_loss : 0.0f;
        virtual_loss.visits = static_cast<int32_t>(std::ceil(strength));
        virtual_loss.value_sum = static_cast<int64_t>(
            std::llround(static_cast<double>(strength) * VALUE_SCALE));
    }
    for (int gi = 0; gi < batch; ++gi) {
        games[gi].node_capacity = arena_capacity;
        games[gi].nodes = std::make_unique<GNode[]>(arena_capacity);
        reset_tree(games[gi]);
    }

    const bool profile = std::getenv("GOMOK_PROFILE") != nullptr;
    const char* cache_entries_text = std::getenv("GOMOK_CACHE_ENTRIES");
    const size_t max_cache_entries = cache_entries_text != nullptr
        ? static_cast<size_t>(std::max(1, std::atoi(cache_entries_text)))
        : 32768;
    std::unordered_map<uint64_t, size_t> evaluation_cache_index;
    evaluation_cache_index.reserve(max_cache_entries);
    std::vector<uint64_t> evaluation_cache_keys;
    std::vector<CachedEvaluation> evaluation_cache_values;
    evaluation_cache_keys.reserve(max_cache_entries);
    evaluation_cache_values.reserve(max_cache_entries);
    size_t next_cache_slot = 0;
    auto cache_evaluation = [&](uint64_t key, const CachedEvaluation& result) {
        if (evaluation_cache_index.count(key)) return;
        if (evaluation_cache_values.size() < max_cache_entries) {
            const size_t slot = evaluation_cache_values.size();
            evaluation_cache_keys.push_back(key);
            evaluation_cache_values.push_back(result);
            evaluation_cache_index.emplace(key, slot);
        } else {
            evaluation_cache_index.erase(evaluation_cache_keys[next_cache_slot]);
            evaluation_cache_keys[next_cache_slot] = key;
            evaluation_cache_values[next_cache_slot] = result;
            evaluation_cache_index.emplace(key, next_cache_slot);
            next_cache_slot = (next_cache_slot + 1) % max_cache_entries;
        }
    };
    int64_t cache_hits = 0;
    int64_t total_evaluations = 0;
    int64_t total_retries = 0;
    int64_t total_terminal = 0;
    int inference_calls = 0;
    int largest_batch = 0;
    int64_t reused_playouts = 0;
    int64_t preserved_nodes = 0;
    int64_t promoted_roots = 0;
    int64_t promotion_cache_hits = 0;

    const char* seed_text = std::getenv("GOMOK_SEED");
    // Derive a per-call stream from the configured seed. Without the batch
    // counter a pinned GOMOK_SEED makes run_batch() replay identical games on
    // every call, so the replay buffer fills with duplicates and self-play
    // contributes no new information.
    const uint32_t base_seed = seed_text != nullptr
        ? static_cast<uint32_t>(std::strtoul(seed_text, nullptr, 10))
        : static_cast<uint32_t>(
              std::chrono::steady_clock::now().time_since_epoch().count());
    const auto seed = static_cast<uint32_t>(
        base_seed ^ (0x9e3779b9U * static_cast<uint32_t>(++batch_counter_)));
    // Only a game's single root expansion draws noise; move sampling happens
    // after worker join. Keeping its stream per game avoids worker-scheduling
    // dependent seeds and repeated noise at every new move.
    std::vector<std::mt19937> rngs;
    rngs.reserve(batch);
    for (int i = 0; i < batch; ++i)
        rngs.emplace_back(seed + 0x9e3779b9U * static_cast<uint32_t>(i + 1));

    const std::vector<int> worker_cpus = search_cpus(inference_threads_);
    const char* persistent_workers_setting =
        std::getenv("GOMOK_PERSISTENT_WORKERS");
    const bool persistent_workers = persistent_workers_setting != nullptr &&
        std::strcmp(persistent_workers_setting, "0") != 0;
    const char* wait_text = std::getenv("GOMOK_BATCH_WAIT_US");
    const int batch_wait_us = wait_text != nullptr
        ? std::max(0, std::atoi(wait_text)) : (accelerated_ ? 100 : 0);

    // Reuse the queues and inference buffers across moves.
    const int max_target = batch * n_playout_;
    const int max_inference_capacity = std::max(
        1, std::min(max_target, batch * leaf_batch_size_));
    std::unique_ptr<LeafTask[]> tasks(new LeafTask[max_target]);
    std::unique_ptr<int[]> ready_slots(new int[max_target]);
    std::unique_ptr<int[]> completion_slots(new int[max_target]);
    std::unique_ptr<std::atomic<uint8_t>[]> published(
        new std::atomic<uint8_t>[max_target]);
    std::vector<int> inference_slots;
    inference_slots.reserve(max_inference_capacity);
    std::vector<float> eval_states(
        static_cast<size_t>(max_inference_capacity) * 4 * BOARD_CELLS);
    std::vector<float> eval_policy(
        static_cast<size_t>(max_inference_capacity) * BOARD_CELLS);
    std::vector<float> eval_value(max_inference_capacity);
    std::vector<CachedEvaluation> fresh_evaluations(max_inference_capacity);
    struct PendingPromotion {
        int game = -1;
        uint32_t chosen = 0;
        PromotionMode mode = PromotionMode::Reset;
        uint64_t key = 0;
        int evaluation_slot = -1;
        CachedEvaluation evaluation;
    };
    std::vector<PendingPromotion> promotions;
    promotions.reserve(batch);
    SearchWorkerPool persistent_pool;
    if (persistent_workers) persistent_pool.start(mcts_threads_, worker_cpus);

    while (true) {
        int active_games = 0;
        int retained_playouts = 0;
        promotions.clear();
        for (int gi = 0; gi < batch; ++gi) {
            if (!games[gi].finished) {
                GameState& game = games[gi];
                if (!reuse_tree) reset_tree(game);
                const int visits = game.nodes[0].visits.load(
                    std::memory_order_relaxed);
                if (visits < 0 || visits > n_playout_ ||
                    game.in_flight.load(std::memory_order_relaxed) != 0)
                    throw std::logic_error("invalid promoted root search budget");
                game.issued.store(visits, std::memory_order_relaxed);
                retained_playouts += visits;
                ++active_games;
            }
        }
        if (active_games == 0) break;

        const int target = active_games * n_playout_;
        reused_playouts += retained_playouts;
        const int inference_capacity = std::max(1, std::min(
            target, batch * leaf_batch_size_));
        for (int i = 0; i < target; ++i)
            published[i].store(0, std::memory_order_relaxed);
        std::atomic<int> ready_tail{0};
        std::atomic<int> completion_head{0};
        std::atomic<int> completion_tail{0};
        std::atomic<int> completed{retained_playouts};
        std::atomic<uint64_t> claim_cursor{0};
        std::atomic<int64_t> retries{0};
        std::atomic<int64_t> terminal_playouts{0};
        std::atomic<bool> worker_failed{false};
        std::exception_ptr worker_error;
        std::mutex worker_error_mutex;

        SearchWorkerPool local_pool;
        SearchWorkerPool& worker_pool = persistent_workers
            ? persistent_pool : local_pool;
        if (!persistent_workers) local_pool.start(mcts_threads_, worker_cpus);
        // A failed inference must cancel the round before its stack-owned
        // queues disappear. The persistent threads remain for the next move.
        struct FinishRound {
            SearchWorkerPool& pool;
            std::atomic<bool>& stop;
            bool active = true;
            ~FinishRound() {
                if (active) {
                    stop.store(true, std::memory_order_release);
                    try { pool.wait(); } catch (...) {}
                }
            }
        } finish_round{worker_pool, worker_failed};
        worker_pool.launch([&](int, SearchWorkerScratch& scratch) {
                try {
                    auto& worker_legal_moves = scratch.legal_moves;
                    auto& worker_candidates = scratch.candidates;
                    if (worker_legal_moves.capacity() < BOARD_CELLS)
                        worker_legal_moves.reserve(BOARD_CELLS);
                    if (worker_candidates.capacity() < BOARD_CELLS)
                        worker_candidates.reserve(BOARD_CELLS);
                    auto complete_one = [&]() {
                        int head = completion_head.load(
                            std::memory_order_relaxed);
                        while (head < completion_tail.load(
                                           std::memory_order_acquire)) {
                            if (completion_head.compare_exchange_weak(
                                    head, head + 1, std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
                                LeafTask& completed_task =
                                    tasks[completion_slots[head]];
                                GameState& completed_game =
                                    games[completed_task.game];
                                expand_leaf(completed_game, completed_task,
                                            completed_task.evaluation,
                                            worker_legal_moves,
                                            worker_candidates,
                                            config_.max_candidates,
                                            config_.dirichlet_eps,
                                            config_.dirichlet_alpha,
                                            rngs[completed_task.game]);
                                backup_path(completed_game,
                                            completed_task.path,
                                            completed_task.evaluation.value,
                                            virtual_loss);
                                completed_game.in_flight.fetch_sub(
                                    1, std::memory_order_release);
                                completed.fetch_add(
                                    1, std::memory_order_release);
                                return true;
                            }
                        }
                        return false;
                    };
                    while (!worker_failed.load(std::memory_order_relaxed)) {
                        if (complete_one()) continue;
                        int game_index = -1;
                        int simulation = -1;
                        for (int attempt = 0; attempt < batch; ++attempt) {
                            const int candidate = static_cast<int>(
                                claim_cursor.fetch_add(1,
                                    std::memory_order_relaxed) % batch);
                            GameState& game = games[candidate];
                            if (game.finished || game.nodes[0].state.load(
                                    std::memory_order_acquire) == NODE_EXPANDING) continue;
                            int active = game.in_flight.load(std::memory_order_relaxed);
                            bool reserved = false;
                            while (active < leaf_batch_size_) {
                                if (game.in_flight.compare_exchange_weak(
                                        active, active + 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
                                    reserved = true;
                                    break;
                                }
                            }
                            if (!reserved) continue;
                            int current = game.issued.load(
                                std::memory_order_relaxed);
                            while (current < n_playout_) {
                                if (game.issued.compare_exchange_weak(
                                        current, current + 1,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
                                    game_index = candidate;
                                    simulation = current;
                                    break;
                                }
                            }
                            if (game_index >= 0) break;
                            game.in_flight.fetch_sub(1, std::memory_order_release);
                        }
                        if (game_index < 0) {
                            bool all_issued = true;
                            for (int gi = 0; gi < batch; ++gi) {
                                if (!games[gi].finished &&
                                    games[gi].issued.load(
                                        std::memory_order_relaxed) < n_playout_) {
                                    all_issued = false;
                                    break;
                                }
                            }
                            if (all_issued) {
                                if (completed.load(std::memory_order_acquire) >=
                                    target) break;
                                std::this_thread::yield();
                                continue;
                            }
                            std::this_thread::yield();
                            continue;
                        }

                        LeafTask& task = tasks[static_cast<size_t>(game_index) *
                                               n_playout_ + simulation];
                        task.game = game_index;
                        task.eval = -1;
                        while (!worker_failed.load(std::memory_order_acquire)) {
                            float terminal_value = 0.0f;
                            const SelectionResult selected = select_leaf(
                                games[game_index], c_puct_, task,
                                terminal_value, virtual_loss);
                            if (selected == SelectionResult::Retry) {
                                retries.fetch_add(1, std::memory_order_relaxed);
                                complete_one();
                                std::this_thread::yield();
                                continue;
                            }
                            if (selected == SelectionResult::Terminal) {
                                terminal_playouts.fetch_add(
                                    1, std::memory_order_relaxed);
                                backup_path(games[game_index], task.path,
                                            terminal_value, virtual_loss);
                                games[game_index].in_flight.fetch_sub(
                                    1, std::memory_order_release);
                                completed.fetch_add(1,
                                                    std::memory_order_release);
                            } else {
                                const int position = ready_tail.fetch_add(
                                    1, std::memory_order_relaxed);
                                if (position >= target)
                                    throw std::logic_error(
                                        "MCTS inference queue overflow");
                                ready_slots[position] = static_cast<int>(
                                    static_cast<size_t>(game_index) *
                                    n_playout_ + simulation);
                                published[position].store(
                                    1, std::memory_order_release);
                            }
                            break;
                        }
                    }
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(worker_error_mutex);
                        if (!worker_error) worker_error = std::current_exception();
                    }
                    worker_failed.store(true, std::memory_order_release);
                }
        }, worker_failed);

        int ready_head = 0;
        int completion_tail_local = 0;

        while (completed.load(std::memory_order_acquire) < target &&
               !worker_failed.load(std::memory_order_acquire)) {
            if (ready_head >= ready_tail.load(std::memory_order_acquire) ||
                !published[ready_head].load(std::memory_order_acquire)) {
                std::this_thread::yield();
                continue;
            }

            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::microseconds(batch_wait_us);
            int contiguous = 0;
            while (contiguous < inference_capacity) {
                const int tail = ready_tail.load(std::memory_order_acquire);
                while (ready_head + contiguous < tail &&
                       contiguous < inference_capacity &&
                       published[ready_head + contiguous].load(
                           std::memory_order_acquire)) {
                    ++contiguous;
                }
                if (contiguous >= inference_capacity ||
                    std::chrono::steady_clock::now() >= deadline) break;
                std::this_thread::yield();
            }
            if (contiguous == 0) continue;

            inference_slots.clear();
            for (int i = 0; i < contiguous; ++i)
                inference_slots.push_back(ready_slots[ready_head + i]);
            ready_head += contiguous;

            std::unordered_map<uint64_t, int> pending_evaluations;
            pending_evaluations.reserve(inference_slots.size());
            int eval_count = 0;
            for (int slot : inference_slots) {
                LeafTask& task = tasks[slot];
                const auto cached = evaluation_cache_index.find(task.key);
                if (cached != evaluation_cache_index.end()) {
                    ++cache_hits;
                    task.eval = -1;
                    continue;
                }
                const auto [pending, inserted] =
                    pending_evaluations.emplace(task.key, eval_count);
                task.eval = pending->second;
                if (inserted) {
                    task.board.encode_state(eval_states.data() +
                        static_cast<size_t>(eval_count) * 4 * BOARD_CELLS);
                    ++eval_count;
                } else {
                    ++cache_hits;
                }
            }

            if (eval_count > 0) {
                total_evaluations += eval_count;
                ++inference_calls;
                largest_batch = std::max(largest_batch, eval_count);
                evaluate_(eval_states.data(), eval_count, eval_policy.data(),
                          eval_value.data());
                for (const auto& entry : pending_evaluations) {
                    const int eval = entry.second;
                    CachedEvaluation& result = fresh_evaluations[eval];
                    result.value = eval_value[eval];
                    std::memcpy(result.policy.data(), eval_policy.data() +
                        static_cast<size_t>(eval) * BOARD_CELLS,
                        BOARD_CELLS * sizeof(float));
                }
            }

            for (int slot : inference_slots) {
                LeafTask& task = tasks[slot];
                const CachedEvaluation& evaluation = task.eval >= 0
                    ? fresh_evaluations[task.eval]
                    : evaluation_cache_values[
                          evaluation_cache_index.at(task.key)];
                task.evaluation = evaluation;
                completion_slots[completion_tail_local++] = slot;
                completion_tail.store(completion_tail_local,
                                      std::memory_order_release);
            }

            // Do not evict entries until this batch has copied its results.
            for (const auto& entry : pending_evaluations) {
                cache_evaluation(entry.first, fresh_evaluations[entry.second]);
            }
        }

        total_retries += retries.load(std::memory_order_relaxed);
        total_terminal += terminal_playouts.load(std::memory_order_relaxed);
        // A worker error also cancels workers still retrying an expanding leaf.
        worker_pool.wait();
        finish_round.active = false;
        if (worker_error) std::rethrow_exception(worker_error);
        if (completed.load(std::memory_order_relaxed) != target)
            throw std::logic_error("MCTS round completed the wrong playout count");

        for (int gi = 0; gi < batch; ++gi) {
            GameState& game = games[gi];
            if (game.finished) continue;
            GNode& root = game.nodes[0];
            if (root.state.load(std::memory_order_acquire) != NODE_EXPANDED ||
                root.child_count == 0) {
                throw std::logic_error("nonterminal self-play root has no moves");
            }

            // A negative override means "use the configured schedule", which
            // explores the opening and sharpens later moves.
            const float temp = fixed_temp >= 0.0f
                ? fixed_temp
                : config_.temperature_at(game.board.move_count());

            std::vector<float> weights(root.child_count);
            int child_visits = 0;
            for (int i = 0; i < root.child_count; ++i)
                child_visits += game.nodes[root.child_begin + i].visits.load(
                    std::memory_order_relaxed);
            if (child_visits == 0 && temp > 1e-6f) {
                // A one-playout budget only evaluates the root. Preserve its
                // prior instead of inventing a uniform search result.
                for (int i = 0; i < root.child_count; ++i)
                    weights[i] = game.nodes[root.child_begin + i].prior;
            } else if (temp <= 1e-6f) {
                int best = 0;
                for (int i = 1; i < root.child_count; ++i) {
                    if (game.nodes[root.child_begin + i].visits.load(
                            std::memory_order_relaxed) >
                        game.nodes[root.child_begin + best].visits.load(
                            std::memory_order_relaxed)) best = i;
                }
                weights[best] = 1.0f;
            } else {
                const double inv_temp = 1.0 / static_cast<double>(temp);
                std::vector<double> log_weights(root.child_count);
                for (int i = 0; i < root.child_count; ++i) {
                    const float visits = static_cast<float>(
                        game.nodes[root.child_begin + i].visits.load(
                            std::memory_order_relaxed));
                    log_weights[i] = visits > 0.0f
                        ? std::log(static_cast<double>(visits)) * inv_temp
                        : -std::numeric_limits<double>::infinity();
                }
                const double max_log = *std::max_element(log_weights.begin(),
                                                        log_weights.end());
                for (int i = 0; i < root.child_count; ++i)
                    weights[i] = std::exp(log_weights[i] - max_log);
            }
            std::discrete_distribution<int> distribution(
                weights.begin(), weights.end());
            const int pick = distribution(rngs[gi]);
            const uint32_t chosen = root.child_begin + pick;
            const int move = game.nodes[chosen].action;

            std::array<float, BOARD_CELLS> probs = root_policy(game);

            // Reserve a little probability mass for legal moves MCTS never
            // considered. Without this the target asserts p = 0 for positions
            // that are merely unvisited, which trains the policy to consider
            // them unplayable and removes any chance of MCTS correcting a
            // mis-prioritised branch later.
            //
            // The mass goes only to empty cells: spreading it over all 225
            // cells would assign probability to occupied, illegal moves and
            // teach the policy to keep mass on them.
            if (config_.unseen_move_smooth > 0.0f && root.child_count > 1) {
                int legal_count = 0;
                for (int m = 0; m < BOARD_CELLS; ++m)
                    if (game.board.is_empty(m)) ++legal_count;
                if (legal_count > 0) {
                    const float smooth = config_.unseen_move_smooth;
                    const float spread =
                        smooth / static_cast<float>(legal_count);
                    for (int m = 0; m < BOARD_CELLS; ++m) {
                        probs[m] = game.board.is_empty(m)
                            ? probs[m] * (1.0f - smooth) + spread
                            : 0.0f;
                    }
                }
            }
            std::array<float, 4 * BOARD_CELLS> state{};
            game.board.encode_state(state.data());
            game.states.push_back(state);
            game.probs.push_back(probs);
            game.players.push_back(game.board.current_player());
            game.board.play(move);

            const int winner = game.board.winner();
            if (winner != 0 || game.board.move_count() == BOARD_CELLS) {
                game.finished = true;
                game.winner = winner;
            } else if (reuse_tree) {
                PendingPromotion promotion;
                promotion.game = gi;
                promotion.chosen = chosen;
                promotion.mode = promotion_mode(game, chosen);
                if (promotion.mode == PromotionMode::Refresh)
                    promotion.key = board_key(game.board);
                promotions.push_back(std::move(promotion));
            }
        }

        // Resolve any promoted-root priors in one inference batch. Most are
        // already present in the cross-move evaluation cache because their
        // node was expanded during the previous move's search.
        std::unordered_map<uint64_t, int> missing_promotions;
        missing_promotions.reserve(promotions.size());
        int promotion_evaluations = 0;
        for (auto& promotion : promotions) {
            if (promotion.mode != PromotionMode::Refresh) continue;
            const auto cached = evaluation_cache_index.find(promotion.key);
            if (cached != evaluation_cache_index.end()) {
                promotion.evaluation = evaluation_cache_values[cached->second];
                ++cache_hits;
                ++promotion_cache_hits;
                continue;
            }
            const auto [entry, inserted] =
                missing_promotions.emplace(promotion.key, promotion_evaluations);
            promotion.evaluation_slot = entry->second;
            if (inserted) {
                if (promotion_evaluations >= max_inference_capacity)
                    throw std::logic_error("MCTS promotion inference batch overflow");
                games[promotion.game].board.encode_state(
                    eval_states.data() + static_cast<size_t>(promotion_evaluations) *
                        4 * BOARD_CELLS);
                ++promotion_evaluations;
            } else {
                ++cache_hits;
            }
        }
        if (promotion_evaluations > 0) {
            total_evaluations += promotion_evaluations;
            ++inference_calls;
            largest_batch = std::max(largest_batch, promotion_evaluations);
            evaluate_(eval_states.data(), promotion_evaluations,
                      eval_policy.data(), eval_value.data());
            for (const auto& entry : missing_promotions) {
                const int slot = entry.second;
                CachedEvaluation& result = fresh_evaluations[slot];
                result.value = eval_value[slot];
                std::memcpy(result.policy.data(),
                            eval_policy.data() + static_cast<size_t>(slot) * BOARD_CELLS,
                            BOARD_CELLS * sizeof(float));
            }
            for (const auto& entry : missing_promotions)
                cache_evaluation(entry.first, fresh_evaluations[entry.second]);
        }
        for (auto& promotion : promotions) {
            if (promotion.evaluation_slot >= 0)
                promotion.evaluation = fresh_evaluations[promotion.evaluation_slot];
            const CachedEvaluation* prior = promotion.mode == PromotionMode::Refresh
                ? &promotion.evaluation : nullptr;
            const uint32_t preserved = promote_subtree(
                games[promotion.game], promotion.chosen, promotion.mode, prior,
                config_.max_candidates, config_.dirichlet_eps,
                config_.dirichlet_alpha,
                rngs[promotion.game]);
            if (preserved > 0) {
                ++promoted_roots;
                preserved_nodes += preserved;
            }
        }
    }

    if (profile) {
        std::fprintf(stderr,
            "[selfplay-profile] evals=%lld calls=%d avg_batch=%.1f "
            "max_batch=%d cache_hits=%lld cache_hit_rate=%.3f cache_size=%zu "
            "retries=%lld terminal=%lld reused_playouts=%lld "
            "promoted_roots=%lld preserved_nodes=%lld "
            "promotion_cache_hits=%lld mcts_threads=%d tree_reuse=%d "
            "persistent_workers=%d\n",
            static_cast<long long>(total_evaluations), inference_calls,
            inference_calls == 0 ? 0.0
                : static_cast<double>(total_evaluations) / inference_calls,
            largest_batch, static_cast<long long>(cache_hits),
            total_evaluations + cache_hits == 0 ? 0.0
                : static_cast<double>(cache_hits) /
                      static_cast<double>(total_evaluations + cache_hits),
            evaluation_cache_values.size(),
            static_cast<long long>(total_retries),
            static_cast<long long>(total_terminal),
            static_cast<long long>(reused_playouts),
            static_cast<long long>(promoted_roots),
            static_cast<long long>(preserved_nodes),
            static_cast<long long>(promotion_cache_hits),
            mcts_threads_, reuse_tree ? 1 : 0,
            persistent_workers ? 1 : 0);
    }

    std::vector<std::vector<Sample>> result(batch);
    for (int gi = 0; gi < batch; ++gi) {
        const GameState& game = games[gi];
        for (size_t i = 0; i < game.states.size(); ++i) {
            Sample sample;
            sample.state = game.states[i];
            sample.probs = game.probs[i];
            sample.z = game.winner == 0 ? 0.0f
                       : (game.players[i] == game.winner ? 1.0f : -1.0f);
            result[gi].push_back(std::move(sample));
        }
    }
    return result;
}

}  // namespace gomoku

namespace gomoku {
namespace {

// Single-game MCTS used by the arena. It runs one search at a time, so it can
// reuse the same select/expand/backup primitives as the batched self-play
// path without any of the queueing machinery.
// One arena game's search state.
//
// Several of these are advanced together by play_match() so their leaf
// evaluations can be merged into a single batched inference call.
struct ArenaGame {
    GameState tree;
    LeafTask task;
    std::vector<int> legal_moves;
    std::vector<std::pair<float, int>> candidates;
    std::mt19937 rng;
    Board board;
    int playout_done = 0;
    bool candidate_is_black = true;
    bool finished = false;
    int winner = 0;

    ArenaGame(int playouts, int max_candidates, uint64_t seed)
        : rng(static_cast<uint32_t>(seed)) {
        const size_t capacity =
            1 + static_cast<size_t>(playouts) * max_candidates;
        tree.node_capacity = capacity;
        tree.nodes = std::make_unique<GNode[]>(capacity);
        legal_moves.reserve(BOARD_CELLS);
        candidates.reserve(BOARD_CELLS);
    }

    // Begins the search for the current position.
    void start_move() {
        reset_tree(tree);
        tree.board = board;
        playout_done = 0;
    }

    bool candidate_to_move() const {
        return (board.current_player() == 1) == candidate_is_black;
    }
};

// Most visited root move, or -1 when the root has no children.
int best_move(const GameState& game) {
    const GNode& root = game.nodes[0];
    if (root.child_count == 0) return -1;
    int best = 0;
    for (int i = 1; i < root.child_count; ++i) {
        if (game.nodes[root.child_begin + i].visits.load(
                std::memory_order_relaxed) >
            game.nodes[root.child_begin + best].visits.load(
                std::memory_order_relaxed))
            best = i;
    }
    return game.nodes[root.child_begin + best].action;
}

}  // namespace

int search_best_move(PureNet& net, const Board& board,
                     const SearchConfig& config) {
    std::array<float, 4 * BOARD_CELLS> state{};
    return search_best_move(board, config,
        [&](const Board& leaf, std::array<float, BOARD_CELLS>& policy, float& value) {
            leaf.encode_state(state.data());
            net.forward_batch_policy(state.data(), 1, policy.data(), &value);
        });
}

int search_best_move(const Board& board, const SearchConfig& config,
                     const PositionEvaluator& evaluate) {
    return search_position(board, config, evaluate).move;
}

SearchResult search_position(const Board& board, const SearchConfig& config,
                             const PositionEvaluator& evaluate) {
    validate_search_config(config);
    if (!evaluate) throw std::invalid_argument("MCTS requires a position evaluator");
    SearchResult result;
    if (board.game_over()) return result;
    // Human play wants the strongest move, not exploration.
    SearchConfig greedy = config;
    greedy.dirichlet_eps = 0.0f;
    greedy.temp_high = 0.0f;
    greedy.temp_low = 0.0f;

    const int playouts = std::max(1, greedy.n_playout);
    ArenaGame game(playouts, greedy.max_candidates, 20240816u);
    game.board = board;
    game.start_move();

    // No parallel paths exist in this search: virtual loss would introduce
    // phantom parent visits and change PUCT despite having no collisions.
    const VirtualLossWeights loss;
    while (game.playout_done < playouts) {
        LeafTask& task = game.task;
        float terminal_value = 0.0f;
        const SelectionResult selected =
            select_leaf(game.tree, greedy.c_puct, task, terminal_value, loss);
        if (selected == SelectionResult::Retry) continue;
        if (selected == SelectionResult::Terminal) {
            backup_path(game.tree, task.path, terminal_value, loss);
            ++game.playout_done;
            continue;
        }
        CachedEvaluation evaluation;
        evaluate(task.board, evaluation.policy, evaluation.value);
        ++result.evaluations;
        expand_leaf(game.tree, task, evaluation, game.legal_moves, game.candidates,
                    greedy.max_candidates, greedy.dirichlet_eps,
                    greedy.dirichlet_alpha, game.rng);
        backup_path(game.tree, task.path, evaluation.value, loss);
        ++game.playout_done;
    }

    result.move = best_move(game.tree);
    result.policy = root_policy(game.tree);
    result.completed_playouts = game.playout_done;
    return result;
}

MatchResult play_match(PureNet& candidate, PureNet& opponent,
                       const SearchConfig& config, int games) {
    return play_match(
        [&candidate](const float* states, int batch, float* policies, float* values) {
            candidate.forward_batch_policy(states, batch, policies, values);
        },
        [&opponent](const float* states, int batch, float* policies, float* values) {
            opponent.forward_batch_policy(states, batch, policies, values);
        }, config, games);
}

MatchResult play_match(const BatchPositionEvaluator& candidate,
                       const BatchPositionEvaluator& opponent,
                       const SearchConfig& config, int games) {
    MatchResult result;
    validate_search_config(config);
    if (games <= 0) return result;
    if (!candidate || !opponent)
        throw std::invalid_argument("arena requires both model evaluators");
    std::vector<int8_t> candidate_outcomes(static_cast<size_t>(games), 0);

    // Explore neither side during the arena: report the strongest move.
    SearchConfig arena = config;
    arena.dirichlet_eps = 0.0f;
    arena.temp_switch_move = 0;
    arena.temp_high = 0.0f;
    arena.temp_low = 0.0f;

    const int playouts = std::max(1, arena.n_playout);
    // Arena games batch independently; each individual tree stays serial.
    const VirtualLossWeights loss;

    // Games advanced in lockstep. Each round every unfinished game contributes
    // at most one leaf, and the leaves are grouped by side so each model gets
    // one batched call. The two models each receive roughly half this batch,
    // avoiding serial single-position inference for every arena game.
    const int kParallelGames = std::min(games, 32);
    const size_t per_state = 4 * BOARD_CELLS;

    std::vector<float> states, policies, values;
    std::vector<int> pending, group;

    for (int base = 0; base < games; base += kParallelGames) {
        const int n = std::min(kParallelGames, games - base);
        std::vector<std::unique_ptr<ArenaGame>> ctx;
        ctx.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            auto game = std::make_unique<ArenaGame>(
                playouts, arena.max_candidates,
                0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(base + i + 1));
            // Reuse the exact same opening for each colour-swapped pair.
            // Empty-board greedy games otherwise repeat just two trajectories
            // regardless of the requested match size.
            const int game_number = base + i;
            const char* seed_text = std::getenv("GOMOK_SEED");
            const uint32_t arena_seed = seed_text != nullptr
                ? static_cast<uint32_t>(std::strtoul(seed_text, nullptr, 10))
                : 20240816u;
            std::seed_seq opening_seed{arena_seed,
                static_cast<uint32_t>(game_number / 2), 0x85ebca6bU};
            std::mt19937 opening_rng(opening_seed);
            for (int ply = 0; ply < arena.eval_opening_moves; ++ply) {
                std::vector<int> opening_moves;
                const int middle = BOARD_SIZE / 2;
                for (int row = middle - 2; row <= middle + 2; ++row)
                    for (int col = middle - 2; col <= middle + 2; ++col) {
                        const int move = row * BOARD_SIZE + col;
                        if (game->board.is_empty(move)) opening_moves.push_back(move);
                    }
                std::uniform_int_distribution<size_t> choose(0, opening_moves.size() - 1);
                game->board.play(ply == 0 ? middle * BOARD_SIZE + middle
                                         : opening_moves[choose(opening_rng)]);
            }
            game->candidate_is_black = (game_number % 2 == 0);
            game->start_move();
            ctx.push_back(std::move(game));
        }

        int remaining = n;
        while (remaining > 0) {
            // Advance each unfinished game to its next inference request.
            pending.clear();
            for (int i = 0; i < n; ++i) {
                ArenaGame& game = *ctx[static_cast<size_t>(i)];
                if (game.finished) continue;
                bool needs_inference = false;
                while (game.playout_done < playouts) {
                    LeafTask& task =
                        game.task;
                    float terminal_value = 0.0f;
                    const SelectionResult selected = select_leaf(
                        game.tree, arena.c_puct, task, terminal_value, loss);
                    if (selected == SelectionResult::Retry) continue;
                    if (selected == SelectionResult::Terminal) {
                        backup_path(game.tree, task.path, terminal_value, loss);
                        ++game.playout_done;
                        continue;
                    }
                    needs_inference = true;
                    break;
                }
                if (needs_inference) pending.push_back(i);
            }

            if (pending.empty()) {
                // Every unfinished game finished this move's budget: play it.
                for (int i = 0; i < n; ++i) {
                    ArenaGame& game = *ctx[static_cast<size_t>(i)];
                    if (game.finished) continue;
                    const int move = best_move(game.tree);
                    if (move < 0 || !game.board.is_empty(move))
                        throw std::logic_error("nonterminal arena root has no legal move");
                    game.board.play(move);
                    const int winner = game.board.winner();
                    if (winner != 0 || game.board.move_count() == BOARD_CELLS) {
                        game.finished = true;
                        game.winner = winner;
                        --remaining;
                        ++result.games;
                        if (winner == 0) {
                            ++result.draws;
                        } else {
                            if (winner == 1) ++result.black_wins;
                            else ++result.white_wins;
                            if ((winner == 1) == game.candidate_is_black) {
                                ++result.wins;
                                candidate_outcomes[base + i] = 1;
                            } else {
                                ++result.losses;
                                candidate_outcomes[base + i] = -1;
                            }
                        }
                        continue;
                    }
                    game.start_move();
                }
                continue;
            }

            // Two batched calls, one per model, instead of one call per leaf.
            for (int side = 0; side < 2; ++side) {
                group.clear();
                for (int index : pending) {
                    const bool is_candidate =
                        ctx[static_cast<size_t>(index)]->candidate_to_move();
                    if ((is_candidate ? 0 : 1) == side) group.push_back(index);
                }
                if (group.empty()) continue;
                const int m = static_cast<int>(group.size());

                states.resize(static_cast<size_t>(m) * per_state);
                policies.resize(static_cast<size_t>(m) * BOARD_CELLS);
                values.resize(static_cast<size_t>(m));
                for (int j = 0; j < m; ++j) {
                    ArenaGame& game = *ctx[static_cast<size_t>(group[static_cast<size_t>(j)])];
                    LeafTask& task =
                        game.task;
                    task.board.encode_state(
                        states.data() + static_cast<size_t>(j) * per_state);
                }

                const auto& evaluate = (side == 0) ? candidate : opponent;
                evaluate(states.data(), m, policies.data(), values.data());

                for (int j = 0; j < m; ++j) {
                    ArenaGame& game = *ctx[static_cast<size_t>(group[static_cast<size_t>(j)])];
                    LeafTask& task =
                        game.task;
                    CachedEvaluation evaluation;
                    evaluation.value = values[static_cast<size_t>(j)];
                    std::memcpy(
                        evaluation.policy.data(),
                        policies.data() + static_cast<size_t>(j) * BOARD_CELLS,
                        BOARD_CELLS * sizeof(float));
                    expand_leaf(game.tree, task, evaluation, game.legal_moves,
                                game.candidates, arena.max_candidates,
                                arena.dirichlet_eps, arena.dirichlet_alpha,
                                game.rng);
                    backup_path(game.tree, task.path,
                                values[static_cast<size_t>(j)], loss);
                    ++game.playout_done;
                }
            }
        }
    }
    for (size_t game = 0; game + 1 < candidate_outcomes.size(); game += 2) {
        const int pair_outcome = candidate_outcomes[game] + candidate_outcomes[game + 1];
        if (pair_outcome > 0) ++result.pair_wins;
        else if (pair_outcome < 0) ++result.pair_losses;
        else ++result.pair_draws;
    }
    return result;
}

}  // namespace gomoku
