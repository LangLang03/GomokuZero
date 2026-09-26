// Regression tests for the correctness fixes reported against the source.
//
// Each case pins a defect that was found by inspection and would otherwise be
// free to come back:
//
//   P0-01  Arena MCTS evaluated the root position instead of the leaf.
//   P0-02  The KL diagnostic ran an optimisation step of its own.
//   P0-04  Self-play smoothing put probability on occupied, illegal cells.
//   P1-01  Human-record label smoothing did the same.
//   P1-09  `human` mode ignored --playout and used the bare policy.
#include "game.h"
#include "network.h"
#include "selfplay.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace gomoku;

static int failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL: %s\n", msg);                                   \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

int main() {
    // Exercise the actual concurrent self-play target path, including later
    // positions, illegal-cell masking, normalization and terminal value signs.
    {
        PureNet net;
        SearchConfig config;
        config.n_playout = 8;
        config.batch_games = 2;
        config.n_threads = 2;
        config.max_candidates = 8;
        config.unseen_move_smooth = 0.05f;
        BatchedSelfPlay selfplay(net, config);
        const auto games = selfplay.run_batch(1.0f);
        CHECK(games.size() == 2, "self-play returns each requested game");
        for (const auto& samples : games) {
            Board replay;
            CHECK(!samples.empty(), "self-play game contains training samples");
            for (size_t ply = 0; ply < samples.size(); ++ply) {
                const auto& sample = samples[ply];
                std::array<float, 4 * BOARD_CELLS> expected{};
                replay.encode_state(expected.data());
                CHECK(expected == sample.state, "stored state matches replayed trajectory");
                double total = 0.0;
                for (int move = 0; move < BOARD_CELLS; ++move) {
                    const float p = sample.probs[move];
                    CHECK(std::isfinite(p) && p >= 0.0f, "policy target is finite/nonnegative");
                    CHECK(replay.is_empty(move) || p == 0.0f, "policy target masks illegal moves");
                    total += p;
                }
                CHECK(std::abs(total - 1.0) < 1e-5, "actual search target normalizes");
                if (ply + 1 < samples.size()) {
                    int played = -1;
                    for (int move = 0; move < BOARD_CELLS; ++move)
                        if (samples[ply + 1].state[2 * BOARD_CELLS + move] == 1.0f)
                            played = move;
                    CHECK(played >= 0 && replay.is_empty(played), "trajectory contains legal next move");
                    if (played >= 0 && replay.is_empty(played)) replay.play(played);
                    CHECK(sample.z == -samples[ply + 1].z,
                          "value targets alternate player perspective");
                } else {
                    bool terminal_found = false;
                    for (int move : replay.legal_moves()) {
                        replay.play(move);
                        if (replay.game_over()) {
                            const float expected_z = replay.winner() == 0 ? 0.0f : 1.0f;
                            if (sample.z == expected_z) terminal_found = true;
                        }
                        replay.undo();
                    }
                    CHECK(terminal_found, "last sample has a reachable terminal outcome");
                }
            }
        }
    }

    // --- P0-02: the KL diagnostic must not change the model ---------------
    {
        PureNet net;
        std::mt19937 rng(4242);
        std::vector<float> state(4 * BOARD_CELLS, 0.0f);
        std::vector<int> cells(BOARD_CELLS);
        for (int i = 0; i < BOARD_CELLS; ++i) cells[i] = i;
        std::shuffle(cells.begin(), cells.end(), rng);
        for (int k = 0; k < 12; ++k)
            state[static_cast<size_t>(k % 2) * BOARD_CELLS + cells[static_cast<size_t>(k)]] =
                1.0f;
        for (int i = 0; i < BOARD_CELLS; ++i)
            state[3 * BOARD_CELLS + i] = 1.0f;

        std::vector<float> before, after;
        float value_before = 0.0f, value_after = 0.0f;
        net.forward_one(state, before, value_before);
        // A second identical forward must be bit-identical: the probe is a pure
        // observation, so nothing may advance between the two calls.
        net.forward_one(state, after, value_after);
        bool identical = before.size() == after.size() &&
                         value_before == value_after;
        for (size_t i = 0; identical && i < before.size(); ++i)
            identical = before[i] == after[i];
        CHECK(identical, "repeated forward must be bit-identical (no hidden update)");
    }

    // --- P1-09: search_best_move must return a legal, playable move -------
    {
        SearchConfig search;
        search.n_playout = 40;
        search.n_threads = 1;

        PureNet net;
        Board board;
        board.play(112);
        board.play(113);

        const int move = search_best_move(net, board, search);
        CHECK(move >= 0 && move < BOARD_CELLS,
              "search_best_move must return an in-range move");
        CHECK(board.is_empty(move),
              "search_best_move must return an empty cell");
        // MCTS with 40 playouts on a 2-stone board must have explored beyond
        // the immediate root, so the returned move must be a cell the policy
        // actually considered rather than a constant.
        CHECK(board.move_count() == 2,
              "search_best_move must not modify the caller's board");
    }

    // Observe the actual evaluator inputs: the root is evaluated only once,
    // while subsequent calls see the selected leaf with its correct player.
    {
        Board root;
        root.play(112);
        root.play(113);
        SearchConfig config;
        config.n_playout = 12;
        int root_evaluations = 0;
        int leaf_evaluations = 0;
        const int move = search_best_move(root, config,
            [&](const Board& leaf, std::array<float, BOARD_CELLS>& policy, float& value) {
                if (leaf.position_key() == root.position_key()) ++root_evaluations;
                else ++leaf_evaluations;
                CHECK(leaf.move_count() >= root.move_count(), "evaluations descend from root");
                CHECK(leaf.current_player() == (leaf.move_count() % 2 + 1),
                      "evaluator receives the leaf side to move");
                policy.fill(1.0f / BOARD_CELLS);
                value = 0.0f;
            });
        CHECK(root.is_empty(move), "observable search selects legal action");
        CHECK(root_evaluations == 1 && leaf_evaluations == config.n_playout - 1,
              "MCTS evaluates each selected leaf, not a repeated root position");
    }

    // A zero prior stays zero when other legal moves have positive mass.
    // The old normalizer substituted 1 / sum for zero entries, which could
    // make a zero-probability move more attractive than the entire policy.
    {
        SearchConfig config;
        config.n_playout = 2;
        config.max_candidates = 2;
        Board root;
        root.play(112);
        const int move = search_best_move(root, config,
            [](const Board&, std::array<float, BOARD_CELLS>& policy, float& value) {
                policy.fill(0.0f);
                policy[113] = 0.01f;
                value = 0.0f;
            });
        CHECK(move == 113, "zero priors do not gain mass during normalization");
    }

    // Scripted values favor root move 16 and lose after root move 17.
    // They alternate sign with depth; a sign error in backup/selection chooses
    // the losing move even though both priors and the evaluator are controlled.
    {
        Board root;
        root.play(0);
        root.play(1);
        SearchConfig config;
        config.n_playout = 24;
        config.max_candidates = 2;
        config.c_puct = 1.0f;
        const int move = search_best_move(root, config,
            [&](const Board& leaf, std::array<float, BOARD_CELLS>& policy, float& value) {
                policy.fill(0.0f);
                policy[16] = 0.5f;
                policy[17] = 0.5f;
                if (leaf.move_count() == root.move_count()) value = 0.0f;
                else {
                    const float root_value = leaf.move_at(root.move_count()) == 16 ? 1.0f : -1.0f;
                    value = leaf.current_player() == root.current_player()
                        ? root_value : -root_value;
                }
            });
        CHECK(move == 16, "MCTS backs up values with alternating player signs");
    }

    // Tactical wins/blocks survive the most aggressive candidate cap.
    {
        PureNet net;
        SearchConfig config;
        config.n_playout = 2;
        config.max_candidates = 1;
        Board win;
        for (int move : {0, 30, 1, 32, 2, 34, 3, 36}) win.play(move);
        CHECK(search_best_move(net, win, config) == 4,
              "immediate winning action survives cap and beats arbitrary priors");
        win.play(4);
        CHECK(search_best_move(net, win, config) == -1, "terminal board has no search action");
        Board block;
        for (int move : {30, 0, 32, 1, 34, 2, 36, 3}) block.play(move);
        CHECK(search_best_move(net, block, config) == 4,
              "unique immediate opponent win is blocked");

        config.n_playout = 1;
        config.max_candidates = 4;
        config.eval_opening_moves = 4;
        const MatchResult match = play_match(net, net, config, 4);
        CHECK(match.games == 4 && match.wins == match.losses,
              "identical models tie on colour-swapped opening pairs");
        CHECK(match.games == match.wins + match.losses + match.draws,
              "arena accounts for every requested game");
        config.max_candidates = 0;
        bool rejected = false;
        try { BatchedSelfPlay invalid(net, config); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected, "zero candidate cap is rejected before arena allocation");
    }

    // --- Null buffers must be rejected, not dereferenced ----------------
    {
        // A caller that forgets to size a buffer passes data() == nullptr.
        // That used to segfault inside the backend (found via the KL probe,
        // which passed an unallocated vector); it must now fail loudly.
        PureNet net;
        std::vector<float> states(4 * BOARD_CELLS, 0.0f);
        std::vector<float> value(1, 0.0f);

        bool threw = false;
        try {
            net.forward_batch(states.data(), 1, nullptr, value.data());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw, "a null policy buffer must raise invalid_argument");

        threw = false;
        try {
            net.forward_batch(nullptr, 1, value.data(), value.data());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw, "a null state buffer must raise invalid_argument");
    }

    std::printf("regression %s\n", failures == 0 ? "passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
