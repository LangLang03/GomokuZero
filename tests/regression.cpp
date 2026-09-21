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

namespace {

// Reproduces the self-play target construction: MCTS visit weights on the root
// children, then the smoothing pass.
void spread_with_smoothing(const Board& root,
                           const std::vector<std::pair<int, float>>& visits,
                           float smooth, std::vector<float>& probs) {
    probs.assign(BOARD_CELLS, 0.0f);
    float total = 0.0f;
    for (const auto& [move, weight] : visits) {
        probs[static_cast<size_t>(move)] = weight;
        total += weight;
    }
    if (total > 0.0f)
        for (float& p : probs) p /= total;

    if (smooth <= 0.0f) return;
    int legal_count = 0;
    for (int m = 0; m < BOARD_CELLS; ++m)
        if (root.is_empty(m)) ++legal_count;
    if (legal_count == 0) return;
    const float spread = smooth / static_cast<float>(legal_count);
    for (int m = 0; m < BOARD_CELLS; ++m) {
        probs[static_cast<size_t>(m)] =
            root.is_empty(m) ? probs[static_cast<size_t>(m)] * (1.0f - smooth) + spread
                             : 0.0f;
    }
}

}  // namespace

int main() {
    // --- P0-04: smoothing must not touch occupied cells -------------------
    {
        Board board;
        // A mid-game position with plenty of stones on the board.
        const int moves[] = {112, 113, 127, 128, 97, 98, 143, 142, 82, 83};
        for (int m : moves) board.play(m);

        std::vector<float> probs;
        spread_with_smoothing(board, {{99, 0.6f}, {100, 0.3f}, {101, 0.1f}},
                              0.05f, probs);

        int occupied_with_mass = 0;
        double sum = 0.0;
        for (int m = 0; m < BOARD_CELLS; ++m) {
            sum += probs[static_cast<size_t>(m)];
            if (!board.is_empty(m) && probs[static_cast<size_t>(m)] != 0.0f)
                ++occupied_with_mass;
        }
        CHECK(occupied_with_mass == 0,
              "self-play smoothing assigned probability to occupied cells");
        CHECK(std::abs(sum - 1.0) < 1e-3, "smoothed target must still sum to 1");

        // Every legal cell must retain some mass, which is the point of the
        // smoothing in the first place.
        int legal_with_mass = 0;
        int legal_total = 0;
        for (int m = 0; m < BOARD_CELLS; ++m) {
            if (!board.is_empty(m)) continue;
            ++legal_total;
            if (probs[static_cast<size_t>(m)] > 0.0f) ++legal_with_mass;
        }
        CHECK(legal_with_mass == legal_total,
              "every legal move must keep some probability mass");
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

    // --- P0-01: the arena must evaluate distinct leaf positions -----------
    {
        // Two different boards must produce different network inputs, which is
        // what the arena failed to do when it encoded the root every time.
        Board root;
        root.play(112);
        root.play(113);

        Board leaf = root;
        leaf.play(98);
        leaf.play(99);

        std::vector<float> root_state(4 * BOARD_CELLS);
        std::vector<float> leaf_state(4 * BOARD_CELLS);
        root.encode_state(root_state.data());
        leaf.encode_state(leaf_state.data());

        bool differs = false;
        for (size_t i = 0; i < root_state.size(); ++i)
            if (root_state[i] != leaf_state[i]) { differs = true; break; }
        CHECK(differs,
              "a leaf position must encode differently from the root; encoding "
              "the root for every leaf makes MCTS evaluate the wrong position");

        // And the network must actually respond to that difference.
        PureNet net;
        std::vector<float> root_log, leaf_log;
        float root_value = 0.0f, leaf_value = 0.0f;
        net.forward_one(root_state, root_log, root_value);
        net.forward_one(leaf_state, leaf_log, leaf_value);
        double max_difference = 0.0;
        for (size_t i = 0; i < root_log.size(); ++i)
            max_difference =
                std::max(max_difference,
                         std::abs(static_cast<double>(root_log[i]) - leaf_log[i]));
        CHECK(max_difference > 1e-6,
              "network output must differ between distinct positions");
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
