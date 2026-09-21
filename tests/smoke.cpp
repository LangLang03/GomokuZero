// Smoke test: board rules, network output invariants, and a small self-play
// batch. Backend agreement (CPU vs LibTorch) lives in backend_parity.cpp.
#include "game.h"
#include "network.h"
#include "selfplay.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using namespace gomoku;

namespace {

int failures = 0;

int fail(const char* message) {
    std::printf("smoke test failed: %s\n", message);
    return 1;
}

#define REQUIRE(cond, msg)                                                    \
    do {                                                                      \
        if (!(cond)) return fail(msg);                                        \
    } while (0)

}  // namespace

int main() {
    // --- Board rules -------------------------------------------------------
    {
        // Moves alternate, so the opponent stones must stay off the line
        // black is building.
        Board board;
        for (int col = 0; col < 4; ++col) {
            board.play(col);                                // black row 0
            board.play(10 * BOARD_SIZE + col);              // white row 10
        }
        board.play(4);                                      // black completes row 0
        REQUIRE(board.winner() == 1, "horizontal five");
        board.undo();
        REQUIRE(board.winner() == 0 && board.move_count() == 8, "undo after a win");

        Board vertical;
        for (int row = 0; row < 4; ++row) {
            vertical.play(row * BOARD_SIZE);
            vertical.play(row * BOARD_SIZE + 10);
        }
        vertical.play(4 * BOARD_SIZE);
        REQUIRE(vertical.winner() == 1, "vertical five");

        Board diagonal;
        for (int i = 0; i < 4; ++i) {
            diagonal.play(i * BOARD_SIZE + i);
            diagonal.play(i * BOARD_SIZE + i + 10);
        }
        diagonal.play(4 * BOARD_SIZE + 4);
        REQUIRE(diagonal.winner() == 1, "diagonal five");

        Board anti_diagonal;
        for (int i = 0; i < 4; ++i) {
            anti_diagonal.play(i * BOARD_SIZE + (4 - i));
            anti_diagonal.play(i * BOARD_SIZE + 10);
        }
        anti_diagonal.play(4 * BOARD_SIZE);
        REQUIRE(anti_diagonal.winner() == 1, "anti-diagonal five");
    }

    // --- Network output invariants ----------------------------------------
    PureNet net;
    // The production trunk must carry real capacity; sanitizer builds
    // deliberately shrink it so a TSan/ASan self-play run finishes.
    if (!NetConfig::kShrunkForSanitizer())
        REQUIRE(net.parameter_count() > 1000000, "parameter count");
    REQUIRE(net.parameter_count() > 0, "parameter count must be non-zero");

    Board board;
    for (int col = 0; col < 4; ++col) {
        board.play(col);
        board.play(BOARD_SIZE + col);
    }
    board.play(4);
    board.undo();

    std::vector<float> state(4 * BOARD_CELLS);
    board.encode_state(state.data());

    std::vector<float> log_policy;
    float value = 0.0f;
    net.forward_one(state, log_policy, value);

    double policy_sum = 0.0;
    for (float logp : log_policy) policy_sum += std::exp(logp);
    REQUIRE(std::isfinite(value), "value is finite");
    REQUIRE(std::abs(policy_sum - 1.0) < 1e-4, "log_policy normalises");

    std::vector<float> direct_policy(BOARD_CELLS);
    float direct_value = 0.0f;
    net.forward_batch_policy(state.data(), 1, direct_policy.data(), &direct_value);
    float max_error = 0.0f;
    for (int i = 0; i < BOARD_CELLS; ++i)
        max_error = std::max(max_error,
                             std::abs(direct_policy[i] - std::exp(log_policy[i])));
    REQUIRE(max_error < 2e-4f, "probability path consistency");
    REQUIRE(std::abs(direct_value - value) < 2e-4f, "probability path value");

    // --- Self-play ---------------------------------------------------------
    SearchConfig search;
    search.n_playout = 50;
    search.batch_games = 2;
    search.n_threads = 1;
    search.dirichlet_eps = 0.25f;
    search.unseen_move_smooth = 0.05f;
    BatchedSelfPlay sp(net, search);

    auto samples = sp.run_batch(-1.0f);
    int total = 0;
    int multi_modal = 0;
    for (const auto& game : samples) {
        if (game.empty()) return fail("self-play empty game");
        total += static_cast<int>(game.size());
        for (const auto& s : game) {
            double sum = 0.0;
            int nonzero = 0;
            for (float p : s.probs) {
                sum += p;
                if (p > 1e-6f) ++nonzero;
            }
            REQUIRE(std::abs(sum - 1.0) < 1e-3, "self-play probs normalise");
            // Unvisited legal moves keep a small mass, so a typical target is
            // supported on many cells rather than a handful.
            if (nonzero > 10) ++multi_modal;
        }
    }
    REQUIRE(total >= 2, "self-play samples");
    REQUIRE(multi_modal > 0, "self-play targets must not be near one-hot");

    std::printf("smoke test passed (%d samples, parameters %zu)\n", total,
                net.parameter_count());
    return failures == 0 ? 0 : 1;
}
