#include "selfplay.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

using namespace gomoku;

int main(int argc, char** argv) {
    try {
        SearchConfig config;
        config.n_playout = argc > 1 ? std::atoi(argv[1]) : 128;
        config.batch_games = argc > 2 ? std::atoi(argv[2]) : 8;
        config.n_threads = 1;
        config.dirichlet_eps = 0.0f;
        config.temp_high = config.temp_low = 0.0f;
        validate_search_config(config);
        int evaluations = 0;
        int calls = 0;
        BatchPositionEvaluator evaluate = [&](const float* states, int batch,
                                               float* policies, float* values) {
            ++calls;
            evaluations += batch;
            for (int b = 0; b < batch; ++b) {
                const float* state = states + static_cast<size_t>(b) * 4 * BOARD_CELLS;
                const bool black = state[3 * BOARD_CELLS] > 0.5f;
                int scripted = -1;
                const int row = black ? 7 : 0;
                const int first = black ? 5 : 0;
                for (int col = first; col < first + 5; ++col) {
                    const int move = row * BOARD_SIZE + col;
                    if (state[move] == 0.0f && state[BOARD_CELLS + move] == 0.0f) {
                        scripted = move;
                        break;
                    }
                }
                if (scripted < 0) {
                    for (int move = 0; move < BOARD_CELLS; ++move)
                        if (state[move] == 0.0f && state[BOARD_CELLS + move] == 0.0f) {
                            scripted = move;
                            break;
                        }
                }
                values[b] = 0.0f;
                for (int move = 0; move < BOARD_CELLS; ++move)
                    policies[static_cast<size_t>(b) * BOARD_CELLS + move] = 1e-6f;
                if (scripted >= 0)
                    policies[static_cast<size_t>(b) * BOARD_CELLS + scripted] = 1.0f;
            }
        };
        const auto start = std::chrono::steady_clock::now();
        BatchedSelfPlay selfplay(config, evaluate);
        const auto games = selfplay.run_batch();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        size_t moves = 0;
        uint64_t digest = 1469598103934665603ULL;
        for (const auto& game : games) {
            moves += game.size();
            for (const auto& sample : game) {
                for (int cell = 0; cell < 3 * BOARD_CELLS; ++cell) {
                    if (sample.state[cell] != 0.0f) {
                        digest ^= static_cast<uint64_t>(cell + 1);
                        digest *= 1099511628211ULL;
                    }
                }
            }
        }
        std::printf("games=%zu moves=%zu evaluations=%d calls=%d seconds=%.3f "
                    "evaluations_per_move=%.3f trajectory=%016llx\n",
                    games.size(), moves, evaluations, calls, seconds,
                    moves ? static_cast<double>(evaluations) / moves : 0.0,
                    static_cast<unsigned long long>(digest));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "benchmark failed: %s\n", error.what());
        return 1;
    }
}
