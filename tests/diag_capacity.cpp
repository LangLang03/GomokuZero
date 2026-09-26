// Architecture and output diagnostics for the model that is actually loaded.
// Random-board entropy cannot diagnose overfitting; compare held-out game loss
// and arena results for that purpose.
#include "game.h"
#include "network.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights-dir> [states=32]\n", argv[0]);
        return 2;
    }
    const int count = argc > 2 ? std::atoi(argv[2]) : 32;
    if (count < 1 || count > 100000) return 2;
    gomoku::PureNet net;
    if (!net.load(argv[1])) {
        std::fprintf(stderr, "cannot load model: %s\n", argv[1]);
        return 1;
    }
    using Config = gomoku::NetConfig;
    std::printf("architecture: residual blocks=%d channels=%d parameters=%zu\n",
                Config::kResBlocks, Config::kChannels, net.parameter_count());
    std::printf("trunk receptive field=%dx%d; board=%dx%d\n",
                Config::receptive_field(), Config::receptive_field(),
                Config::kBoardSize, Config::kBoardSize);
    std::printf("policy head: 1x1 C->2, dense %d->%d; value hidden=%d\n",
                Config::kPolicyFcIn, Config::kPolicyOut, Config::kValueHidden);
    std::mt19937 rng(12345);
    std::vector<float> state(4 * Config::kCells), log_policy;
    double entropy = 0.0, illegal_mass = 0.0, value_mean = 0.0;
    int saturated = 0;
    for (int sample = 0; sample < count; ++sample) {
        gomoku::Board board;
        for (int move = 0; move < 2 + sample % 24; ++move) {
            const auto legal = board.legal_moves();
            if (legal.empty() || board.winner() != 0) break;
            board.play(legal[rng() % legal.size()]);
        }
        board.encode_state(state.data());
        float value = 0.0f;
        net.forward_one(state, log_policy, value);
        if (!std::isfinite(value)) return 1;
        for (int cell = 0; cell < Config::kCells; ++cell) {
            const double probability = std::exp(log_policy[cell]);
            if (!std::isfinite(probability)) return 1;
            entropy -= probability * log_policy[cell];
            if (state[cell] != 0.0f || state[Config::kCells + cell] != 0.0f)
                illegal_mass += probability;
        }
        value_mean += value;
        saturated += std::abs(value) > 0.99f;
    }
    std::printf("random legal positions=%d; mean entropy=%.5f (uniform=%.5f)\n",
                count, entropy / count, std::log(static_cast<double>(Config::kCells)));
    std::printf("raw policy mass on occupied cells=%.5f; mean value=%.5f; |value|>0.99=%.2f%%\n",
                illegal_mass / count, value_mean / count, 100.0 * saturated / count);
    std::puts("Output statistics are probes, not evidence of playing strength or generalisation.");
    return 0;
}
