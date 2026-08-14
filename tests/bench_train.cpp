// Five batch-512 training steps.
// Usage: bench_train <weights-dir> [inference-threads]
#include "network.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace gomoku;

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    PureNet net(2e-3);
    if (!net.load(argv[1])) return 1;
    if (argc > 2) net.set_inference_threads(std::atoi(argv[2]));

    const int B = 512;
    std::vector<float> states(B * 900), probs(B * 225), winners(B);
    std::mt19937 rng(7);
    for (int i = 0; i < B * 900; ++i)
        states[i] = (rng() % 7 == 0) ? 1.0f : 0.0f;
    for (float& p : probs) p = 1.0f / 225.0f;
    for (int i = 0; i < B; ++i) winners[i] = (rng() & 1) ? 1.0f : -1.0f;

    net.train_step(states, B, probs, winners, 2e-3);
    auto t0 = std::chrono::steady_clock::now();
    PureNet::TrainStats s{};
    for (int e = 0; e < 5; ++e) s = net.train_step(states, B, probs, winners, 2e-3);
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("train B=512 x5: %.1f ms total -> %.1f ms/step (loss=%.4f entropy=%.4f)\n",
                ms, ms / 5, s.loss, s.entropy);
    std::fflush(stdout);
    return 0;
}
