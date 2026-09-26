#include "network.h"
#include "game.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

// End-to-end timings include host/device transfers and synchronisation.
// Usage: gomoku_backend_benchmark [batch=32] [steps=12] [weights-dir]
int main(int argc, char** argv) {
    const int batch = argc > 1 ? std::atoi(argv[1]) : 32;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 12;
    if (batch <= 0 || steps <= 0) return 2;
    gomoku::PureNet net;
    if (!net.using_cuda()) return 77;
    if (const char* amp = std::getenv("GOMOK_BENCH_BF16"))
        net.set_mixed_precision(std::string(amp) == "1");
    if (argc > 3 && !net.load(argv[3])) return 2;
    if (const char* destination = std::getenv("GOMOK_BENCH_EXPORT"))
        if (!net.export_inference(destination)) return 2;
    std::printf("backend=cuda,channels=%d,blocks=%d,bf16=%d,fused_adam=%s,tf32_training=%s,tf32_inference=%s,pinned_inference=%s,seed=%s\n",
                gomoku::NetConfig::kChannels, gomoku::NetConfig::kResBlocks,
                net.mixed_precision(),
                std::getenv("GOMOK_FUSED_ADAM") ? std::getenv("GOMOK_FUSED_ADAM") : "1",
                std::getenv("GOMOK_TF32") ? std::getenv("GOMOK_TF32") : "1",
                std::getenv("GOMOK_INFERENCE_TF32") ? std::getenv("GOMOK_INFERENCE_TF32") : "1",
                std::getenv("GOMOK_PINNED_INFERENCE") ? std::getenv("GOMOK_PINNED_INFERENCE") : "0",
                std::getenv("GOMOK_SEED") ? std::getenv("GOMOK_SEED") : "random");
    const int cells = gomoku::NetConfig::kCells;
    std::vector<float> states(static_cast<size_t>(batch) * 4 * cells, 0.0f);
    std::vector<float> targets(static_cast<size_t>(batch) * cells, 0.0f);
    std::vector<float> winners(batch), policy(targets.size()), value(batch);
    std::mt19937 rng(20260926);
    for (int b = 0; b < batch; ++b) {
        gomoku::Board board;
        for (int i = 0; i < 8 + b % 12 && !board.game_over(); ++i) {
            const auto legal = board.legal_moves();
            board.play(legal[rng() % legal.size()]);
        }
        board.encode_state(states.data() + static_cast<size_t>(b) * 4 * cells);
        const auto legal = board.legal_moves();
        targets[static_cast<size_t>(b) * cells + legal[rng() % legal.size()]] = 1.0f;
        winners[b] = b % 2 ? 1.0f : -1.0f;
    }
    using Clock = std::chrono::steady_clock;
    auto measure = [&](const char* phase, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        net.synchronize();
        const auto begin = Clock::now();
        for (int i = 0; i < steps; ++i) fn();
        net.synchronize();
        const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        std::printf("%s,batch=%d,steps=%d,ms=%.3f,samples_per_second=%.3f\n",
                    phase, batch, steps, 1000.0 * seconds / steps, batch * steps / seconds);
        std::fflush(stdout);
    };
    measure("inference", [&] { net.forward_batch(states.data(), batch, policy.data(), value.data()); });
    gomoku::PureNet::TrainStats metrics{};
    float initial_loss = 0.0f;
    int updates = 0;
    measure("training", [&] {
        metrics = net.train_step(states, batch, targets, winners, 1e-4);
        if (updates++ == 0) initial_loss = metrics.loss;
    });
    std::printf("initial_loss=%.6f,final_loss=%.6f,policy=%.6f,value=%.6f,entropy=%.6f\n",
                initial_loss, metrics.loss, metrics.policy_loss, metrics.value_loss, metrics.entropy);
    return std::isfinite(metrics.loss) ? 0 : 1;
}
