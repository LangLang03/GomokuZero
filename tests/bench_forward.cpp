#include "network.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace gomoku;
int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: bench <dir> <B> <mode: fp|q8|naive>\n"); return 2; }
    PureNet net;
    if (!net.load(argv[1])) return 1;
    const int B = std::atoi(argv[2]);
    const std::string mode = argv[3];
    std::vector<float> states(B * 900, 0.0f), policy(B * 225), value(B);
    for (int i = 0; i < B * 900; ++i) states[i] = (i % 7 == 0) ? 1.0f : 0.0f;
    if (mode == "fp")   { net.set_fast_inference(true);  net.set_quantized_inference(false); }
    if (mode == "q8")   { net.set_fast_inference(true);  net.set_quantized_inference(true);  }
    if (mode == "naive"){ net.set_fast_inference(false); net.set_quantized_inference(false); }
    for (int i = 0; i < 2; ++i) net.forward_batch(states.data(), B, policy.data(), value.data());
    auto t0 = std::chrono::steady_clock::now();
    const int iters = mode == "naive" ? 1 : 10;
    for (int i = 0; i < iters; ++i) net.forward_batch(states.data(), B, policy.data(), value.data());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("%s B=%d: %.2f ms -> %.1f us/state -> %.0f states/s\n", mode.c_str(), B, ms, ms * 1000 / B, B * 1000.0 / ms);
    std::fflush(stdout);
    return 0;
}
