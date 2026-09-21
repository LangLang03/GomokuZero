// Prints the native forward pass for a fixed position, layer by layer.
// Used to localise divergence against an independent implementation.
// Build: g++ -O2 -std=c++17 -I src -DGOMOK_DUMP_FORWARD \
//        -o /tmp/forward_dump tests/forward_dump.cpp src/network.cpp \
//        src/gradient.cpp src/game.cpp src/logger.cpp src/network_cuda_stub.cpp
#include "game.h"
#include "network.h"
#include <cstdio>
#include <vector>
using namespace gomoku;

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <weights-dir>\n", argv[0]); return 2; }
    PureNet net;
    if (!net.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    net.set_cuda_enabled(false);
    // Deterministic position: black at 112,113,114 / white at 127,128.
    std::vector<float> state(4 * BOARD_CELLS, 0.0f);
    state[0 * BOARD_CELLS + 112] = 1.0f;
    state[0 * BOARD_CELLS + 113] = 1.0f;
    state[0 * BOARD_CELLS + 114] = 1.0f;
    state[1 * BOARD_CELLS + 127] = 1.0f;
    state[1 * BOARD_CELLS + 128] = 1.0f;
    state[2 * BOARD_CELLS + 128] = 1.0f;
    for (int i = 0; i < BOARD_CELLS; ++i) state[3 * BOARD_CELLS + i] = 1.0f;
    std::vector<float> log_policy;
    float value = 0.0f;
    net.forward_one(state, log_policy, value);
    std::printf("value=%.8f\n", value);
    std::printf("logp[0]=%.8f logp[112]=%.8f logp[113]=%.8f\n",
                log_policy[0], log_policy[112], log_policy[113]);
    return 0;
}
