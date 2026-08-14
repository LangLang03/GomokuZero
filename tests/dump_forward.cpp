// Dumps a fixed forward pass for ONNX comparison.
#include "network.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace gomoku;

static void write_float_file(const char* path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

static void make_state(std::vector<float>& state) {
    // Same channel layout as export_onnx.py.
    state.assign(4 * 225, 0.0f);
    int own[] = {7, 112, 38, 157, 83, 166, 19, 130, 55, 144};
    int opp[] = {8, 113, 39, 156, 84, 165, 20, 131, 56, 143};
    for (int i : own) state[0 * 225 + i] = 1.0f;
    for (int i : opp) state[1 * 225 + i] = 1.0f;
    state[2 * 225 + 144] = 1.0f;  // last move
    for (int i = 0; i < 225; ++i) state[3 * 225 + i] = 1.0f;  // black to move
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dump_forward <weights-dir> <out-prefix>\n");
        return 2;
    }
    PureNet net;
    if (!net.load(argv[1])) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }
    std::vector<float> state;
    make_state(state);

    std::vector<float> policy(225);
    float value = 0.0f;

    net.set_fast_inference(true);
    net.set_quantized_inference(false);
    net.forward_one(state, policy, value);
    write_float_file((std::string(argv[2]) + ".state.bin").c_str(), state.data(), state.size());
    write_float_file((std::string(argv[2]) + ".policy.bin").c_str(), policy.data(), policy.size());
    write_float_file((std::string(argv[2]) + ".value.bin").c_str(), &value, 1);

    std::vector<float> policy_q(225);
    float value_q = 0.0f;
    net.set_quantized_inference(true);
    net.forward_one(state, policy_q, value_q);
    write_float_file((std::string(argv[2]) + ".policy_q.bin").c_str(), policy_q.data(), policy_q.size());
    write_float_file((std::string(argv[2]) + ".value_q.bin").c_str(), &value_q, 1);

    std::printf("value=%.6f value_q=%.6f\n", value, value_q);
    int best = 0;
    for (int i = 1; i < 225; ++i)
        if (policy[i] > policy[best]) best = i;
    std::printf("argmax=%d (row %d col %d)\n", best, best / 15, best % 15);
    return 0;
}
