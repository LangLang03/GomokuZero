#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace gomoku {

// Reference CPU inference implementation.
// Weights use PyTorch order: conv [out,in,kh,kw], fc [out,in].
class HandNet {
public:
    static constexpr int W = 15, H = 15;
    static constexpr int N_IN = 4;
    static constexpr int C1 = 32, C2 = 64, C3 = 128;
    static constexpr int N_OUT = W * H;

    // Loads <dir>/<name>.bin.
    bool load(const std::string& dir);

    // states [B][4][15][15] -> log_policy [B][225], value [B]
    void forward(const float* states, int B,
                 float* log_policy, float* value) const;

    void forward_one(const float* state, float* log_policy, float& value) const {
        forward(state, 1, log_policy, &value);
    }

    bool loaded() const { return loaded_; }

private:
    // Trunk.
    std::vector<float> conv1_w_, conv1_b_;   // 4->32, 3x3
    std::vector<float> conv2_w_, conv2_b_;   // 32->64, 3x3
    std::vector<float> conv3_w_, conv3_b_;   // 64->128, 3x3
    // Policy head.
    std::vector<float> act_conv1_w_, act_conv1_b_;  // 128->4, 1x1
    std::vector<float> act_fc1_w_, act_fc1_b_;      // 4*225 -> 225
    // Value head.
    std::vector<float> val_conv1_w_, val_conv1_b_;  // 128->2, 1x1
    std::vector<float> val_fc1_w_, val_fc1_b_;      // 2*225 -> 64
    std::vector<float> val_fc2_w_, val_fc2_b_;      // 64 -> 1

    bool loaded_ = false;
};

}  // namespace gomoku
