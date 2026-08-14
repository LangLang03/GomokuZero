#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace gomoku {

// Hand-written CPU inference for the policy-value net (15x15 Gomoku).
//
// Replaces torch C++ forward for self-play: the net is tiny (~0.33M params,
// 3 small conv layers), so a direct CPU implementation is both faster (no
// torch kernel-launch overhead) and free of the torch multi-thread SEGVs we
// hit during long training runs.
//
// Weight layout matches PyTorch exactly (exported by tools/export_model.py
// as raw float32 bins): conv weight [out,in,kh,kw], fc weight [out,in].
class HandNet {
public:
    static constexpr int W = 15, H = 15;      // board
    static constexpr int N_IN = 4;            // input channels
    static constexpr int C1 = 32, C2 = 64, C3 = 128;  // conv channels
    static constexpr int N_OUT = W * H;       // 225 moves

    // load all 16 weight tensors from <dir>/<name>.bin
    bool load(const std::string& dir);

    // batch forward: states [B][4][15][15] (row-major) ->
    //   log_policy [B][225] (log softmax) and value [B][1] (tanh)
    // Uses OpenMP over the batch dimension.
    void forward(const float* states, int B,
                 float* log_policy, float* value) const;

    // single-state convenience
    void forward_one(const float* state, float* log_policy, float& value) const {
        forward(state, 1, log_policy, &value);
    }

    bool loaded() const { return loaded_; }

private:
    // conv layers (weights stored as [out][in][kh][kw] flattened)
    std::vector<float> conv1_w_, conv1_b_;   // 4->32, 3x3
    std::vector<float> conv2_w_, conv2_b_;   // 32->64, 3x3
    std::vector<float> conv3_w_, conv3_b_;   // 64->128, 3x3
    // policy head
    std::vector<float> act_conv1_w_, act_conv1_b_;  // 128->4, 1x1
    std::vector<float> act_fc1_w_, act_fc1_b_;      // 4*225 -> 225
    // value head
    std::vector<float> val_conv1_w_, val_conv1_b_;  // 128->2, 1x1
    std::vector<float> val_fc1_w_, val_fc1_b_;      // 2*225 -> 64
    std::vector<float> val_fc2_w_, val_fc2_b_;      // 64 -> 1

    bool loaded_ = false;
};

}  // namespace gomoku
