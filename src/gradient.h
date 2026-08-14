#pragma once
#include <vector>
#include <cstddef>

namespace gomoku {

// Hand-written backprop for the policy-value net.
//
// All layers store (weights, grads) in simple vectors. The net is tiny
// (~0.33M params), so gradient computation via direct formulas (no autograd)
// is fast and fully deterministic. No torch anywhere.
//
// Weight layout (PyTorch-compatible):
//   conv3x3: [out][in][3][3]   conv1x1: [out][in]
//   fc:      [out][in]         bias:    [out]
// Input image layout: [C][H][W], H=W=15.

constexpr int HW_INF = 225;   // 15*15

// ---- layer parameter + gradient holders ----
struct ConvLayer {
    int in_ch, out_ch, kh, kw;  // kh, kw: 3 or 1
    std::vector<float> w;       // [out][in][kh][kw]
    std::vector<float> b;       // [out]
    std::vector<float> gw;      // same shape as w
    std::vector<float> gb;      // same shape as b

    ConvLayer(int input_channels, int output_channels,
              int kernel_h, int kernel_w)
        : in_ch(input_channels), out_ch(output_channels),
          kh(kernel_h), kw(kernel_w) {}
};

struct FCLayer {
    int in_n, out_n;
    std::vector<float> w;       // [out][in]
    std::vector<float> b;       // [out]
    std::vector<float> gw;
    std::vector<float> gb;

    FCLayer(int inputs, int outputs) : in_n(inputs), out_n(outputs) {}
};

// ---- forward (with cache for backward) ----
// conv3x3 padding=1, stride=1; relu applied by caller via cached input
void conv3x3_forward(const ConvLayer& l, const std::vector<float>& in,
                     std::vector<float>& out);
void conv3x3_forward(const ConvLayer& l, const float* in,
                     std::vector<float>& out);
void conv1x1_forward(const ConvLayer& l, const std::vector<float>& in,
                     std::vector<float>& out);
void fc_forward(const FCLayer& l, const std::vector<float>& in,
                std::vector<float>& out);

// ---- backward ----
// in/out are the cached forward tensors; grad_out propagates to grad_in.
// Layer gradients are accumulated. The caller owns zeroing them before a batch.
// grad_out and grad_in must be different vectors.
void conv3x3_backward(ConvLayer& l, const std::vector<float>& in,
                      const std::vector<float>& grad_out,
                      std::vector<float>& grad_in);
void conv1x1_backward(ConvLayer& l, const std::vector<float>& in,
                      const std::vector<float>& grad_out,
                      std::vector<float>& grad_in);
void fc_backward(FCLayer& l, const std::vector<float>& in,
                 const std::vector<float>& grad_out,
                 std::vector<float>& grad_in);

// relu helpers (in-place on cached values)
inline void relu_forward(std::vector<float>& x) {
    for (auto& v : x) if (v < 0) v = 0;
}
// grad_out = dL/dy; grad_in = grad_out * (y > 0)  (y is post-relu cache)
void relu_backward(const std::vector<float>& y, std::vector<float>& grad);

}  // namespace gomoku
