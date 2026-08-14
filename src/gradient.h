#pragma once
#include <vector>
#include <cstddef>

namespace gomoku {

// Native forward and backward kernels. Tensor layout matches PyTorch:
//   conv3x3: [out][in][3][3]   conv1x1: [out][in]
//   fc:      [out][in]         bias:    [out]
// Activations use [C][15][15].

constexpr int HW_INF = 225;   // 15*15

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

// 3x3 convolution uses padding 1 and stride 1.
void conv3x3_forward(const ConvLayer& l, const std::vector<float>& in,
                     std::vector<float>& out);
void conv3x3_forward(const ConvLayer& l, const float* in,
                     std::vector<float>& out);
void conv1x1_forward(const ConvLayer& l, const std::vector<float>& in,
                     std::vector<float>& out);
void fc_forward(const FCLayer& l, const std::vector<float>& in,
                std::vector<float>& out);

// Gradients accumulate into the layer; the caller clears them per batch.
// grad_out and grad_in must not alias.
void conv3x3_backward(ConvLayer& l, const std::vector<float>& in,
                      const std::vector<float>& grad_out,
                      std::vector<float>& grad_in);
void conv1x1_backward(ConvLayer& l, const std::vector<float>& in,
                      const std::vector<float>& grad_out,
                      std::vector<float>& grad_in);
void fc_backward(FCLayer& l, const std::vector<float>& in,
                 const std::vector<float>& grad_out,
                 std::vector<float>& grad_in);

inline void relu_forward(std::vector<float>& x) {
    for (auto& v : x) if (v < 0) v = 0;
}
// y is the post-ReLU activation.
void relu_backward(const std::vector<float>& y, std::vector<float>& grad);

}  // namespace gomoku
