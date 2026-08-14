#include "gradient.h"
#include <algorithm>
#include <cmath>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <immintrin.h>
#define GOMOK_X86_AVX2 1
#else
#define GOMOK_X86_AVX2 0
#endif

namespace gomoku {
namespace {

constexpr int SIDE = 15;
constexpr int CELLS = SIDE * SIDE;

#if GOMOK_X86_AVX2
bool use_avx2_fma() {
    static const bool supported = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") &&
               __builtin_cpu_supports("fma");
    }();
    return supported;
}

__attribute__((target("avx2,fma")))
float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
float sum_avx(const float* values, int count) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= count; i += 8)
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(values + i));
    float result = horizontal_sum(sum);
    for (; i < count; ++i) result += values[i];
    return result;
}

__attribute__((target("avx2,fma")))
void conv3x3_forward_avx(const ConvLayer& layer, const float* input,
                         std::vector<float>& output) {
    output.resize(static_cast<size_t>(layer.out_ch) * CELLS);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        float* out = output.data() + static_cast<size_t>(out_channel) * CELLS;
        std::fill(out, out + CELLS, layer.b[out_channel]);
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch * 9;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input + static_cast<size_t>(in_channel) * CELLS;
            const float* kernel = weights + static_cast<size_t>(in_channel) * 9;
            for (int kr = 0; kr < 3; ++kr) {
                const int row_begin = std::max(0, 1 - kr);
                const int row_end = std::min(SIDE, SIDE + 1 - kr);
                for (int kc = 0; kc < 3; ++kc) {
                    const int col_begin = std::max(0, 1 - kc);
                    const int col_end = std::min(SIDE, SIDE + 1 - kc);
                    const __m256 weight = _mm256_set1_ps(kernel[kr * 3 + kc]);
                    for (int row = row_begin; row < row_end; ++row) {
                        float* destination = out + row * SIDE + col_begin;
                        const float* source = in + (row + kr - 1) * SIDE +
                                              col_begin + kc - 1;
                        const int length = col_end - col_begin;
                        int col = 0;
                        for (; col + 8 <= length; col += 8) {
                            const __m256 x = _mm256_loadu_ps(source + col);
                            const __m256 y = _mm256_loadu_ps(destination + col);
                            _mm256_storeu_ps(destination + col,
                                _mm256_fmadd_ps(x, weight, y));
                        }
                        const float scalar_weight = kernel[kr * 3 + kc];
                        for (; col < length; ++col)
                            destination[col] += source[col] * scalar_weight;
                    }
                }
            }
        }
    }
}

__attribute__((target("avx2,fma")))
void conv3x3_backward_avx(ConvLayer& layer, const std::vector<float>& input,
                          const std::vector<float>& grad_output,
                          std::vector<float>& grad_input) {
    grad_input.assign(static_cast<size_t>(layer.in_ch) * CELLS, 0.0f);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        const float* go = grad_output.data() +
                          static_cast<size_t>(out_channel) * CELLS;
        layer.gb[out_channel] += sum_avx(go, CELLS);
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch * 9;
        float* grad_weights = layer.gw.data() +
            static_cast<size_t>(out_channel) * layer.in_ch * 9;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input.data() +
                              static_cast<size_t>(in_channel) * CELLS;
            float* gi = grad_input.data() +
                        static_cast<size_t>(in_channel) * CELLS;
            const float* kernel = weights + static_cast<size_t>(in_channel) * 9;
            float* grad_kernel = grad_weights +
                                 static_cast<size_t>(in_channel) * 9;
            for (int kr = 0; kr < 3; ++kr) {
                const int row_begin = std::max(0, 1 - kr);
                const int row_end = std::min(SIDE, SIDE + 1 - kr);
                for (int kc = 0; kc < 3; ++kc) {
                    const int col_begin = std::max(0, 1 - kc);
                    const int col_end = std::min(SIDE, SIDE + 1 - kc);
                    const int length = col_end - col_begin;
                    const __m256 weight = _mm256_set1_ps(kernel[kr * 3 + kc]);
                    __m256 grad_weight = _mm256_setzero_ps();
                    float scalar_grad_weight = 0.0f;
                    for (int row = row_begin; row < row_end; ++row) {
                        const float* go_row = go + row * SIDE + col_begin;
                        const float* in_row = in + (row + kr - 1) * SIDE +
                                              col_begin + kc - 1;
                        float* gi_row = gi + (row + kr - 1) * SIDE +
                                        col_begin + kc - 1;
                        int col = 0;
                        for (; col + 8 <= length; col += 8) {
                            const __m256 gradient = _mm256_loadu_ps(go_row + col);
                            const __m256 x = _mm256_loadu_ps(in_row + col);
                            grad_weight = _mm256_fmadd_ps(
                                gradient, x, grad_weight);
                            const __m256 previous = _mm256_loadu_ps(gi_row + col);
                            _mm256_storeu_ps(gi_row + col,
                                _mm256_fmadd_ps(gradient, weight, previous));
                        }
                        const float scalar_weight = kernel[kr * 3 + kc];
                        for (; col < length; ++col) {
                            scalar_grad_weight += go_row[col] * in_row[col];
                            gi_row[col] += go_row[col] * scalar_weight;
                        }
                    }
                    grad_kernel[kr * 3 + kc] +=
                        horizontal_sum(grad_weight) + scalar_grad_weight;
                }
            }
        }
    }
}

__attribute__((target("avx2,fma")))
void conv1x1_forward_avx(const ConvLayer& layer,
                         const std::vector<float>& input,
                         std::vector<float>& output) {
    output.resize(static_cast<size_t>(layer.out_ch) * CELLS);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        float* out = output.data() + static_cast<size_t>(out_channel) * CELLS;
        std::fill(out, out + CELLS, layer.b[out_channel]);
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input.data() +
                              static_cast<size_t>(in_channel) * CELLS;
            const __m256 weight = _mm256_set1_ps(weights[in_channel]);
            int cell = 0;
            for (; cell + 8 <= CELLS; cell += 8) {
                const __m256 x = _mm256_loadu_ps(in + cell);
                const __m256 y = _mm256_loadu_ps(out + cell);
                _mm256_storeu_ps(out + cell, _mm256_fmadd_ps(x, weight, y));
            }
            for (; cell < CELLS; ++cell)
                out[cell] += in[cell] * weights[in_channel];
        }
    }
}

__attribute__((target("avx2,fma")))
void conv1x1_backward_avx(ConvLayer& layer,
                          const std::vector<float>& input,
                          const std::vector<float>& grad_output,
                          std::vector<float>& grad_input) {
    grad_input.assign(static_cast<size_t>(layer.in_ch) * CELLS, 0.0f);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        const float* go = grad_output.data() +
                          static_cast<size_t>(out_channel) * CELLS;
        layer.gb[out_channel] += sum_avx(go, CELLS);
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch;
        float* grad_weights = layer.gw.data() +
            static_cast<size_t>(out_channel) * layer.in_ch;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input.data() +
                              static_cast<size_t>(in_channel) * CELLS;
            float* gi = grad_input.data() +
                        static_cast<size_t>(in_channel) * CELLS;
            const __m256 weight = _mm256_set1_ps(weights[in_channel]);
            __m256 grad_weight = _mm256_setzero_ps();
            int cell = 0;
            for (; cell + 8 <= CELLS; cell += 8) {
                const __m256 gradient = _mm256_loadu_ps(go + cell);
                const __m256 x = _mm256_loadu_ps(in + cell);
                grad_weight = _mm256_fmadd_ps(gradient, x, grad_weight);
                const __m256 previous = _mm256_loadu_ps(gi + cell);
                _mm256_storeu_ps(gi + cell,
                    _mm256_fmadd_ps(gradient, weight, previous));
            }
            float scalar_grad_weight = 0.0f;
            const float scalar_weight = weights[in_channel];
            for (; cell < CELLS; ++cell) {
                scalar_grad_weight += go[cell] * in[cell];
                gi[cell] += go[cell] * scalar_weight;
            }
            grad_weights[in_channel] +=
                horizontal_sum(grad_weight) + scalar_grad_weight;
        }
    }
}

__attribute__((target("avx2,fma")))
void fc_forward_avx(const FCLayer& layer, const std::vector<float>& input,
                    std::vector<float>& output) {
    output.resize(layer.out_n);
    for (int out = 0; out < layer.out_n; ++out) {
        const float* weights = layer.w.data() +
                               static_cast<size_t>(out) * layer.in_n;
        __m256 sum = _mm256_setzero_ps();
        int in = 0;
        for (; in + 8 <= layer.in_n; in += 8)
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(input.data() + in),
                                  _mm256_loadu_ps(weights + in), sum);
        float result = layer.b[out] + horizontal_sum(sum);
        for (; in < layer.in_n; ++in) result += input[in] * weights[in];
        output[out] = result;
    }
}

__attribute__((target("avx2,fma")))
void fc_backward_avx(FCLayer& layer, const std::vector<float>& input,
                     const std::vector<float>& grad_output,
                     std::vector<float>& grad_input) {
    grad_input.assign(layer.in_n, 0.0f);
    for (int out = 0; out < layer.out_n; ++out) {
        const float gradient_scalar = grad_output[out];
        layer.gb[out] += gradient_scalar;
        const __m256 gradient = _mm256_set1_ps(gradient_scalar);
        const float* weights = layer.w.data() +
                               static_cast<size_t>(out) * layer.in_n;
        float* grad_weights = layer.gw.data() +
                              static_cast<size_t>(out) * layer.in_n;
        int in = 0;
        for (; in + 8 <= layer.in_n; in += 8) {
            const __m256 x = _mm256_loadu_ps(input.data() + in);
            const __m256 gw = _mm256_loadu_ps(grad_weights + in);
            _mm256_storeu_ps(grad_weights + in,
                _mm256_fmadd_ps(gradient, x, gw));
            const __m256 weight = _mm256_loadu_ps(weights + in);
            const __m256 gi = _mm256_loadu_ps(grad_input.data() + in);
            _mm256_storeu_ps(grad_input.data() + in,
                _mm256_fmadd_ps(gradient, weight, gi));
        }
        for (; in < layer.in_n; ++in) {
            grad_weights[in] += gradient_scalar * input[in];
            grad_input[in] += gradient_scalar * weights[in];
        }
    }
}

__attribute__((target("avx2")))
void relu_backward_avx(const std::vector<float>& output,
                       std::vector<float>& gradient) {
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= gradient.size(); i += 8) {
        const __m256 y = _mm256_loadu_ps(output.data() + i);
        const __m256 g = _mm256_loadu_ps(gradient.data() + i);
        _mm256_storeu_ps(gradient.data() + i,
            _mm256_and_ps(g, _mm256_cmp_ps(y, zero, _CMP_GT_OQ)));
    }
    for (; i < gradient.size(); ++i)
        if (output[i] <= 0.0f) gradient[i] = 0.0f;
}
#endif

void conv3x3_forward_scalar(const ConvLayer& layer, const float* input,
                            std::vector<float>& output) {
    output.resize(static_cast<size_t>(layer.out_ch) * CELLS);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        float* out = output.data() + static_cast<size_t>(out_channel) * CELLS;
        std::fill(out, out + CELLS, layer.b[out_channel]);
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch * 9;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input + static_cast<size_t>(in_channel) * CELLS;
            const float* kernel = weights + static_cast<size_t>(in_channel) * 9;
            for (int kr = 0; kr < 3; ++kr) {
                const int row_begin = std::max(0, 1 - kr);
                const int row_end = std::min(SIDE, SIDE + 1 - kr);
                for (int kc = 0; kc < 3; ++kc) {
                    const int col_begin = std::max(0, 1 - kc);
                    const int col_end = std::min(SIDE, SIDE + 1 - kc);
                    const float weight = kernel[kr * 3 + kc];
                    for (int row = row_begin; row < row_end; ++row) {
                        float* destination = out + row * SIDE + col_begin;
                        const float* source = in + (row + kr - 1) * SIDE +
                                              col_begin + kc - 1;
                        for (int col = 0; col < col_end - col_begin; ++col)
                            destination[col] += source[col] * weight;
                    }
                }
            }
        }
    }
}

void conv3x3_backward_scalar(ConvLayer& layer,
                             const std::vector<float>& input,
                             const std::vector<float>& grad_output,
                             std::vector<float>& grad_input) {
    grad_input.assign(static_cast<size_t>(layer.in_ch) * CELLS, 0.0f);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        const float* go = grad_output.data() +
                          static_cast<size_t>(out_channel) * CELLS;
        for (int cell = 0; cell < CELLS; ++cell)
            layer.gb[out_channel] += go[cell];
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch * 9;
        float* grad_weights = layer.gw.data() +
            static_cast<size_t>(out_channel) * layer.in_ch * 9;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input.data() +
                              static_cast<size_t>(in_channel) * CELLS;
            float* gi = grad_input.data() +
                        static_cast<size_t>(in_channel) * CELLS;
            const float* kernel = weights + static_cast<size_t>(in_channel) * 9;
            float* grad_kernel = grad_weights +
                                 static_cast<size_t>(in_channel) * 9;
            for (int kr = 0; kr < 3; ++kr) {
                const int row_begin = std::max(0, 1 - kr);
                const int row_end = std::min(SIDE, SIDE + 1 - kr);
                for (int kc = 0; kc < 3; ++kc) {
                    const int col_begin = std::max(0, 1 - kc);
                    const int col_end = std::min(SIDE, SIDE + 1 - kc);
                    float grad_weight = 0.0f;
                    const float weight = kernel[kr * 3 + kc];
                    for (int row = row_begin; row < row_end; ++row) {
                        const float* go_row = go + row * SIDE + col_begin;
                        const float* in_row = in + (row + kr - 1) * SIDE +
                                              col_begin + kc - 1;
                        float* gi_row = gi + (row + kr - 1) * SIDE +
                                        col_begin + kc - 1;
                        for (int col = 0; col < col_end - col_begin; ++col) {
                            grad_weight += go_row[col] * in_row[col];
                            gi_row[col] += go_row[col] * weight;
                        }
                    }
                    grad_kernel[kr * 3 + kc] += grad_weight;
                }
            }
        }
    }
}

void conv1x1_forward_scalar(const ConvLayer& layer,
                            const std::vector<float>& input,
                            std::vector<float>& output) {
    output.resize(static_cast<size_t>(layer.out_ch) * CELLS);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        float* out = output.data() + static_cast<size_t>(out_channel) * CELLS;
        std::fill(out, out + CELLS, layer.b[out_channel]);
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input.data() +
                              static_cast<size_t>(in_channel) * CELLS;
            const float weight = weights[in_channel];
            for (int cell = 0; cell < CELLS; ++cell)
                out[cell] += in[cell] * weight;
        }
    }
}

void conv1x1_backward_scalar(ConvLayer& layer,
                             const std::vector<float>& input,
                             const std::vector<float>& grad_output,
                             std::vector<float>& grad_input) {
    grad_input.assign(static_cast<size_t>(layer.in_ch) * CELLS, 0.0f);
    for (int out_channel = 0; out_channel < layer.out_ch; ++out_channel) {
        const float* go = grad_output.data() +
                          static_cast<size_t>(out_channel) * CELLS;
        for (int cell = 0; cell < CELLS; ++cell)
            layer.gb[out_channel] += go[cell];
        const float* weights = layer.w.data() +
            static_cast<size_t>(out_channel) * layer.in_ch;
        float* grad_weights = layer.gw.data() +
            static_cast<size_t>(out_channel) * layer.in_ch;
        for (int in_channel = 0; in_channel < layer.in_ch; ++in_channel) {
            const float* in = input.data() +
                              static_cast<size_t>(in_channel) * CELLS;
            float* gi = grad_input.data() +
                        static_cast<size_t>(in_channel) * CELLS;
            float grad_weight = 0.0f;
            for (int cell = 0; cell < CELLS; ++cell) {
                grad_weight += go[cell] * in[cell];
                gi[cell] += go[cell] * weights[in_channel];
            }
            grad_weights[in_channel] += grad_weight;
        }
    }
}

void fc_forward_scalar(const FCLayer& layer, const std::vector<float>& input,
                       std::vector<float>& output) {
    output.resize(layer.out_n);
    for (int out = 0; out < layer.out_n; ++out) {
        float result = layer.b[out];
        const float* weights = layer.w.data() +
                               static_cast<size_t>(out) * layer.in_n;
        for (int in = 0; in < layer.in_n; ++in)
            result += input[in] * weights[in];
        output[out] = result;
    }
}

void fc_backward_scalar(FCLayer& layer, const std::vector<float>& input,
                        const std::vector<float>& grad_output,
                        std::vector<float>& grad_input) {
    grad_input.assign(layer.in_n, 0.0f);
    for (int out = 0; out < layer.out_n; ++out) {
        const float gradient = grad_output[out];
        layer.gb[out] += gradient;
        const float* weights = layer.w.data() +
                               static_cast<size_t>(out) * layer.in_n;
        float* grad_weights = layer.gw.data() +
                              static_cast<size_t>(out) * layer.in_n;
        for (int in = 0; in < layer.in_n; ++in) {
            grad_weights[in] += gradient * input[in];
            grad_input[in] += gradient * weights[in];
        }
    }
}

}  // namespace

void conv3x3_forward(const ConvLayer& layer, const std::vector<float>& input,
                     std::vector<float>& output) {
    conv3x3_forward(layer, input.data(), output);
}

void conv3x3_forward(const ConvLayer& layer, const float* input,
                     std::vector<float>& output) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        conv3x3_forward_avx(layer, input, output);
        return;
    }
#endif
    conv3x3_forward_scalar(layer, input, output);
}

void conv3x3_backward(ConvLayer& layer, const std::vector<float>& input,
                      const std::vector<float>& grad_output,
                      std::vector<float>& grad_input) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        conv3x3_backward_avx(layer, input, grad_output, grad_input);
        return;
    }
#endif
    conv3x3_backward_scalar(layer, input, grad_output, grad_input);
}

void conv1x1_forward(const ConvLayer& layer,
                     const std::vector<float>& input,
                     std::vector<float>& output) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        conv1x1_forward_avx(layer, input, output);
        return;
    }
#endif
    conv1x1_forward_scalar(layer, input, output);
}

void conv1x1_backward(ConvLayer& layer, const std::vector<float>& input,
                      const std::vector<float>& grad_output,
                      std::vector<float>& grad_input) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        conv1x1_backward_avx(layer, input, grad_output, grad_input);
        return;
    }
#endif
    conv1x1_backward_scalar(layer, input, grad_output, grad_input);
}

void fc_forward(const FCLayer& layer, const std::vector<float>& input,
                std::vector<float>& output) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        fc_forward_avx(layer, input, output);
        return;
    }
#endif
    fc_forward_scalar(layer, input, output);
}

void fc_backward(FCLayer& layer, const std::vector<float>& input,
                 const std::vector<float>& grad_output,
                 std::vector<float>& grad_input) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        fc_backward_avx(layer, input, grad_output, grad_input);
        return;
    }
#endif
    fc_backward_scalar(layer, input, grad_output, grad_input);
}

void relu_backward(const std::vector<float>& output,
                   std::vector<float>& gradient) {
#if GOMOK_X86_AVX2
    if (use_avx2_fma()) {
        relu_backward_avx(output, gradient);
        return;
    }
#endif
    for (size_t i = 0; i < gradient.size(); ++i)
        if (output[i] <= 0.0f) gradient[i] = 0.0f;
}

}  // namespace gomoku
