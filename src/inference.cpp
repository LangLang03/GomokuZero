#include "inference.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gomoku {

// ---- weight loading ----
static bool load_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    f.close();
    return true;
}

bool HandNet::load(const std::string& dir) {
    std::vector<std::pair<std::string, std::vector<float>*>> layers = {
        {"conv1_weight", &conv1_w_}, {"conv1_bias", &conv1_b_},
        {"conv2_weight", &conv2_w_}, {"conv2_bias", &conv2_b_},
        {"conv3_weight", &conv3_w_}, {"conv3_bias", &conv3_b_},
        {"act_conv1_weight", &act_conv1_w_}, {"act_conv1_bias", &act_conv1_b_},
        {"act_fc1_weight", &act_fc1_w_}, {"act_fc1_bias", &act_fc1_b_},
        {"val_conv1_weight", &val_conv1_w_}, {"val_conv1_bias", &val_conv1_b_},
        {"val_fc1_weight", &val_fc1_w_}, {"val_fc1_bias", &val_fc1_b_},
        {"val_fc2_weight", &val_fc2_w_}, {"val_fc2_bias", &val_fc2_b_},
    };
    for (auto& [name, ptr] : layers) {
        if (!load_bin(dir + "/" + name + ".bin", *ptr)) {
            return false;
        }
    }
    loaded_ = true;
    return true;
}

// ---- conv3x3 with padding=1 (stride=1) ----
// in:  [Cin][H][W], out: [Cout][H][W]
// weight: [Cout][Cin][3][3]
static void conv3x3(const float* in, const float* w, const float* b,
                    int Cin, int Cout, int H, int W, float* out) {
    for (int co = 0; co < Cout; ++co) {
        const float* wc = w + co * Cin * 9;
        for (int h = 0; h < H; ++h) {
            for (int wd = 0; wd < W; ++wd) {
                float acc = b[co];
                for (int ci = 0; ci < Cin; ++ci) {
                    const float* in_c = in + ci * H * W;
                    const float* wk = wc + ci * 9;
                    for (int kh = -1; kh <= 1; ++kh) {
                        int ih = h + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = -1; kw <= 1; ++kw) {
                            int iw = wd + kw;
                            if (iw < 0 || iw >= W) continue;
                            acc += in_c[ih * W + iw] * wk[(kh + 1) * 3 + (kw + 1)];
                        }
                    }
                }
                out[co * H * W + h * W + wd] = acc;
            }
        }
    }
}

// ---- conv1x1 (no padding) ----
static void conv1x1(const float* in, const float* w, const float* b,
                    int Cin, int Cout, int H, int W, float* out) {
    for (int co = 0; co < Cout; ++co) {
        const float* wc = w + co * Cin;
        for (int h = 0; h < H; ++h) {
            for (int wd = 0; wd < W; ++wd) {
                float acc = b[co];
                for (int ci = 0; ci < Cin; ++ci) {
                    acc += in[ci * H * W + h * W + wd] * wc[ci];
                }
                out[co * H * W + h * W + wd] = acc;
            }
        }
    }
}

// ---- fully connected (row-major [out][in]) ----
static void fc(const float* in, const float* w, const float* b,
               int n_in, int n_out, float* out) {
    for (int o = 0; o < n_out; ++o) {
        float acc = b[o];
        const float* wrow = w + o * n_in;
        for (int i = 0; i < n_in; ++i) {
            acc += in[i] * wrow[i];
        }
        out[o] = acc;
    }
}

// ---- forward (batch) ----
void HandNet::forward(const float* states, int B,
                      float* log_policy, float* value) const {
    // per-batch scratch
    const int HW = H * W;
    // conv1 out [32][HW], conv2 [64][HW], conv3 [128][HW]
    std::vector<float> x1(C1 * HW), x2(C2 * HW), x3(C3 * HW);
    // policy: conv [4][HW], flat [4*HW], logits [225]
    std::vector<float> pa(4 * HW), pflat(4 * HW), plogit(N_OUT);
    // value: conv [2][HW], flat [2*HW], hidden [64]
    std::vector<float> va(2 * HW), vflat(2 * HW), vhidden(64);

    for (int b = 0; b < B; ++b) {
        const float* s = states + b * N_IN * HW;

        // conv1: 4->32 relu
        conv3x3(s, conv1_w_.data(), conv1_b_.data(), N_IN, C1, H, W, x1.data());
        for (auto& v : x1) v = std::max(v, 0.0f);
        // conv2: 32->64 relu
        conv3x3(x1.data(), conv2_w_.data(), conv2_b_.data(), C1, C2, H, W, x2.data());
        for (auto& v : x2) v = std::max(v, 0.0f);
        // conv3: 64->128 relu
        conv3x3(x2.data(), conv3_w_.data(), conv3_b_.data(), C2, C3, H, W, x3.data());
        for (auto& v : x3) v = std::max(v, 0.0f);

        // policy head
        conv1x1(x3.data(), act_conv1_w_.data(), act_conv1_b_.data(),
                C3, 4, H, W, pa.data());
        for (int i = 0; i < 4 * HW; ++i) pflat[i] = std::max(pa[i], 0.0f);
        fc(pflat.data(), act_fc1_w_.data(), act_fc1_b_.data(),
           4 * HW, N_OUT, plogit.data());
        // log softmax
        float mx = *std::max_element(plogit.begin(), plogit.end());
        float sum = 0.0f;
        for (int i = 0; i < N_OUT; ++i) {
            plogit[i] = plogit[i] - mx;
            sum += std::exp(plogit[i]);
        }
        float lse = mx + std::log(sum);  // log-sum-exp
        float* lp = log_policy + b * N_OUT;
        for (int i = 0; i < N_OUT; ++i) lp[i] = plogit[i] + mx - lse;

        // value head
        conv1x1(x3.data(), val_conv1_w_.data(), val_conv1_b_.data(),
                C3, 2, H, W, va.data());
        for (int i = 0; i < 2 * HW; ++i) vflat[i] = std::max(va[i], 0.0f);
        fc(vflat.data(), val_fc1_w_.data(), val_fc1_b_.data(),
           2 * HW, 64, vhidden.data());
        for (int i = 0; i < 64; ++i) vhidden[i] = std::max(vhidden[i], 0.0f);
        float vlogit;
        fc(vhidden.data(), val_fc2_w_.data(), val_fc2_b_.data(),
           64, 1, &vlogit);
        value[b] = std::tanh(vlogit);
    }
}

}  // namespace gomoku
