// Compares native backward kernels with central finite differences.
#include "gradient.h"
#include <cmath>
#include <cstdio>
#include <random>

using namespace gomoku;

static int fails = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s\n", msg);                                    \
            ++fails;                                                           \
        }                                                                      \
    } while (0)

namespace {

constexpr float EPS = 1e-2f;

template <typename L, typename Fwd, typename Bwd>
void check_layer_grads(const char* name, L& l, const std::vector<float>& in,
                       Fwd fwd, Bwd bwd) {
    // loss = sum(out)
    std::vector<float> out;
    fwd(l, in, out);
    std::vector<float> grad_out(out.size(), 1.0f);
    std::vector<float> grad_in(in.size(), 0.0f);

    l.gw.assign(l.w.size(), 0.0f);
    l.gb.assign(l.b.size(), 0.0f);
    bwd(l, in, grad_out, grad_in);

    auto loss_at = [&](const std::vector<float>& w, const std::vector<float>& b) {
        L copy = l;
        copy.w = w;
        copy.b = b;
        std::vector<float> o;
        fwd(copy, in, o);
        float s = 0.0f;
        for (float v : o) s += v;
        return s;
    };

    for (size_t i = 0; i < l.w.size(); ++i) {
        auto wp = l.w, wm = l.w;
        wp[i] += EPS;
        wm[i] -= EPS;
        const float num = (loss_at(wp, l.b) - loss_at(wm, l.b)) / (2 * EPS);
        const float tol = 1e-2f + 1e-2f * std::abs(l.gw[i]);
        if (std::abs(num - l.gw[i]) > tol) {
            std::printf("FAIL: %s weight[%zu] num=%g analytic=%g\n",
                        name, i, num, l.gw[i]);
            ++fails;
        }
    }
    for (size_t i = 0; i < l.b.size(); ++i) {
        auto bp = l.b, bm = l.b;
        bp[i] += EPS;
        bm[i] -= EPS;
        const float num = (loss_at(l.w, bp) - loss_at(l.w, bm)) / (2 * EPS);
        const float tol = 1e-2f + 1e-2f * std::abs(l.gb[i]);
        if (std::abs(num - l.gb[i]) > tol) {
            std::printf("FAIL: %s bias[%zu] num=%g analytic=%g\n",
                        name, i, num, l.gb[i]);
            ++fails;
        }
    }
    // Sample input coordinates to keep the test short.
    std::mt19937 rng(7);
    std::uniform_int_distribution<size_t> pick(0, in.size() - 1);
    for (int k = 0; k < 8; ++k) {
        const size_t i = pick(rng);
        auto ip = in, im = in;
        ip[i] += EPS;
        im[i] -= EPS;
        auto lp = [&](const std::vector<float>& x) {
            std::vector<float> o;
            fwd(l, x, o);
            float s = 0.0f;
            for (float v : o) s += v;
            return s;
        };
        const float num = (lp(ip) - lp(im)) / (2 * EPS);
        const float tol = 1e-2f + 1e-2f * std::abs(grad_in[i]);
        if (std::abs(num - grad_in[i]) > tol) {
            std::printf("FAIL: %s input[%zu] num=%g analytic=%g\n",
                        name, i, num, grad_in[i]);
            ++fails;
        }
    }
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    constexpr int H = 15, W = 15;

    {
        ConvLayer l(2, 3, 3, 3);
        l.w.resize(3 * 2 * 9);
        l.b.resize(3);
        for (float& v : l.w) v = dist(rng);
        for (float& v : l.b) v = dist(rng);
        std::vector<float> in(2 * H * W);
        for (float& v : in) v = dist(rng);
        check_layer_grads("conv3x3", l, in,
                          [](ConvLayer& x, const std::vector<float>& i,
                             std::vector<float>& o) { conv3x3_forward(x, i, o); },
                          [](ConvLayer& x, const std::vector<float>& i,
                             const std::vector<float>& go,
                             std::vector<float>& gi) {
                              conv3x3_backward(x, i, go, gi);
                          });
    }
    {
        ConvLayer l(3, 2, 1, 1);
        l.w.resize(2 * 3);
        l.b.resize(2);
        for (float& v : l.w) v = dist(rng);
        for (float& v : l.b) v = dist(rng);
        std::vector<float> in(3 * H * W);
        for (float& v : in) v = dist(rng);
        check_layer_grads("conv1x1", l, in,
                          [](ConvLayer& x, const std::vector<float>& i,
                             std::vector<float>& o) { conv1x1_forward(x, i, o); },
                          [](ConvLayer& x, const std::vector<float>& i,
                             const std::vector<float>& go,
                             std::vector<float>& gi) {
                              conv1x1_backward(x, i, go, gi);
                          });
    }
    {
        FCLayer l(6, 4);
        l.w.resize(4 * 6);
        l.b.resize(4);
        for (float& v : l.w) v = dist(rng);
        for (float& v : l.b) v = dist(rng);
        std::vector<float> in(6);
        for (float& v : in) v = dist(rng);
        check_layer_grads("fc", l, in,
                          [](FCLayer& x, const std::vector<float>& i,
                             std::vector<float>& o) { fc_forward(x, i, o); },
                          [](FCLayer& x, const std::vector<float>& i,
                             const std::vector<float>& go,
                             std::vector<float>& gi) {
                              fc_backward(x, i, go, gi);
                          });
    }
    {
        // ReLU mask.
        std::vector<float> y = {-1.0f, 0.5f, 0.0f, 2.0f, -3.0f};
        std::vector<float> g = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        relu_backward(y, g);
        CHECK(g[0] == 0.0f && g[1] == 1.0f && g[2] == 0.0f &&
              g[3] == 1.0f && g[4] == 0.0f, "relu backward mask");
    }

    if (fails == 0) {
        std::printf("gradient_check passed\n");
        return 0;
    }
    std::printf("gradient_check: %d failures\n", fails);
    return 1;
}
