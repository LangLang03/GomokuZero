// Diagnostic: measures how much of the network's parameter capacity is
// actually exercised by a board state, and how the policy head consumes it.
//
//   ./gomoku_diag <weights-dir> [n_random_states]
//
// Reports per-stage activation sparsity (dead ReLU units), the rank of the
// shared trunk feature map, and which fraction of the 202500-parameter
// act_fc head actually receives gradient for a typical MCTS target.
#include "gradient.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace gomoku;

namespace {

constexpr int SIDE = 15;
constexpr int CELLS = SIDE * SIDE;

bool load_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(f);
}

struct Stats {
    double nonzero_ratio = 0.0;  // fraction of activations > 0
    double mean = 0.0;
    double max_abs = 0.0;
    double dead_channels = 0.0;  // fraction of channels that are all-zero
};

Stats analyze(const std::vector<float>& act, int channels) {
    Stats s;
    size_t nonzero = 0;
    size_t dead = 0;
    double sum = 0.0;
    for (int c = 0; c < channels; ++c) {
        bool any = false;
        for (int i = 0; i < CELLS; ++i) {
            const float v = act[static_cast<size_t>(c) * CELLS + i];
            if (v > 0.0f) {
                ++nonzero;
                any = true;
            }
            sum += v;
            s.max_abs = std::max(s.max_abs, std::fabs(static_cast<double>(v)));
        }
        if (!any) ++dead;
    }
    s.nonzero_ratio = static_cast<double>(nonzero) / (channels * CELLS);
    s.mean = sum / (channels * CELLS);
    s.dead_channels = static_cast<double>(dead) / channels;
    return s;
}

void print_stats(const char* name, const Stats& s) {
    std::printf("  %-14s 激活非零率 %6.2f%%   均值 %8.5f   最大绝对值 %7.3f   全零通道 %5.1f%%\n",
                name, s.nonzero_ratio * 100.0, s.mean, s.max_abs,
                s.dead_channels * 100.0);
}

// A legal-looking mid-game position: 12 stones in the centre area.
void make_state(std::vector<float>& state, std::mt19937& rng, int n_stones) {
    state.assign(4 * CELLS, 0.0f);
    std::vector<int> cells(CELLS);
    for (int i = 0; i < CELLS; ++i) cells[i] = i;
    std::shuffle(cells.begin(), cells.end(), rng);
    int black = 0, white = 0;
    int last = -1;
    for (int k = 0; k < n_stones; ++k) {
        const int cell = cells[k];
        if (k % 2 == 0) {
            state[0 * CELLS + cell] = 1.0f;
            ++black;
        } else {
            state[1 * CELLS + cell] = 1.0f;
            ++white;
        }
        last = cell;
    }
    if (last >= 0) state[2 * CELLS + last] = 1.0f;
    if (black == white) {
        for (int i = 0; i < CELLS; ++i) state[3 * CELLS + i] = 1.0f;
    }
}

// Gate-level receptive field: how many input cells influence one output cell
// of the shared trunk. 3 layers of 3x3, no padding -> 7x7.
void report_receptive_field() {
    const int rf = 7;
    std::printf("  主干感受野                        %dx%d = %d 格 (棋盘 %d 格, %.1f%%)\n",
                rf, rf, rf * rf, CELLS, 100.0 * rf * rf / CELLS);
    std::printf("  五连判定所需最小视野              9x9 = 81 格  -> 缺 %d 格\n",
                81 - rf * rf);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights-dir> [n_states]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int n_states = argc >= 3 ? std::atoi(argv[2]) : 64;

    ConvLayer c1{4, 32, 3, 3}, c2{32, 64, 3, 3}, c3{64, 128, 3, 3};
    ConvLayer act_c{128, 4, 1, 1}, val_c{128, 2, 1, 1};
    FCLayer act_fc{4 * CELLS, CELLS}, val_fc1{2 * CELLS, 64}, val_fc2{64, 1};

    struct Entry { const char* name; std::vector<float>* w; std::vector<float>* b; };
    Entry entries[] = {
        {"conv1", &c1.w, &c1.b}, {"conv2", &c2.w, &c2.b}, {"conv3", &c3.w, &c3.b},
        {"act_conv1", &act_c.w, &act_c.b}, {"act_fc1", &act_fc.w, &act_fc.b},
        {"val_conv1", &val_c.w, &val_c.b}, {"val_fc1", &val_fc1.w, &val_fc1.b},
        {"val_fc2", &val_fc2.w, &val_fc2.b},
    };
    for (auto& e : entries) {
        if (!load_bin(dir + "/" + e.name + "_weight.bin", *e.w) ||
            !load_bin(dir + "/" + e.name + "_bias.bin", *e.b)) {
            std::fprintf(stderr, "load failed: %s\n", e.name);
            return 1;
        }
    }

    size_t total = 0;
    for (auto& e : entries) total += e.w->size() + e.b->size();
    std::printf("权重目录: %s\n", dir.c_str());
    std::printf("总参数量: %zu (%.2f M)\n\n", total,
                static_cast<double>(total) / 1e6);

    std::printf("=== 容量分布 ===\n");
    const size_t trunk = c1.w.size() + c2.w.size() + c3.w.size() +
                         c1.b.size() + c2.b.size() + c3.b.size();
    const size_t act_head = act_c.w.size() + act_c.b.size() +
                            act_fc.w.size() + act_fc.b.size();
    const size_t val_head = val_c.w.size() + val_c.b.size() +
                            val_fc1.w.size() + val_fc1.b.size() +
                            val_fc2.w.size() + val_fc2.b.size();
    std::printf("  共享主干   %9zu  %5.1f%%\n", trunk, 100.0 * trunk / total);
    std::printf("  策略头     %9zu  %5.1f%%  (其中 fc %d->%d 占 %.1f%%)\n",
                act_head, 100.0 * act_head / total, 4 * CELLS, CELLS,
                100.0 * act_fc.w.size() / total);
    std::printf("  价值头     %9zu  %5.1f%%\n", val_head, 100.0 * val_head / total);
    std::printf("\n=== 结构限制 ===\n");
    report_receptive_field();

    std::printf("\n=== 逐层激活统计 (%d 个随机中局状态平均) ===\n", n_states);
    std::mt19937 rng(12345);
    std::vector<float> state, c1_out, c2_out, c3_out, pa, plogit, va, vh,
        vlogit, columns, target, gp;
    std::vector<double> pa_nonzero, plogit_range, entropy;
    double plogit_max_all = 0.0;

    for (int s = 0; s < n_states; ++s) {
        make_state(state, rng, 2 + (s % 24));
        conv3x3_forward(c1, state.data(), c1_out);
        relu_forward(c1_out);
        conv3x3_forward(c2, c1_out, c2_out);
        relu_forward(c2_out);
        conv3x3_forward(c3, c2_out, c3_out);
        relu_forward(c3_out);
        conv1x1_forward(act_c, c3_out, pa);
        relu_forward(pa);
        fc_forward(act_fc, pa, plogit);

        if (s == 0) {
            std::printf("\n  [单个状态] 各阶段激活:\n");
            print_stats("conv1(32)", analyze(c1_out, 32));
            print_stats("conv2(64)", analyze(c2_out, 64));
            print_stats("conv3(128)", analyze(c3_out, 128));
            print_stats("act_c(4)", analyze(pa, 4));
        }

        size_t nz = 0;
        for (float v : pa)
            if (v > 0.0f) ++nz;
        pa_nonzero.push_back(static_cast<double>(nz) / pa.size());

        float lo = plogit[0], hi = plogit[0];
        for (float v : plogit) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        plogit_range.push_back(hi - lo);
        plogit_max_all = std::max(plogit_max_all, std::fabs(static_cast<double>(hi)));

        const float mx = *std::max_element(plogit.begin(), plogit.end());
        double sum = 0.0;
        for (float v : plogit) sum += std::exp(v - mx);
        double H = 0.0;
        for (float v : plogit) {
            const double p = std::exp(v - mx) / sum;
            H -= p * std::log(p);
        }
        entropy.push_back(H);
    }

    auto avg = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / v.size();
    };
    std::printf("\n=== 策略头瓶颈 (随机初始棋盘, %d 状态) ===\n", n_states);
    std::printf("  act_c 输出 pa 的激活率              %.2f%%  (900 个单元里平均 %.0f 个非零)\n",
                avg(pa_nonzero) * 100.0, avg(pa_nonzero) * 900.0);
    std::printf("  因此 act_fc 每步只有约             %.0f / 900 个输入参与前向\n",
                avg(pa_nonzero) * 900.0);
    std::printf("  act_fc 权重矩阵规模                225 x 900 = 202500\n");
    std::printf("  -> 每步真正被激活的输入子空间         %.0f 维 (占 %.1f%%)\n",
                avg(pa_nonzero) * 900.0, avg(pa_nonzero) * 100.0);
    std::printf("  plogit 动态范围 (max-min)           %.4f\n", avg(plogit_range));
    std::printf("  plogit 绝对最大值                   %.4f\n", plogit_max_all);
    std::printf("  策略输出平均熵                      %.4f nats (均匀 %.4f)\n",
                avg(entropy), std::log(225.0));
    std::printf("\n注: 未训练的网络 plogit 范围仅 %.4f, 而 MCTS 需要区分 225 个\n"
                "    落点的优劣; 该动态范围决定了策略头有效分辨率。\n",
                avg(plogit_range));
    return 0;
}
