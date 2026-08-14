#include "game.h"
#include "network.h"
#include "selfplay.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace gomoku;

namespace {
int fail(const std::string& message) {
    std::cerr << "smoke test failed: " << message << '\n';
    return 1;
}

bool test_convolution() {
    constexpr int cells = BOARD_CELLS;
    ConvLayer layer(2, 3, 3, 3);
    layer.w.resize(2 * 3 * 9);
    layer.b = {0.1f, -0.2f, 0.3f};
    std::vector<float> input(2 * cells);
    for (size_t i = 0; i < layer.w.size(); ++i)
        layer.w[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 17.0f;
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 19.0f;

    std::vector<float> actual;
    conv3x3_forward(layer, input, actual);
    for (int co = 0; co < 3; ++co) {
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                float expected = layer.b[co];
                for (int ci = 0; ci < 2; ++ci) {
                    for (int kr = -1; kr <= 1; ++kr) {
                        for (int kc = -1; kc <= 1; ++kc) {
                            const int ir = row + kr;
                            const int ic = col + kc;
                            if (ir >= 0 && ir < BOARD_SIZE &&
                                ic >= 0 && ic < BOARD_SIZE) {
                                expected += input[ci * cells + ir * BOARD_SIZE + ic] *
                                    layer.w[((co * 2 + ci) * 3 + kr + 1) * 3 + kc + 1];
                            }
                        }
                    }
                }
                const float got = actual[co * cells + row * BOARD_SIZE + col];
                if (std::abs(expected - got) > 2e-5f) return false;
            }
        }
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gomoku_smoke <roundtrip-dir>\n";
        return 2;
    }
    if (!test_convolution()) return fail("optimized convolution");

    Board board;
    for (int col = 0; col < 4; ++col) {
        board.play(col);
        board.play(BOARD_SIZE + col);
    }
    board.play(4);
    if (board.winner() != 1) return fail("winner detection");
    board.undo();
    if (board.winner() != 0 || board.move_count() != 8)
        return fail("undo after a win");

    PureNet net;
    std::vector<float> state(4 * BOARD_CELLS);
    board.encode_state(state.data());
    std::vector<float> log_policy;
    float value = 0.0f;
    net.set_fast_inference(false);
    net.forward_one(state, log_policy, value);
    const std::vector<float> reference_policy = log_policy;
    const float reference_value = value;
    net.set_fast_inference(true);
    net.forward_one(state, log_policy, value);
    float max_policy_error = 0.0f;
    for (size_t i = 0; i < log_policy.size(); ++i)
        max_policy_error = std::max(max_policy_error,
            std::abs(log_policy[i] - reference_policy[i]));
    if (max_policy_error > 2e-4f ||
        std::abs(value - reference_value) > 2e-4f)
        return fail("fast inference consistency");
    double policy_sum = 0.0;
    for (float logp : log_policy) policy_sum += std::exp(logp);
    if (!std::isfinite(value) || std::abs(policy_sum - 1.0) > 1e-4)
        return fail("network output");

    std::vector<float> direct_policy(BOARD_CELLS);
    float direct_value = 0.0f;
    // Probability and log-probability APIs must agree.
    net.forward_batch_policy(state.data(), 1, direct_policy.data(), &direct_value);
    for (int i = 0; i < BOARD_CELLS; ++i) {
        if (std::abs(direct_policy[i] - std::exp(log_policy[i])) > 2e-4f)
            return fail("probability path consistency");
    }
    if (std::abs(direct_value - value) > 2e-4f)
        return fail("probability path value");

    // INT8 stays close to FP32.
    net.set_quantized_inference(true);
    std::vector<float> q_policy;
    float q_value = 0.0f;
    net.forward_one(state, q_policy, q_value);
    net.set_quantized_inference(false);
    max_policy_error = 0.0f;
    for (size_t i = 0; i < log_policy.size(); ++i)
        max_policy_error = std::max(max_policy_error,
            std::abs(q_policy[i] - log_policy[i]));
    if (max_policy_error > 0.2f || std::abs(q_value - value) > 0.05f)
        return fail("quantized inference consistency");

    // Save/load round trip.
    std::filesystem::create_directories(argv[1]);
    if (!net.save(argv[1])) return fail("save");
    PureNet net2(2e-3);
    net2.set_fast_inference(true);
    if (!net2.load(argv[1])) return fail("load");
    std::vector<float> log_policy2;
    float value2 = 0.0f;
    net2.forward_one(state, log_policy2, value2);
    max_policy_error = 0.0f;
    for (size_t i = 0; i < log_policy.size(); ++i)
        max_policy_error = std::max(max_policy_error,
            std::abs(log_policy2[i] - log_policy[i]));
    if (max_policy_error > 2e-4f || std::abs(value2 - value) > 2e-4f)
        return fail("save/load roundtrip");

    // A small self-play batch must produce normalized targets.
    net2.set_quantized_inference(false);
    BatchedSelfPlay sp(net2, 5.0f, 50, 2, 1);
    auto samples = sp.run_batch(1.0f);
    int total = 0;
    for (const auto& game : samples) {
        if (game.empty()) return fail("self-play empty game");
        total += static_cast<int>(game.size());
        for (const auto& s : game) {
            double sum = 0.0;
            for (float p : s.probs) sum += p;
            if (std::abs(sum - 1.0) > 1e-3) return fail("self-play probs");
        }
    }
    if (total < 2) return fail("self-play samples");

    std::cout << "smoke test passed\n";
    return 0;
}
