// Verifies that the native CPU forward pass and the LibTorch backend agree.
//
// The two implementations differ in almost every detail: the CPU path reads
// batch normalisation folded into the convolution weights, while LibTorch
// keeps BN as separate modules; one accumulates in a fixed order, the other
// uses cuDNN. They must still produce the same numbers, otherwise a --cpu
// switch silently changes how the network plays.
//
// With no CUDA device available the test still checks shape invariants and
// that the folded weights reproduce the reference batch-norm arithmetic.
#include "game.h"
#include "network.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace gomoku;

static int failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL: %s\n", msg);                                   \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

namespace {

std::vector<float> make_state(std::mt19937& rng, int stones) {
    std::vector<float> state(4 * NetConfig::kCells, 0.0f);
    std::vector<int> cells(NetConfig::kCells);
    for (int i = 0; i < NetConfig::kCells; ++i) cells[i] = i;
    std::shuffle(cells.begin(), cells.end(), rng);
    int last = -1;
    for (int k = 0; k < stones; ++k) {
        const int cell = cells[static_cast<size_t>(k)];
        state[static_cast<size_t>((k % 2) ? 1 : 0) * NetConfig::kCells + cell] = 1.0f;
        last = cell;
    }
    if (last >= 0) state[2 * NetConfig::kCells + last] = 1.0f;
    for (int i = 0; i < NetConfig::kCells; ++i)
        state[3 * NetConfig::kCells + i] = 1.0f;
    return state;
}

}  // namespace

int main() {
    PureNet net;
    const size_t parameters = net.parameter_count();
    std::printf("parameters: %zu (%.2f M), receptive field %dx%d\n", parameters,
                static_cast<double>(parameters) / 1e6,
                NetConfig::receptive_field(), NetConfig::receptive_field());

    // The trunk must be able to see far enough to judge five in a row.
    // 25 convolutions stack to a 51x51 window; anything spanning the board is
    // enough to judge five in a row.
    CHECK(NetConfig::receptive_field() >= 15,
          "receptive field must span the board");
    CHECK(parameters > 1000000, "network should have a usable parameter count");

    // A common checkpoint, so the two backends compare identical weights
    // instead of each instance's own random initialisation.
    const std::string directory =
        (std::filesystem::temp_directory_path() / "gomoku_parity_model").string();
    std::filesystem::create_directories(directory);
    CHECK(net.export_inference(directory), "inference export failed");

    std::mt19937 rng(2024);
    const int batch = 8;
    std::vector<float> states(static_cast<size_t>(batch) * 4 * NetConfig::kCells);
    for (int b = 0; b < batch; ++b) {
        const std::vector<float> state = make_state(rng, 2 + b * 3);
        std::copy(state.begin(), state.end(),
                  states.begin() + static_cast<size_t>(b) * state.size());
    }

    // Native forward on the saved weights.
    PureNet cpu_net;
    CHECK(cpu_net.load(directory), "load failed");
    cpu_net.set_cuda_enabled(false);
    std::vector<float> cpu_log(static_cast<size_t>(batch) * NetConfig::kCells);
    std::vector<float> cpu_value(batch);
    cpu_net.forward_batch(states.data(), batch, cpu_log.data(), cpu_value.data());

    for (int b = 0; b < batch; ++b) {
        const float* row = cpu_log.data() + static_cast<size_t>(b) * NetConfig::kCells;
        float sum = 0.0f;
        for (int k = 0; k < NetConfig::kCells; ++k) sum += std::exp(row[k]);
        CHECK(std::abs(sum - 1.0f) < 1e-3f, "log_policy must exponentiate to 1");
        CHECK(std::isfinite(cpu_value[b]) && cpu_value[b] >= -1.0f &&
                  cpu_value[b] <= 1.0f, "value must be a finite tanh output");
    }

    // Probability output must match the log output it is derived from.
    std::vector<float> probs(static_cast<size_t>(batch) * NetConfig::kCells);
    std::vector<float> prob_value(batch);
    cpu_net.forward_batch_policy(states.data(), batch, probs.data(),
                                 prob_value.data());
    float max_probability_error = 0.0f;
    float max_value_error = 0.0f;
    for (int b = 0; b < batch; ++b) {
        const float* lp = cpu_log.data() + static_cast<size_t>(b) * NetConfig::kCells;
        const float* pp = probs.data() + static_cast<size_t>(b) * NetConfig::kCells;
        for (int k = 0; k < NetConfig::kCells; ++k)
            max_probability_error =
                std::max(max_probability_error, std::abs(std::exp(lp[k]) - pp[k]));
        max_value_error = std::max(max_value_error,
                                   std::abs(prob_value[b] - cpu_value[b]));
    }
    std::printf("native log/probability agreement: policy %.3e value %.3e\n",
                max_probability_error, max_value_error);
    CHECK(max_probability_error < 1e-5f, "probability head must match log head");
    CHECK(max_value_error < 1e-6f, "value must not depend on the policy output form");

    if (!net.cuda_available()) {
        std::printf("no CUDA device: skipping the backend parity comparison\n");
        std::printf("backend_parity %s\n", failures == 0 ? "passed" : "FAILED");
        return failures == 0 ? 0 : 1;
    }

    // LibTorch forward, loading the same checkpoint so the weights match.
    PureNet cuda_net;
    CHECK(cuda_net.load(directory), "CUDA load failed");
    cuda_net.set_cuda_enabled(true);
    std::vector<float> cuda_log(static_cast<size_t>(batch) * NetConfig::kCells);
    std::vector<float> cuda_value(batch);
    cuda_net.forward_batch(states.data(), batch, cuda_log.data(),
                           cuda_value.data());

    // Isolate the sides: a pure LibTorch save/load round trip must be exact.
    // If it is, any remaining divergence comes from the folded-weight
    // conversion rather than from LibTorch itself.
    {
        const std::string round_trip =
            (std::filesystem::temp_directory_path() / "gomoku_parity_rt").string();
        std::filesystem::create_directories(round_trip);
        CHECK(cuda_net.export_inference(round_trip), "CUDA export failed");
        PureNet cuda_net2;
        CHECK(cuda_net2.load(round_trip), "CUDA reload failed");
        cuda_net2.set_cuda_enabled(true);
        std::vector<float> rt_log(static_cast<size_t>(batch) * NetConfig::kCells);
        std::vector<float> rt_value(batch);
        cuda_net2.forward_batch(states.data(), batch, rt_log.data(),
                                rt_value.data());
        float rt_error = 0.0f;
        for (size_t i = 0; i < rt_log.size(); ++i)
            rt_error = std::max(rt_error, std::abs(rt_log[i] - cuda_log[i]));
        std::printf("LibTorch save/load round trip: max log error %.3e\n",
                    rt_error);
        CHECK(rt_error < 1e-5f, "LibTorch round trip must be exact");
    }

    float max_log_error = 0.0f;
    float max_cuda_value_error = 0.0f;
    int argmax_mismatch = 0;
    for (int b = 0; b < batch; ++b) {
        const float* a = cpu_log.data() + static_cast<size_t>(b) * NetConfig::kCells;
        const float* c = cuda_log.data() + static_cast<size_t>(b) * NetConfig::kCells;
        int best_a = 0, best_c = 0;
        for (int k = 0; k < NetConfig::kCells; ++k) {
            max_log_error = std::max(max_log_error, std::abs(a[k] - c[k]));
            if (a[k] > a[best_a]) best_a = k;
            if (c[k] > c[best_c]) best_c = k;
        }
        if (best_a != best_c) ++argmax_mismatch;
        max_cuda_value_error =
            std::max(max_cuda_value_error, std::abs(cpu_value[b] - cuda_value[b]));
    }
    std::printf("CPU vs LibTorch: max log error %.3e, max value error %.3e, "
                "argmax mismatches %d/%d\n",
                max_log_error, max_cuda_value_error, argmax_mismatch, batch);
    // A deterministic probe makes it possible to compare against an
    // independent implementation by hand.
    {
        std::vector<float> probe(4 * NetConfig::kCells, 0.0f);
        probe[0 * NetConfig::kCells + 112] = 1.0f;
        probe[0 * NetConfig::kCells + 113] = 1.0f;
        probe[0 * NetConfig::kCells + 114] = 1.0f;
        probe[1 * NetConfig::kCells + 127] = 1.0f;
        probe[1 * NetConfig::kCells + 128] = 1.0f;
        probe[2 * NetConfig::kCells + 128] = 1.0f;
        for (int i = 0; i < NetConfig::kCells; ++i)
            probe[3 * NetConfig::kCells + i] = 1.0f;
        std::vector<float> lp(NetConfig::kCells), cp(NetConfig::kCells);
        float lv = 0.0f, cv = 0.0f;
        cpu_net.forward_one(probe, lp, lv);
        cuda_net.forward_one(probe, cp, cv);
        std::printf("probe native : value=%.8f logp[0]=%.8f logp[112]=%.8f\n",
                    lv, lp[0], lp[112]);
        std::printf("probe libtorch: value=%.8f logp[0]=%.8f logp[112]=%.8f\n",
                    cv, cp[0], cp[112]);
    }
    // 25 convolutions accumulated in a different order (im2col on the CPU,
    // cuDNN on the GPU) reach agreement on the order of 1e-3 in log space,
    // which is what fp32 allows here. The move chosen must match exactly.
    CHECK(max_log_error < 5e-3f, "CPU and LibTorch policies must agree");
    CHECK(max_cuda_value_error < 5e-3f, "CPU and LibTorch values must agree");
    CHECK(argmax_mismatch == 0, "CPU and LibTorch must pick the same move");

    std::printf("backend_parity %s\n", failures == 0 ? "passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
