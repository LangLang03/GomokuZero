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
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>
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
    setenv("GOMOK_INFERENCE_TF32", "0", 1);
    setenv("GOMOK_PINNED_INFERENCE", "0", 1);
    PureNet net;
    const size_t parameters = net.parameter_count();
    std::printf("parameters: %zu (%.2f M), receptive field %dx%d\n", parameters,
                static_cast<double>(parameters) / 1e6,
                NetConfig::receptive_field(), NetConfig::receptive_field());

    // The trunk must be able to see far enough to judge five in a row.
    // The production trunk stacks 25 convolutions into a 51x51 window, which
    // spans the board and so can see a five in a row. A sanitizer build
    // deliberately shrinks the trunk, so only require the production shape to
    // satisfy this.
    if (!NetConfig::kShrunkForSanitizer())
        CHECK(NetConfig::receptive_field() >= 15,
              "receptive field must span the board");
    if (!NetConfig::kShrunkForSanitizer())
        CHECK(parameters > 1000000,
              "network should have a usable parameter count");

    // A common checkpoint, so the two backends compare identical weights
    // instead of each instance's own random initialisation.
    const auto temp_root = std::filesystem::temp_directory_path() /
        ("gomoku_parity_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{temp_root};
    const std::string directory = (temp_root / "model").string();
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
    cuda_net.set_mixed_precision(std::getenv("GOMOK_TEST_BF16") != nullptr);
    CHECK(cuda_net.load(directory), "CUDA load failed");
    cuda_net.set_cuda_enabled(true);
    std::vector<float> cuda_log(static_cast<size_t>(batch) * NetConfig::kCells);
    std::vector<float> cuda_value(batch);
    cuda_net.forward_batch(states.data(), batch, cuda_log.data(),
                           cuda_value.data());

    // Exercise the opt-in pinned transfer path with both growth and reuse of
    // its staging buffers, comparing identical weights and inputs on CUDA.
    setenv("GOMOK_PINNED_INFERENCE", "1", 1);
    PureNet pinned_net;
    pinned_net.set_mixed_precision(std::getenv("GOMOK_TEST_BF16") != nullptr);
    CHECK(pinned_net.using_cuda(), "pinned inference requires CUDA");
    CHECK(pinned_net.load(directory), "pinned CUDA load failed");
    setenv("GOMOK_PINNED_INFERENCE", "0", 1);
    for (int count : {1, 3, 8, 2}) {
        std::vector<float> plain_log(static_cast<size_t>(count) * NetConfig::kCells);
        std::vector<float> pinned_log(plain_log.size());
        std::vector<float> plain_value(count), pinned_value(count);
        cuda_net.forward_batch(states.data(), count, plain_log.data(), plain_value.data());
        pinned_net.forward_batch(states.data(), count, pinned_log.data(), pinned_value.data());
        float difference = 0.0f;
        for (size_t i = 0; i < plain_log.size(); ++i)
            difference = std::max(difference, std::abs(plain_log[i] - pinned_log[i]));
        for (int i = 0; i < count; ++i)
            difference = std::max(difference, std::abs(plain_value[i] - pinned_value[i]));
        CHECK(difference < 1e-5f, "pinned CUDA log policy/value parity");
    }
    std::vector<float> pinned_probs(static_cast<size_t>(batch) * NetConfig::kCells);
    std::vector<float> pinned_prob_value(batch);
    pinned_net.forward_batch_policy(states.data(), batch, pinned_probs.data(),
                                    pinned_prob_value.data());
    for (size_t i = 0; i < pinned_probs.size(); ++i)
        CHECK(std::abs(pinned_probs[i] - std::exp(cuda_log[i])) < 1e-5f,
              "pinned CUDA probability parity");

    // Isolate the sides: a pure LibTorch save/load round trip must be exact.
    // If it is, any remaining divergence comes from the folded-weight
    // conversion rather than from LibTorch itself.
    {
        const std::string round_trip =
            (temp_root / "round_trip").string();
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

    // A zero supervision weight must suppress the value-head gradient even
    // while policy training and batch-normalisation statistics continue.
    auto bytes = [](const std::string& file) {
        std::ifstream stream(file, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
    };
    const auto checkpoint = (temp_root / "training").string();
    cuda_net.set_weight_decay(0.0);
    CHECK(cuda_net.save_training_state(checkpoint, 0), "save fresh training state");
    const auto value_weights_before = bytes(checkpoint + "/training_fc1_weight.bin");
    std::vector<float> targets(static_cast<size_t>(batch) * NetConfig::kCells, 0.0f);
    std::vector<float> outcomes(batch), value_weights(batch, 0.0f);
    for (int b = 0; b < batch; ++b) {
        targets[static_cast<size_t>(b) * NetConfig::kCells + (b * 17) % NetConfig::kCells] = 1.0f;
        outcomes[b] = b % 2 ? 1.0f : -1.0f;
    }
    const auto policy_only = cuda_net.train_step(states, batch, targets, outcomes,
                                                 1e-4, 1.0, value_weights);
    CHECK(policy_only.value_loss == 0.0f, "zero weights must give zero value loss");
    CHECK(std::isfinite(policy_only.grad_norm) && policy_only.grad_norm > 0.0f,
          "policy training must produce finite nonzero gradients");
    CHECK(cuda_net.save_training_state(checkpoint, 1), "save policy-only checkpoint");
    CHECK(value_weights_before == bytes(checkpoint + "/training_fc1_weight.bin"),
          "value-head parameters must not update without value supervision");

    cuda_net.train_step(states, batch, targets, outcomes, 1e-4);
    CHECK(cuda_net.save_training_state(checkpoint, 7), "save trained checkpoint");
    PureNet resumed;
    resumed.set_mixed_precision(cuda_net.mixed_precision());
    int resumed_batch = -1;
    CHECK(resumed.load_training_state(checkpoint, &resumed_batch), "restore raw BN and Adam state");
    CHECK(resumed_batch == 7, "restore batch index");
    // Compare the fused optimizer with LibTorch's independent reference Adam,
    // including restored moments. The environment is read at construction.
    const char* previous_fused = std::getenv("GOMOK_FUSED_ADAM");
    const std::string previous_fused_value = previous_fused ? previous_fused : "";
    setenv("GOMOK_FUSED_ADAM", "0", 1);
    PureNet reference_adam;
    if (previous_fused) setenv("GOMOK_FUSED_ADAM", previous_fused_value.c_str(), 1);
    else unsetenv("GOMOK_FUSED_ADAM");
    reference_adam.set_mixed_precision(cuda_net.mixed_precision());
    CHECK(reference_adam.load_training_state(checkpoint, nullptr), "load reference Adam model");
    auto compare = [&](PureNet& a, PureNet& b, float tolerance, const char* message) {
        std::vector<float> ap(NetConfig::kCells), bp(NetConfig::kCells);
        float av = 0.0f, bv = 0.0f;
        a.forward_batch(states.data(), 1, ap.data(), &av);
        b.forward_batch(states.data(), 1, bp.data(), &bv);
        float error = std::abs(av - bv);
        for (size_t i = 0; i < ap.size(); ++i) error = std::max(error, std::abs(ap[i] - bp[i]));
        std::printf("%s: %.3e\n", message, error);
        CHECK(error < tolerance, message);
    };
    compare(cuda_net, resumed, 1e-5f, "checkpoint forward equivalence");
    const auto uninterrupted = cuda_net.train_step(states, batch, targets, outcomes, 1e-4);
    const auto continued = resumed.train_step(states, batch, targets, outcomes, 1e-4);
    reference_adam.train_step(states, batch, targets, outcomes, 1e-4);
    CHECK(std::abs(uninterrupted.loss - continued.loss) < 1e-4f,
          "resumed next-step objective must agree");
    compare(cuda_net, resumed, 2e-4f, "checkpoint next-update equivalence");
    compare(cuda_net, reference_adam, 2e-4f, "fused versus reference Adam update equivalence");
    resumed.set_cuda_enabled(false);
    compare(cuda_net, resumed, 5e-3f, "live trained CPU mirror equivalence");
    resumed.set_cuda_enabled(true);
    compare(cuda_net, resumed, 2e-4f, "switching back preserves updated CUDA weights");

    // Missing Adam state must be rejected without partially replacing live
    // weights; inference exports remain a separate explicit loading path.
    std::filesystem::remove(checkpoint + "/training_optimizer.pt");
    CHECK(!resumed.load_training_state(checkpoint, nullptr), "reject incomplete optimizer checkpoint");
    compare(cuda_net, resumed, 2e-4f, "failed checkpoint load leaves live model intact");

    std::printf("backend_parity %s\n", failures == 0 ? "passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
