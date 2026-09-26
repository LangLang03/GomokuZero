#include "game.h"
#include "network.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using gomoku::PureNet;

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

std::vector<float> tensor(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    uint32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<float> result(count);
    stream.read(reinterpret_cast<char*>(result.data()), count * sizeof(float));
    if (!stream) throw std::runtime_error("invalid test checkpoint tensor");
    return result;
}

float parameter_error(const fs::path& a, const fs::path& b) {
    float error = 0.0f;
    for (const auto& entry : fs::directory_iterator(a)) {
        const auto name = entry.path().filename().string();
        if (name.rfind("training_", 0) != 0 || entry.path().extension() != ".bin") continue;
        const auto left = tensor(entry.path()), right = tensor(b / name);
        if (left.size() != right.size()) return std::numeric_limits<float>::infinity();
        float local_error = 0.0f;
        for (size_t i = 0; i < left.size(); ++i)
            local_error = std::max(local_error, std::abs(left[i] - right[i]));
        if (local_error > 2e-5f) std::cout << name << ": " << local_error << '\n';
        error = std::max(error, local_error);
    }
    return error;
}

int64_t optimizer_step(const fs::path& path, const std::string& field = "optimizer_step") {
    std::ifstream stream(path / "training.meta");
    std::string key, value;
    while (stream >> key >> value) if (key == field) return std::stoll(value);
    return -1;
}

std::vector<std::vector<float>> value_parameters(const fs::path& path) {
    std::vector<std::vector<float>> result;
    const auto conv = "training_conv" + std::to_string(2 + 2 * gomoku::NetConfig::kResBlocks) + "_";
    // Only trainable tensors (plus the compatibility conv bias): running BN
    // statistics are allowed to adapt while the shared policy trunk changes.
    for (const char* suffix : {"weight", "bias", "bn_gamma", "bn_beta"})
        result.push_back(tensor(path / (conv + suffix + ".bin")));
    for (int fc : {1, 2})
        for (const char* suffix : {"weight", "bias"})
            result.push_back(tensor(path / ("training_fc" + std::to_string(fc) + "_" + suffix + ".bin")));
    return result;
}

struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() /
        ("gomoku_backend_training_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { fs::create_directories(path); }
    ~TemporaryDirectory() { std::error_code error; fs::remove_all(path, error); }
};
}

int main() {
    setenv("GOMOK_SEED", "20260926", 1);
    setenv("GOMOK_DETERMINISTIC", "1", 1);
    setenv("GOMOK_INFERENCE_TF32", "0", 1);
    setenv("GOMOK_TF32", "0", 1);
    setenv("GOMOK_FUSED_ADAM", "1", 1);
    constexpr int batch = 4, cells = gomoku::NetConfig::kCells;
    std::vector<float> states(batch * 4 * cells), targets(batch * cells), winners(batch);
    for (int b = 0; b < batch; ++b) {
        gomoku::Board board;
        for (int i = 0; i < 4 + b; ++i) board.play((i * 17 + b * 11) % cells);
        board.encode_state(states.data() + b * 4 * cells);
        targets[b * cells + board.legal_moves().front()] = 1.0f;
        winners[b] = b % 2 ? 1.0f : -1.0f;
    }
    PureNet net;
    TemporaryDirectory temporary;
    const auto a = temporary.path / "a", b = temporary.path / "b";
    bool invalid_clip_rejected = false;
    try { net.set_grad_clip(std::numeric_limits<double>::infinity()); }
    catch (const std::invalid_argument&) { invalid_clip_rejected = true; }
    check(invalid_clip_rejected, "non-finite clipping must be rejected");
    {
        // The thread setting previously had no effect. Parallel inference must
        // preserve exactly the scalar implementation's accumulation order.
        net.set_cuda_enabled(false);
        std::vector<float> serial(batch * cells), parallel(batch * cells), sv(batch), pv(batch);
        net.set_inference_threads(1);
        net.forward_batch(states.data(), batch, serial.data(), sv.data());
        check(std::all_of(sv.begin(), sv.end(), [](float value) { return value == 0.0f; }),
              "new models give an exactly neutral value on distinct legal states");
        net.set_inference_threads(2);
        net.forward_batch(states.data(), batch, parallel.data(), pv.data());
        check(serial == parallel && sv == pv, "CPU thread count preserves exact outputs");
        PureNet seeded;
        seeded.set_cuda_enabled(false);
        seeded.forward_batch(states.data(), batch, parallel.data(), pv.data());
        check(serial == parallel && sv == pv, "configured seed reproduces initialization");
        net.set_cuda_enabled(true);
    }
    if (!net.cuda_available()) {
        check(!net.save_training_state(a.string(), 0),
              "native inference export must not pretend to save optimizer state");
        std::cout << "backend_training: CPU checks complete, CUDA checks skipped\n";
        return failures ? 1 : 0;
    }
    {
        // Policy-only record pretraining has no valid outcome labels. Its
        // shared trunk may change, but the untrained value head must remain
        // neutral until actual value supervision becomes available.
        PureNet neutral;
        neutral.set_weight_decay(1e-4);
        std::vector<float> policy_before(batch * cells), policy_after(batch * cells), values(batch);
        neutral.forward_batch(states.data(), batch, policy_before.data(), values.data());
        check(std::all_of(values.begin(), values.end(), [](float value) { return value == 0.0f; }),
              "CUDA initialization also gives an exactly neutral value");
        const std::vector<float> no_outcomes(batch, 0.0f);
        const auto checkpoint = temporary.path / "neutral_value_checkpoint";
        const auto next_checkpoint = temporary.path / "neutral_value_next";
        int completed_steps = 0;
        check(neutral.save_training_state(checkpoint.string(), completed_steps), "save initial neutral value parameters");
        const auto initial_value_parameters = value_parameters(checkpoint);
        for (bool global_zero : {false, true}) {
            for (int step = 0; step < 3; ++step) {
                const auto stats = global_zero
                    ? neutral.train_step(states, batch, targets, winners, 1e-4, 0.0)
                    : neutral.train_step(states, batch, targets, winners, 1e-4, 1.0, no_outcomes);
                ++completed_steps;
                check(stats.loss == stats.policy_loss, "policy-only updates have no value objective");
                neutral.forward_batch(states.data(), batch, policy_after.data(), values.data());
                check(std::all_of(values.begin(), values.end(), [](float value) { return value == 0.0f; }),
                      "policy-only training preserves the neutral value prior");
            }
            check(neutral.save_training_state(checkpoint.string(), completed_steps), "save policy-only value parameters");
            check(value_parameters(checkpoint) == initial_value_parameters,
                  "neither zero sample weights nor zero global weight decays unsupervised value parameters");
            check(optimizer_step(checkpoint, "value_optimizer_step") == 0,
                  "policy-only updates do not advance value Adam moments or bias correction");
        }
        float policy_change = 0.0f;
        for (size_t i = 0; i < policy_before.size(); ++i)
            policy_change = std::max(policy_change, std::abs(policy_before[i] - policy_after[i]));
        check(policy_change > 1e-4f, "neutral value initialization still allows policy learning");

        // Resume a policy-only checkpoint with no value moments using the
        // standard optimizer. Its first supervised update checks the fused
        // kernel's independent value-step counter against LibTorch Adam.
        setenv("GOMOK_FUSED_ADAM", "0", 1);
        PureNet reference_value;
        setenv("GOMOK_FUSED_ADAM", "1", 1);
        check(reference_value.load_training_state(checkpoint.string(), nullptr),
              "policy-only checkpoint resumes with absent value moments");
        const auto first_supervised = neutral.train_step(states, batch, targets, winners, 5e-4);
        const auto first_reference = reference_value.train_step(states, batch, targets, winners, 5e-4);
        ++completed_steps;
        check(std::abs(first_supervised.loss - first_reference.loss) < 1e-6f,
              "first value-supervised loss agrees after policy-only resume");
        check(neutral.save_training_state(checkpoint.string(), completed_steps) &&
              reference_value.save_training_state(next_checkpoint.string(), completed_steps), "save first value Adam update");
        check(parameter_error(checkpoint, next_checkpoint) < 2e-5f,
              "first value Adam update uses its own bias-correction step");
        check(optimizer_step(checkpoint, "value_optimizer_step") == 1,
              "first supervised batch is value Adam step one");

        const float initial_value_loss = first_supervised.value_loss;
        float final_value_loss = initial_value_loss;
        for (int step = 1; step < 12; ++step) {
            const auto stats = neutral.train_step(states, batch, targets, winners, 5e-4);
            ++completed_steps;
            final_value_loss = stats.value_loss;
        }
        neutral.forward_batch(states.data(), batch, policy_after.data(), values.data());
        check(initial_value_loss == 1.0f && final_value_loss < initial_value_loss - 0.01f,
              "enabling outcome supervision decreases the previously neutral value loss");
        check(std::any_of(values.begin(), values.end(), [](float value) { return std::abs(value) > 1e-4f; }),
              "outcome supervision moves value predictions away from zero");
        const auto bounds = std::minmax_element(values.begin(), values.end());
        check(*bounds.second - *bounds.first > 1e-3f,
              "value supervision learns different positions, not only a constant bias");
        std::cout << "neutral value supervision loss: " << initial_value_loss << " -> " << final_value_loss
                  << ", policy-only max log-policy change: " << policy_change << '\n';

        check(neutral.save_training_state(checkpoint.string(), completed_steps), "save learned value after policy-only pretraining");
        const auto learned_weights = tensor(checkpoint / "training_fc2_weight.bin");
        check(std::any_of(learned_weights.begin(), learned_weights.end(), [](float value) { return value != 0.0f; }),
              "outcome supervision trains the initially zero final-layer weights");
        const auto learned_value_parameters = value_parameters(checkpoint);
        for (bool global_zero : {false, true}) {
            if (global_zero) neutral.train_step(states, batch, targets, winners, 1e-4, 0.0);
            else neutral.train_step(states, batch, targets, winners, 1e-4, 1.0, no_outcomes);
            ++completed_steps;
            check(neutral.save_training_state(checkpoint.string(), completed_steps), "save supervised-to-policy-only transition");
            check(value_parameters(checkpoint) == learned_value_parameters,
                  "returning to policy-only training clears stale value gradients and skips weight decay");
            check(optimizer_step(checkpoint, "value_optimizer_step") == 12,
                  "returning to policy-only training preserves existing value Adam state");
        }
        neutral.forward_batch(states.data(), batch, policy_after.data(), values.data());
        PureNet resumed_value;
        check(resumed_value.load_training_state(checkpoint.string(), nullptr), "restore learned value checkpoint");
        std::vector<float> resumed_policy(batch * cells), resumed_values(batch);
        resumed_value.forward_batch(states.data(), batch, resumed_policy.data(), resumed_values.data());
        float resumed_value_error = 0.0f;
        for (int i = 0; i < batch; ++i)
            resumed_value_error = std::max(resumed_value_error, std::abs(values[i] - resumed_values[i]));
        check(resumed_value_error < 1e-6f, "loading learned value never reapplies zero initialization");
        const auto continued = neutral.train_step(states, batch, targets, winners, 5e-4);
        const auto resumed = resumed_value.train_step(states, batch, targets, winners, 5e-4);
        ++completed_steps;
        check(std::abs(continued.loss - resumed.loss) < 1e-6f,
              "neutral-initialized value head resumes with the same next-step objective");
        check(neutral.save_training_state(checkpoint.string(), completed_steps) &&
              resumed_value.save_training_state(next_checkpoint.string(), completed_steps), "save learned-value continuation probes");
        check(parameter_error(checkpoint, next_checkpoint) < 1e-6f,
              "neutral-initialized value head resumes with equivalent parameters and BN state");
    }
    net.set_weight_decay(1e-4);
    net.set_grad_clip(0.7);
    // Two confidence scales represent the same weighted mean, including when
    // their total is below one. Unlabelled examples contribute no value loss.
    PureNet scaled;
    scaled.set_weight_decay(1e-4);
    scaled.set_grad_clip(0.7);
    const auto full = net.train_step(states, batch, targets, winners, 1e-4, 1.0,
                                    {1.0f, 0.0f, 2.0f, 0.0f});
    const auto small = scaled.train_step(states, batch, targets, winners, 1e-4, 1.0,
                                         {0.125f, 0.0f, 0.25f, 0.0f});
    check(std::abs(full.value_loss - small.value_loss) < 1e-6f,
          "value confidence normalization is scale invariant");
    check(net.save_training_state(a.string(), 1) && scaled.save_training_state(b.string(), 1),
          "save confidence normalization probes");
    check(parameter_error(a, b) < 1e-6f, "confidence scaling preserves parameter update");

    // Reference LibTorch Adam and the fused kernel must optimize the same
    // objective including coupled weight decay and global gradient clipping.
    setenv("GOMOK_FUSED_ADAM", "0", 1);
    PureNet reference;
    setenv("GOMOK_FUSED_ADAM", "1", 1);
    // Before value updates were counted separately, a checkpoint contained
    // only optimizer_step. Its fully supervised state still loads faithfully.
    std::string current_metadata, legacy_metadata, metadata_line;
    {
        std::ifstream metadata(a / "training.meta");
        while (std::getline(metadata, metadata_line)) {
            current_metadata += metadata_line + '\n';
            if (metadata_line.rfind("value_optimizer_step ", 0) != 0)
                legacy_metadata += metadata_line + '\n';
        }
    }
    { std::ofstream metadata(a / "training.meta"); metadata << legacy_metadata; }
    check(reference.load_training_state(a.string(), nullptr), "load reference optimizer state");
    { std::ofstream metadata(a / "training.meta"); metadata << current_metadata; }
    check(reference.grad_clip() == 0.7, "checkpoint restores gradient clipping");
    float adam_error = 0.0f;
    for (int step = 0; step < 2; ++step) {
        // Compare each optimizer update from identical parameters/moments.
        // Chaining tiny rounding differences through millions of ReLU units
        // can change which units have nonzero derivatives on the next step.
        check(net.save_training_state(a.string(), step + 1) &&
              reference.load_training_state(a.string(), nullptr), "reset optimizer reference");
        const auto fused = net.train_step(states, batch, targets, winners, 1e-4);
        const auto standard = reference.train_step(states, batch, targets, winners, 1e-4);
        check(std::abs(fused.loss - standard.loss) < 2e-4f,
              "fused Adam and reference objective agree");
        check(net.save_training_state(a.string(), step + 2) &&
              reference.save_training_state(b.string(), step + 2), "save optimizer equivalence probes");
        adam_error = std::max(adam_error, parameter_error(a, b));
    }
    check(net.save_training_state(a.string(), 3) && reference.save_training_state(b.string(), 3),
          "save optimizer equivalence probes");
    std::cout << "fused/reference max parameter or BN difference: " << adam_error << '\n';
    check(adam_error < 2e-5f, "fused Adam matches standard Adam updates");

    // Saving while serving CPU inference must retain the authoritative Adam
    // state, and switching back must preserve the next training update.
    net.set_cuda_enabled(false);
    check(net.save_training_state(b.string(), 3), "save full state while CUDA inference is disabled");
    check(optimizer_step(b) == 3 && fs::exists(b / "training_optimizer.pt"),
          "CPU-mode checkpoint contains current Adam moments and step");
    net.set_cuda_enabled(true);

    // Forward-time BatchNorm updates used to survive a rejected update; the
    // Adam step counter advanced too. Stress finite input that overflows the
    // convolution to exercise rollback rather than only argument validation.
    std::vector<float> overflow(states.size(), std::numeric_limits<float>::max());
    bool rejected = false;
    try { net.train_step(overflow, batch, targets, winners, 1e-4); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "non-finite forward or backward must reject the update");
    check(net.save_training_state(b.string(), 3), "save state after rejected update");
    check(parameter_error(a, b) == 0.0f && optimizer_step(b) == 3,
          "rejected update preserves weights, BatchNorm statistics and Adam step");
    PureNet restored;
    check(restored.load_training_state(a.string(), nullptr), "restore before-rejection state");
    const auto recovered = net.train_step(states, batch, targets, winners, 1e-4);
    const auto uninterrupted = restored.train_step(states, batch, targets, winners, 1e-4);
    check(std::abs(recovered.loss - uninterrupted.loss) < 1e-6f,
          "training recovers from rejected update with unchanged objective");
    check(net.save_training_state(a.string(), 4) && restored.save_training_state(b.string(), 4),
          "save recovery update");
    check(parameter_error(a, b) < 1e-6f, "rejected update does not corrupt Adam moments");

    // Compare the default training precision to the strict-fp32 reference at
    // the same checkpoint and data, with inference held in fp32 for both.
    setenv("GOMOK_TF32", "1", 1);
    PureNet tf32;
    check(tf32.load_training_state(a.string(), nullptr), "restore TF32 comparison state");
    const auto strict_metrics = net.train_step(states, batch, targets, winners, 1e-4);
    const auto tf32_metrics = tf32.train_step(states, batch, targets, winners, 1e-4);
    const float loss_error = std::abs(strict_metrics.loss - tf32_metrics.loss);
    std::vector<float> strict_policy(batch * cells), fast_policy(batch * cells), strict_value(batch), fast_value(batch);
    net.forward_batch(states.data(), batch, strict_policy.data(), strict_value.data());
    tf32.forward_batch(states.data(), batch, fast_policy.data(), fast_value.data());
    float policy_error = 0.0f, value_error = 0.0f;
    for (size_t i = 0; i < strict_policy.size(); ++i)
        policy_error = std::max(policy_error, std::abs(strict_policy[i] - fast_policy[i]));
    for (int i = 0; i < batch; ++i)
        value_error = std::max(value_error, std::abs(strict_value[i] - fast_value[i]));
    std::cout << "TF32/fp32 loss difference: " << loss_error
              << ", next-update max log-policy difference: " << policy_error
              << ", value difference: " << value_error << '\n';
    check(loss_error < 1e-3f && policy_error < 1e-2f && value_error < 1e-3f,
          "TF32 training stays close to strict fp32 reference");

    PureNet bf16;
    check(bf16.load_training_state(a.string(), nullptr), "restore BF16 comparison state");
    bf16.set_mixed_precision(true);
    const auto bf16_metrics = bf16.train_step(states, batch, targets, winners, 1e-4);
    bf16.forward_batch(states.data(), batch, fast_policy.data(), fast_value.data());
    policy_error = value_error = 0.0f;
    for (size_t i = 0; i < strict_policy.size(); ++i)
        policy_error = std::max(policy_error, std::abs(strict_policy[i] - fast_policy[i]));
    for (int i = 0; i < batch; ++i)
        value_error = std::max(value_error, std::abs(strict_value[i] - fast_value[i]));
    const float bf16_loss_error = std::abs(strict_metrics.loss - bf16_metrics.loss);
    std::cout << "BF16/fp32 loss difference: " << bf16_loss_error
              << ", next-update max log-policy difference: " << policy_error
              << ", value difference: " << value_error << '\n';
    check(bf16_loss_error < 0.05f && policy_error < 0.1f && value_error < 0.02f,
          "optional BF16 stays within its single-update accuracy budget");
    check(bf16.save_training_state(b.string(), 5), "save BF16 training state");
    bf16.set_cuda_enabled(false);
    bf16.forward_batch(states.data(), batch, strict_policy.data(), strict_value.data());
    policy_error = value_error = 0.0f;
    for (size_t i = 0; i < strict_policy.size(); ++i)
        policy_error = std::max(policy_error, std::abs(strict_policy[i] - fast_policy[i]));
    for (int i = 0; i < batch; ++i)
        value_error = std::max(value_error, std::abs(strict_value[i] - fast_value[i]));
    check(policy_error < 5e-3f && value_error < 5e-3f,
          "BF16 training keeps strict-fp32 CUDA/native inference compatibility");

    // TF32 inference is independently checked on the same trained weights.
    setenv("GOMOK_INFERENCE_TF32", "1", 1);
    PureNet fast_inference;
    check(fast_inference.load_training_state(b.string(), nullptr), "restore TF32 inference probe");
    fast_inference.forward_batch(states.data(), batch, fast_policy.data(), fast_value.data());
    float legal_probability_error = 0.0f;
    policy_error = value_error = 0.0f;
    int move_changes = 0;
    for (int row = 0; row < batch; ++row) {
        double strict_sum = 0.0, fast_sum = 0.0;
        int strict_best = -1, fast_best = -1;
        for (int move = 0; move < cells; ++move) {
            const size_t cell = static_cast<size_t>(row) * cells + move;
            policy_error = std::max(policy_error, std::abs(strict_policy[cell] - fast_policy[cell]));
            if (states[row * 4 * cells + move] || states[row * 4 * cells + cells + move]) continue;
            strict_sum += std::exp(strict_policy[cell]);
            fast_sum += std::exp(fast_policy[cell]);
            if (strict_best < 0 || strict_policy[cell] > strict_policy[row * cells + strict_best]) strict_best = move;
            if (fast_best < 0 || fast_policy[cell] > fast_policy[row * cells + fast_best]) fast_best = move;
        }
        for (int move = 0; move < cells; ++move) {
            if (states[row * 4 * cells + move] || states[row * 4 * cells + cells + move]) continue;
            const size_t cell = static_cast<size_t>(row) * cells + move;
            legal_probability_error = std::max(legal_probability_error, static_cast<float>(std::abs(
                std::exp(strict_policy[cell]) / strict_sum - std::exp(fast_policy[cell]) / fast_sum)));
        }
        move_changes += strict_best != fast_best;
        value_error = std::max(value_error, std::abs(strict_value[row] - fast_value[row]));
    }
    std::cout << "TF32/native inference max log-policy difference: " << policy_error
              << ", legal probability difference: " << legal_probability_error
              << ", value difference: " << value_error << ", move changes: " << move_changes << '\n';
    check(legal_probability_error < 1e-4f && value_error < 5e-3f && move_changes == 0,
          "fast TF32 inference preserves legal distribution and selected actions on the probe");
    const auto inference_export = temporary.path / "inference";
    check(!net.export_inference(b.string()), "refuse stale training metadata in inference export destination");
    check(net.export_inference(inference_export.string()) && fast_inference.load(inference_export.string()),
          "load fresh inference-only model after training checkpoint");
    check(fast_inference.resume_batch_index() == 0,
          "inference-only model starts a fresh training schedule");
    std::cout << "backend_training " << (failures ? "FAILED" : "passed") << '\n';
    return failures ? 1 : 0;
}
