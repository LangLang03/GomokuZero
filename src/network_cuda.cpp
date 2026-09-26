#include "network_cuda.h"
#include "net_config.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#ifdef GOMOK_HAS_TORCH
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_fused_adam.h>

namespace gomoku {
namespace {

constexpr int kCells = NetConfig::kCells;
constexpr int kChannels = NetConfig::kChannels;
constexpr int kResBlocks = NetConfig::kResBlocks;

bool environment_enabled(const char* name, bool fallback = true) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value) != "0";
}

// NoTF32Guard only controls matrix multiplication; cuDNN convolutions have a
// separate process-wide flag. Serialize network submissions while selecting
// precision, so concurrent model instances cannot change each other's mode.
std::mutex cuda_precision_mutex;
class CudaPrecisionGuard {
public:
    explicit CudaPrecisionGuard(bool tf32)
        : lock_(cuda_precision_mutex),
          previous_cudnn_(at::globalContext().allowTF32CuDNN()),
          previous_cublas_(at::globalContext().allowTF32CuBLAS()) {
        at::globalContext().setAllowTF32CuDNN(tf32);
        at::globalContext().setAllowTF32CuBLAS(tf32);
    }
    ~CudaPrecisionGuard() {
        at::globalContext().setAllowTF32CuDNN(previous_cudnn_);
        at::globalContext().setAllowTF32CuBLAS(previous_cublas_);
    }
private:
    std::unique_lock<std::mutex> lock_;
    bool previous_cudnn_, previous_cublas_;
};

class TrainingAutocast {
public:
    explicit TrainingAutocast(bool enabled)
        : was_enabled_(at::autocast::is_autocast_enabled(at::kCUDA)),
          previous_dtype_(at::autocast::get_autocast_dtype(at::kCUDA)) {
        at::autocast::increment_nesting();
        at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCUDA, enabled);
    }
    ~TrainingAutocast() {
        at::autocast::set_autocast_enabled(at::kCUDA, was_enabled_);
        at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype_);
        if (at::autocast::decrement_nesting() == 0) at::autocast::clear_cache();
    }
private:
    bool was_enabled_;
    at::ScalarType previous_dtype_;
};

// Two 3x3 convolutions with a skip connection, each followed by batch norm.
struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int channels)
        : conv_a(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
          bn_a(channels),
          conv_b(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
          bn_b(channels) {
        register_module("conv_a", conv_a);
        register_module("bn_a", bn_a);
        register_module("conv_b", conv_b);
        register_module("bn_b", bn_b);

    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor out = torch::relu(bn_a(conv_a(x)));
        out = bn_b(conv_b(out));
        return torch::relu(x + out);
    }

    torch::nn::Conv2d conv_a{nullptr}, conv_b{nullptr};
    torch::nn::BatchNorm2d bn_a{nullptr}, bn_b{nullptr};
};
TORCH_MODULE(ResBlock);

struct CudaNetImpl : torch::nn::Module {
    CudaNetImpl() {
        stem = register_module(
            "stem",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                  NetConfig::kInputChannels, kChannels, 3)
                                  .padding(1)));
        stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(kChannels));
        for (int i = 0; i < kResBlocks; ++i) {
            auto block = ResBlock(kChannels);
            register_module("res" + std::to_string(i), block);
            trunk.push_back(block);
        }
        policy_conv = register_module(
            "policy_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(kChannels, 2, 1)));
        policy_bn = register_module("policy_bn", torch::nn::BatchNorm2d(2));
        policy_fc = register_module(
            "policy_fc", torch::nn::Linear(2 * kCells, kCells));
        value_conv = register_module(
            "value_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(kChannels, 1, 1)));
        value_bn = register_module("value_bn", torch::nn::BatchNorm2d(1));
        value_fc1 = register_module(
            "value_fc1", torch::nn::Linear(kCells, NetConfig::kValueHidden));
        value_fc2 =
            register_module("value_fc2", torch::nn::Linear(NetConfig::kValueHidden, 1));


    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        torch::Tensor h = torch::relu(stem_bn(stem(x)));
        for (auto& block : trunk) h = block->forward(h);

        torch::Tensor p = torch::relu(policy_bn(policy_conv(h)));
        p = p.reshape({-1, 2 * kCells});
        p = torch::log_softmax(policy_fc(p).to(torch::kFloat32), 1);

        torch::Tensor v = torch::relu(value_bn(value_conv(h)));
        v = v.reshape({-1, kCells});
        v = torch::relu(value_fc1(v));
        // BF16 tanh can round to exactly +/-1 early and zero its derivative.
        // Keep both final head nonlinearities in FP32.
        v = torch::tanh(value_fc2(v).to(torch::kFloat32));
        return {p, v.to(torch::kFloat32)};
    }

    torch::nn::Conv2d stem{nullptr};
    torch::nn::BatchNorm2d stem_bn{nullptr};
    std::vector<ResBlock> trunk;
    torch::nn::Conv2d policy_conv{nullptr}, value_conv{nullptr};
    torch::nn::BatchNorm2d policy_bn{nullptr}, value_bn{nullptr};
    torch::nn::Linear policy_fc{nullptr}, value_fc1{nullptr}, value_fc2{nullptr};
};
TORCH_MODULE(CudaNet);

// Batch normalisation folded into its convolution, which is the on-disk form
// the native CPU path reads. Folding is an exact affine rewrite of inference,
// so both implementations see identical numbers.
void fold_bn(const torch::Tensor& weight, const torch::Tensor& bias,
             const torch::Tensor& gamma, const torch::Tensor& beta,
             const torch::Tensor& mean, const torch::Tensor& var, double eps,
             torch::Tensor& out_w, torch::Tensor& out_b) {
    const torch::Tensor inv_std = 1.0 / (var + eps).sqrt();
    const torch::Tensor scale = (gamma * inv_std).view({-1, 1, 1, 1});
    out_w = (weight * scale).contiguous();
    out_b = (beta - gamma * mean * inv_std + bias.flatten() * scale.flatten())
                .contiguous();
}

std::vector<float> to_vector(const torch::Tensor& tensor) {
    const torch::Tensor cpu = tensor.detach().to(torch::kCPU).contiguous();
    std::vector<float> out(static_cast<size_t>(cpu.numel()));
    std::memcpy(out.data(), cpu.data_ptr<float>(), out.size() * sizeof(float));
    return out;
}

bool write_bin(const std::string& path, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    const uint32_t count = static_cast<uint32_t>(data.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    // Small tensors can stay in the stream buffer until destruction. Check
    // flush/close explicitly so a failed final write cannot report success.
    f.flush();
    f.close();
    return static_cast<bool>(f);
}

bool read_bin(const std::string& path, std::vector<float>& data, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f || count != n) return false;
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(f);
}

}  // namespace

struct CudaNetworkBackend::Impl {
    explicit Impl(PureNet& network, double learning_rate)
        : owner(network), net(CudaNet()),
          fused_adam(environment_enabled("GOMOK_FUSED_ADAM")),
          tf32_training(environment_enabled("GOMOK_TF32")),
          tf32_inference(environment_enabled("GOMOK_INFERENCE_TF32")),
          pinned_inference(environment_enabled("GOMOK_PINNED_INFERENCE", false)) {
        try {
            available = torch::cuda::is_available();
            if (!available) return;
            device = torch::Device(torch::kCUDA, 0);
            // MCTS batch sizes vary, so cuDNN autotuning is a net loss here.
            {
                std::lock_guard<std::mutex> precision_lock(cuda_precision_mutex);
                at::globalContext().setBenchmarkCuDNN(false);
                if (environment_enabled("GOMOK_DETERMINISTIC", false))
                    at::globalContext().setDeterministicCuDNN(true);
            }
            net->to(device);
            for_each_conv([](const std::string&, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d&) {
                conv->weight.set_data(conv->weight.contiguous(torch::MemoryFormat::ChannelsLast));
                // BN removes any constant convolution bias during training.
                // Its mathematically zero gradient picks up reduction noise,
                // which Adam can amplify into updates nearly as large as lr.
                // Retain the tensor for old checkpoints/folded inference, but
                // stop optimizing this redundant parameter.
                conv->bias.set_requires_grad(false);
            });
            for (const auto& parameter : net->named_parameters())
                if (parameter.key().rfind("value_", 0) == 0)
                    value_parameters.insert(parameter.value().unsafeGetTensorImpl());
            optimizer = std::make_unique<torch::optim::Adam>(
                net->parameters(), torch::optim::AdamOptions(learning_rate)
                                       .weight_decay(owner.weight_decay()));
        } catch (const c10::Error& error) {
            std::cerr << "[net] CUDA initialization failed: "
                      << error.what_without_backtrace() << '\n';
            available = false;
            optimizer.reset();
        }
    }

    // Visits every convolution / batch-norm pair in a fixed order.
    template <typename Fn>
    void for_each_conv(Fn&& fn) {
        fn("stem", net->stem, net->stem_bn);
        for (int i = 0; i < kResBlocks; ++i) {
            ResBlock& block = net->trunk[static_cast<size_t>(i)];
            const std::string base = "res" + std::to_string(i);
            fn(base + "_a", block->conv_a, block->bn_a);
            fn(base + "_b", block->conv_b, block->bn_b);
        }
        fn("policy_conv1", net->policy_conv, net->policy_bn);
        fn("val_conv1", net->value_conv, net->value_bn);
    }

    template <typename Fn>
    void for_each_fc(Fn&& fn) {
        fn("act_fc1", net->policy_fc);
        fn("val_fc1", net->value_fc1);
        fn("val_fc2", net->value_fc2);
    }

    // Fold once per model generation. Self-play performs thousands of forwards
    // between updates, so this removes every batch-normalisation kernel there.
    void invalidate_inference() { inference_weights.clear(); inference_biases.clear(); }

    // BatchNorm updates its running statistics during forward(), before the
    // gradient validity check. Keep one compact snapshot so a rejected update
    // cannot poison all subsequent evaluation or a later checkpoint.
    torch::Tensor snapshot_batch_norm() {
        std::vector<torch::Tensor> statistics;
        for_each_conv([&](const std::string&, torch::nn::Conv2d&,
                          torch::nn::BatchNorm2d& bn) {
            statistics.push_back(bn->running_mean);
            statistics.push_back(bn->running_var);
        });
        return torch::cat(statistics);
    }

    void restore_batch_norm(const torch::Tensor& snapshot) {
        torch::NoGradGuard guard;
        int64_t offset = 0;
        for_each_conv([&](const std::string&, torch::nn::Conv2d&,
                          torch::nn::BatchNorm2d& bn) {
            for (auto* statistic : {&bn->running_mean, &bn->running_var}) {
                statistic->copy_(snapshot.narrow(0, offset, statistic->numel()));
                offset += statistic->numel();
            }
            bn->num_batches_tracked.fill_(step_count);
        });
    }

    std::pair<torch::Tensor, torch::Tensor> inference(torch::Tensor x) {
        if (inference_weights.empty()) {
            for_each_conv([&](const std::string&, torch::nn::Conv2d& conv,
                              torch::nn::BatchNorm2d& bn) {
                torch::Tensor w, b;
                fold_bn(conv->weight, conv->bias, bn->weight, bn->bias,
                        bn->running_mean, bn->running_var, bn->options.eps(), w, b);
                inference_weights.push_back(w.contiguous(torch::MemoryFormat::ChannelsLast));
                inference_biases.push_back(b);
            });
        }
        size_t layer = 0;
        auto conv = [&](const torch::Tensor& in, int padding) {
            const auto result = torch::conv2d(in, inference_weights[layer],
                                             inference_biases[layer], 1, padding);
            ++layer;
            return result;
        };
        auto h = torch::relu(conv(x, 1));
        for (int i = 0; i < kResBlocks; ++i) {
            auto a = torch::relu(conv(h, 1));
            h = torch::relu(h + conv(a, 1));
        }
        auto p = torch::relu(conv(h, 0)).reshape({-1, 2 * kCells});
        p = torch::log_softmax(net->policy_fc(p), 1);
        auto v = torch::relu(conv(h, 0)).reshape({-1, kCells});
        v = torch::tanh(net->value_fc2(torch::relu(net->value_fc1(v))));
        return {p, v};
    }

    // Keep LibTorch's standard Adam state/serialization while executing all
    // tensor updates in a fused CUDA kernel. No model/optimizer format change.
    void fused_step(bool value_supervised) {
        torch::NoGradGuard guard;
        for (auto& group : optimizer->param_groups()) {
            auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
            std::vector<torch::Tensor> params, grads, means, variances, maxima, steps;
            if (!adam_step.defined())
                adam_step = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
            adam_step.fill_(step_count + 1);
            if (value_supervised) {
                if (!value_adam_step.defined())
                    value_adam_step = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
                value_adam_step.fill_(value_step_count + 1);
            }
            for (auto& parameter : group.params()) {
                if (!parameter.grad().defined()) continue;
                auto& state_ptr = optimizer->state()[parameter.unsafeGetTensorImpl()];
                if (!state_ptr) {
                    auto state = std::make_unique<torch::optim::AdamParamState>();
                    state->exp_avg(torch::zeros_like(parameter));
                    state->exp_avg_sq(torch::zeros_like(parameter));
                    if (options.amsgrad()) state->max_exp_avg_sq(torch::zeros_like(parameter));
                    state_ptr = std::move(state);
                }
                auto& state = static_cast<torch::optim::AdamParamState&>(*state_ptr);
                // A policy-only batch must neither decay the unused value
                // branch nor advance its Adam bias-correction counter.
                const bool value_parameter = value_parameters.count(parameter.unsafeGetTensorImpl()) != 0;
                state.step(value_parameter ? value_step_count + 1 : step_count + 1);
                params.push_back(parameter);
                grads.push_back(parameter.grad());
                means.push_back(state.exp_avg());
                variances.push_back(state.exp_avg_sq());
                if (options.amsgrad()) maxima.push_back(state.max_exp_avg_sq());
                steps.push_back(value_parameter ? value_adam_step : adam_step);
            }
            at::_fused_adam_(params, grads, means, variances, maxima, steps,
                             options.lr(), std::get<0>(options.betas()),
                             std::get<1>(options.betas()), options.weight_decay(),
                             options.eps(), options.amsgrad(), false, {}, {});
        }
    }

    std::vector<torch::Tensor> inference_weights, inference_biases;
    // Reuse page-locked staging allocations across variable MCTS batch sizes.
    // forward() serializes access through CudaPrecisionGuard.
    torch::Tensor pinned_states, pinned_policy, pinned_value;
    int64_t pinned_capacity = 0;
    torch::Tensor adam_step, value_adam_step;
    std::unordered_set<void*> value_parameters;
    PureNet& owner;
    CudaNet net;
    torch::Device device{torch::kCPU};
    std::unique_ptr<torch::optim::Adam> optimizer;
    bool available = false;
    bool fused_adam = true;
    bool tf32_training = true;
    bool tf32_inference = true;
    bool pinned_inference = false;
    // LibTorch's C++ Adam does not expose its step counter, so it is tracked
    // here for the checkpoint metadata.
    int64_t step_count = 0;
    int64_t value_step_count = 0;
};

CudaNetworkBackend::CudaNetworkBackend(PureNet& owner, double learning_rate)
    : impl_(std::make_unique<Impl>(owner, learning_rate)) {}

CudaNetworkBackend::~CudaNetworkBackend() = default;

bool CudaNetworkBackend::available() const { return impl_->available; }

void CudaNetworkBackend::synchronize() const {
    if (impl_->available) torch::cuda::synchronize(impl_->device.index());
}

void CudaNetworkBackend::set_weight_decay(double weight_decay) {
    if (!impl_->available || !impl_->optimizer) return;
    for (auto& group : impl_->optimizer->param_groups())
        static_cast<torch::optim::AdamOptions&>(group.options())
            .weight_decay(weight_decay);
}

void CudaNetworkBackend::sync_to_cpu() {
    if (!impl_->available) return;
    PureNet& owner = impl_->owner;
    auto copy_conv = [](PureNet::ConvBlock& target, torch::nn::Conv2d& conv,
                        torch::nn::BatchNorm2d& bn) {
        target.conv.w = to_vector(conv->weight);
        target.conv.b = to_vector(conv->bias);
        target.bn_gamma = to_vector(bn->weight);
        target.bn_beta = to_vector(bn->bias);
        target.bn_mean = to_vector(bn->running_mean);
        target.bn_var = to_vector(bn->running_var);
    };
    copy_conv(owner.stem_, impl_->net->stem, impl_->net->stem_bn);
    for (int i = 0; i < kResBlocks; ++i) {
        copy_conv(owner.trunk_[i]->a, impl_->net->trunk[i]->conv_a, impl_->net->trunk[i]->bn_a);
        copy_conv(owner.trunk_[i]->b, impl_->net->trunk[i]->conv_b, impl_->net->trunk[i]->bn_b);
    }
    copy_conv(owner.policy_conv_, impl_->net->policy_conv, impl_->net->policy_bn);
    copy_conv(owner.value_conv_, impl_->net->value_conv, impl_->net->value_bn);
    auto copy_fc = [](FCLayer& target, torch::nn::Linear& fc) {
        target.w = to_vector(fc->weight);
        target.b = to_vector(fc->bias);
    };
    copy_fc(owner.policy_fc_, impl_->net->policy_fc);
    copy_fc(owner.value_fc1_, impl_->net->value_fc1);
    copy_fc(owner.value_fc2_, impl_->net->value_fc2);
    owner.rebuild_folded_weights();
    owner.loaded_ = true;
}

void CudaNetworkBackend::sync_from_cpu() {
    if (!impl_->available) return;
    torch::NoGradGuard guard;
    PureNet& owner = impl_->owner;
    auto copy_tensor = [](const std::vector<float>& source, torch::Tensor& target) {
        target.copy_(torch::from_blob(const_cast<float*>(source.data()), target.sizes(),
                                     torch::TensorOptions().dtype(torch::kFloat32)));
    };
    auto copy_conv = [&](const PureNet::ConvBlock& source, torch::nn::Conv2d& conv,
                         torch::nn::BatchNorm2d& bn) {
        copy_tensor(source.conv.w, conv->weight);
        copy_tensor(source.conv.b, conv->bias);
        copy_tensor(source.bn_gamma, bn->weight);
        copy_tensor(source.bn_beta, bn->bias);
        copy_tensor(source.bn_mean, bn->running_mean);
        copy_tensor(source.bn_var, bn->running_var);
        bn->num_batches_tracked.zero_();
    };
    copy_conv(owner.stem_, impl_->net->stem, impl_->net->stem_bn);
    for (int i = 0; i < kResBlocks; ++i) {
        copy_conv(owner.trunk_[i]->a, impl_->net->trunk[i]->conv_a, impl_->net->trunk[i]->bn_a);
        copy_conv(owner.trunk_[i]->b, impl_->net->trunk[i]->conv_b, impl_->net->trunk[i]->bn_b);
    }
    copy_conv(owner.policy_conv_, impl_->net->policy_conv, impl_->net->policy_bn);
    copy_conv(owner.value_conv_, impl_->net->value_conv, impl_->net->value_bn);
    auto copy_fc = [&](const FCLayer& source, torch::nn::Linear& fc) {
        copy_tensor(source.w, fc->weight);
        copy_tensor(source.b, fc->bias);
    };
    copy_fc(owner.policy_fc_, impl_->net->policy_fc);
    copy_fc(owner.value_fc1_, impl_->net->value_fc1);
    copy_fc(owner.value_fc2_, impl_->net->value_fc2);
    impl_->invalidate_inference();
}

bool CudaNetworkBackend::load(const std::string& directory) {
    if (!impl_->available) return false;
    torch::NoGradGuard guard;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> pending;
    auto prepare = [&](const std::string& file, torch::Tensor& target) {
        std::vector<float> data;
        if (!read_bin(directory + "/" + file, data, static_cast<size_t>(target.numel()))) return false;
        for (float x : data) if (!std::isfinite(x)) return false;
        pending.emplace_back(target, torch::from_blob(data.data(), target.sizes(),
                             torch::TensorOptions().dtype(torch::kFloat32)).clone());
        return true;
    };
    bool ok = true;
    impl_->for_each_conv([&](const std::string& name, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d&) {
        if (ok) ok = prepare(name + "_weight.bin", conv->weight) &&
                     prepare(name + "_bias.bin", conv->bias);
    });
    impl_->for_each_fc([&](const std::string& name, torch::nn::Linear& fc) {
        if (ok) ok = prepare(name + "_weight.bin", fc->weight) &&
                     prepare(name + "_bias.bin", fc->bias);
    });
    if (!ok) return false;
    for (auto& pair : pending) pair.first.copy_(pair.second);
    impl_->for_each_conv([&](const std::string&, torch::nn::Conv2d&,
                             torch::nn::BatchNorm2d& bn) {
        bn->weight.fill_(1.0);
        bn->bias.zero_();
        bn->running_mean.zero_();
        // Folded weights already include epsilon; make BN exactly identity.
        bn->running_var.fill_(1.0 - bn->options.eps());
        bn->num_batches_tracked.zero_();
    });
    impl_->optimizer->state().clear();
    impl_->step_count = 0;
    impl_->value_step_count = 0;
    impl_->invalidate_inference();
    return true;
}

bool CudaNetworkBackend::save(const std::string& directory) const {
    std::error_code directory_error;
    std::filesystem::create_directories(directory, directory_error);
    if (directory_error) return false;
    if (!impl_->available) return false;
    torch::NoGradGuard no_grad;
    bool ok = true;
    impl_->for_each_conv([&](const std::string& name, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d& bn) {
        if (!ok) return;
        torch::Tensor w, b;
        fold_bn(conv->weight, conv->bias, bn->weight, bn->bias,
                bn->running_mean, bn->running_var, bn->options.eps(), w, b);
        ok = write_bin(directory + "/" + name + "_weight.bin", to_vector(w)) &&
             write_bin(directory + "/" + name + "_bias.bin", to_vector(b));
    });
    impl_->for_each_fc([&](const std::string& name, torch::nn::Linear& fc) {
        if (!ok) return;
        ok = write_bin(directory + "/" + name + "_weight.bin",
                       to_vector(fc->weight)) &&
             write_bin(directory + "/" + name + "_bias.bin", to_vector(fc->bias));
    });
    return ok;
}

void CudaNetworkBackend::forward(const float* states, int batch, float* policy,
                                 float* value, bool probabilities) const {
    torch::InferenceMode inference_mode;
    CudaPrecisionGuard precision(impl_->tf32_inference);
    // TF32 is the fast default. GOMOK_INFERENCE_TF32=0 selects strict fp32
    // (both cuDNN and cuBLAS) for CPU parity or precision-sensitive deployment.
    std::optional<at::NoTF32Guard> no_tf32;
    if (!impl_->tf32_inference) no_tf32.emplace();
    impl_->net->eval();
    if (impl_->pinned_inference) {
        if (batch > impl_->pinned_capacity) {
            const auto pinned = torch::TensorOptions().dtype(torch::kFloat32)
                .device(torch::kCPU).pinned_memory(true);
            impl_->pinned_states = torch::empty(
                {batch, NetConfig::kInputChannels, 15, 15}, pinned);
            impl_->pinned_policy = torch::empty({batch, kCells}, pinned);
            impl_->pinned_value = torch::empty({batch}, pinned);
            impl_->pinned_capacity = batch;
        }
        auto host_states = impl_->pinned_states.narrow(0, 0, batch);
        std::memcpy(host_states.data_ptr<float>(), states,
                    static_cast<size_t>(batch) * NetConfig::kInputChannels *
                        kCells * sizeof(float));
        auto device_states = host_states.to(impl_->device, true)
            .contiguous(torch::MemoryFormat::ChannelsLast);
        auto [log_policy, values] = impl_->inference(device_states);
        auto output_policy = probabilities ? log_policy.exp() : log_policy;
        impl_->pinned_policy.narrow(0, 0, batch).copy_(output_policy, true);
        impl_->pinned_value.narrow(0, 0, batch)
            .copy_(values.reshape({batch}), true);
        // Search consumes both outputs before it can submit another batch.
        // Synchronize before reading the pinned host buffers.
        torch::cuda::synchronize(impl_->device.index());
        std::memcpy(policy, impl_->pinned_policy.data_ptr<float>(),
                    static_cast<size_t>(batch) * kCells * sizeof(float));
        std::memcpy(value, impl_->pinned_value.data_ptr<float>(),
                    static_cast<size_t>(batch) * sizeof(float));
        return;
    }
    auto cpu_states = torch::from_blob(
        const_cast<float*>(states), {batch, NetConfig::kInputChannels, 15, 15},
        torch::TensorOptions().dtype(torch::kFloat32));
    auto [log_policy, values] =
        impl_->inference(cpu_states.to(impl_->device).contiguous(torch::MemoryFormat::ChannelsLast));
    torch::Tensor output_policy =
        probabilities ? log_policy.exp() : log_policy;
    output_policy = output_policy.to(torch::kCPU).contiguous();
    values = values.reshape({batch}).to(torch::kCPU).contiguous();
    std::memcpy(policy, output_policy.data_ptr<float>(),
                static_cast<size_t>(batch) * kCells * sizeof(float));
    std::memcpy(value, values.data_ptr<float>(),
                static_cast<size_t>(batch) * sizeof(float));
}

bool CudaNetworkBackend::has_training_state(
    const std::string& directory) const {
    std::ifstream meta(directory + "/training.meta");
    return meta.is_open();
}

bool CudaNetworkBackend::save_training_state(
    const std::string& directory, int batch_index, int64_t step,
    const std::vector<float>& moments_m,
    const std::vector<float>& moments_v) const {
    std::error_code directory_error;
    std::filesystem::create_directories(directory, directory_error);
    if (directory_error) return false;
    if (!impl_->available) return false;
    torch::NoGradGuard no_grad;
    bool ok = true;
    size_t tensor_index = 0;

    // Raw parameterisation: the convolution keeps its own weights and the
    // batch norm keeps gamma/beta/running statistics separately. Folding them
    // is exact for inference but a folded file cannot resume training
    // faithfully, because it no longer represents the same parameters.
    impl_->for_each_conv([&](const std::string& name, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d& bn) {
        if (!ok) return;
        const std::string prefix =
            "training_conv" + std::to_string(tensor_index++) + "_";
        ok = write_bin(directory + "/" + prefix + "weight.bin",
                       to_vector(conv->weight)) &&
             write_bin(directory + "/" + prefix + "bias.bin",
                       to_vector(conv->bias)) &&
             write_bin(directory + "/" + prefix + "bn_gamma.bin",
                       to_vector(bn->weight)) &&
             write_bin(directory + "/" + prefix + "bn_beta.bin",
                       to_vector(bn->bias)) &&
             write_bin(directory + "/" + prefix + "bn_mean.bin",
                       to_vector(bn->running_mean)) &&
             write_bin(directory + "/" + prefix + "bn_var.bin",
                       to_vector(bn->running_var));
    });
    size_t fc_index = 0;
    impl_->for_each_fc([&](const std::string& name, torch::nn::Linear& fc) {
        if (!ok) return;
        const std::string prefix =
            "training_fc" + std::to_string(fc_index++) + "_";
        ok = write_bin(directory + "/" + prefix + "weight.bin",
                       to_vector(fc->weight)) &&
             write_bin(directory + "/" + prefix + "bias.bin",
                       to_vector(fc->bias));
    });
    if (!ok) return false;

    // The optimiser holds the Adam moments. Resuming without them restarts the
    // moment estimates from zero, which changes the effective step size for
    // thousands of updates, so they are part of the checkpoint.
    int64_t saved_step = step;
    if (impl_->optimizer) {
        try {
            // PyTorchStreamWriter can throw again from its destructor after
            // a disk write fails, terminating the process before our catch
            // runs. Finish its archive entirely in memory, then handle disk
            // errors with an ordinary stream whose close is checked below.
            size_t reserved_bytes = 256 * 1024;
            for (const auto& item : impl_->optimizer->state()) {
                const auto& state = static_cast<const torch::optim::AdamParamState&>(*item.second);
                for (const auto& tensor : {state.exp_avg(), state.exp_avg_sq(), state.max_exp_avg_sq()})
                    if (tensor.defined()) reserved_bytes += tensor.nbytes();
            }
            std::vector<char> serialized;
            serialized.reserve(reserved_bytes);
            {
                torch::serialize::OutputArchive archive;
                impl_->optimizer->save(archive);
                archive.save_to([&](const void* data, size_t bytes) -> size_t {
                    const size_t offset = serialized.size();
                    serialized.resize(offset + bytes);
                    if (data != nullptr && bytes > 0)
                        std::memcpy(serialized.data() + offset, data, bytes);
                    return bytes;
                });
            }
            std::ofstream optimizer_file(directory + "/training_optimizer.pt", std::ios::binary);
            if (!optimizer_file.is_open()) return false;
            optimizer_file.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
            optimizer_file.flush();
            const bool written = static_cast<bool>(optimizer_file);
            optimizer_file.close();
            if (!written || !optimizer_file) return false;
            if (saved_step == 0) saved_step = impl_->step_count;
        } catch (const c10::Error& error) {
            std::cerr << "[net] optimiser state save failed: "
                      << error.what_without_backtrace() << '\n';
            ok = false;
        }
    }
    if (!ok) return false;

    std::ofstream meta(directory + "/training.meta");
    if (!meta.is_open()) return false;
    meta << "version 2\n";
    meta << "batch_index " << batch_index << "\n";
    meta << "optimizer_step " << saved_step << "\n";
    meta << "conv_count " << tensor_index << "\n";
    meta << "fc_count " << fc_index << "\n";
    // Retained so a version-1 reader still sees the keys it expects.
    meta << "moments_m " << moments_m.size() << "\n";
    meta << "moments_v " << moments_v.size() << "\n";
    meta << "grad_clip " << std::setprecision(17) << impl_->owner.grad_clip() << "\n";
    meta << "value_optimizer_step " << impl_->value_step_count << "\n";
    meta.close();
    if (!meta) return false;

    // The inference export travels with the checkpoint so MCTS and `human`
    // can consume the same directory.
    return save(directory);
}

bool CudaNetworkBackend::load_training_state(
    const std::string& directory, int* batch_index, int64_t* step,
    std::vector<float>* moments_m, std::vector<float>* moments_v) {
    if (!impl_->available) return false;
    std::ifstream meta(directory + "/training.meta");
    auto read_field = [&](const char* expected, auto& value) {
        std::string key;
        return static_cast<bool>(meta >> key >> value) && key == expected;
    };
    int version = 0, batches = 0, conv_count = 0, fc_count = 0;
    int64_t optimizer_step = 0;
    size_t m_size = 0, v_size = 0;
    if (!read_field("version", version) || (version != 1 && version != 2) ||
        !read_field("batch_index", batches) || batches < 0 ||
        !read_field("optimizer_step", optimizer_step) || optimizer_step < 0 ||
        !read_field("conv_count", conv_count) || conv_count != 3 + 2 * kResBlocks ||
        !read_field("fc_count", fc_count) || fc_count != 3 ||
        !read_field("moments_m", m_size) || !read_field("moments_v", v_size))
        return false;
    // Older checkpoints updated every value parameter even on policy-only
    // batches, so their value counter equals the global optimizer counter.
    double grad_clip = impl_->owner.grad_clip();
    int64_t value_optimizer_step = optimizer_step;
    bool seen_clip = false, seen_value_step = false;
    std::string optional_key;
    while (meta >> optional_key) {
        if (optional_key == "grad_clip" && !seen_clip) {
            if (!(meta >> grad_clip) || !std::isfinite(grad_clip) || grad_clip < 0.0) return false;
            seen_clip = true;
        } else if (optional_key == "value_optimizer_step" && !seen_value_step) {
            if (!(meta >> value_optimizer_step) || value_optimizer_step < 0 ||
                value_optimizer_step > optimizer_step) return false;
            seen_value_step = true;
        } else return false;
    }
    if (!meta.eof()) return false;

    // Validate the whole checkpoint before touching live parameters. A failed
    // load must leave the current model and Adam moments usable.
    torch::NoGradGuard no_grad;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> pending;
    auto prepare = [&](const std::string& path, torch::Tensor& target, bool variance = false) {
        std::vector<float> data;
        if (!read_bin(path, data, static_cast<size_t>(target.numel()))) return false;
        for (float x : data)
            if (!std::isfinite(x) || (variance && x < 0.0f)) return false;
        pending.emplace_back(target, torch::from_blob(data.data(), target.sizes(),
                              torch::TensorOptions().dtype(torch::kFloat32)).clone());
        return true;
    };
    bool ok = true;
    size_t index = 0;
    impl_->for_each_conv([&](const std::string&, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d& bn) {
        if (!ok) return;
        const auto prefix = directory + "/training_conv" + std::to_string(index++) + "_";
        ok = prepare(prefix + "weight.bin", conv->weight) &&
             prepare(prefix + "bias.bin", conv->bias) &&
             prepare(prefix + "bn_gamma.bin", bn->weight) &&
             prepare(prefix + "bn_beta.bin", bn->bias) &&
             prepare(prefix + "bn_mean.bin", bn->running_mean) &&
             prepare(prefix + "bn_var.bin", bn->running_var, true);
    });
    index = 0;
    impl_->for_each_fc([&](const std::string&, torch::nn::Linear& fc) {
        if (!ok) return;
        const auto prefix = directory + "/training_fc" + std::to_string(index++) + "_";
        ok = prepare(prefix + "weight.bin", fc->weight) && prepare(prefix + "bias.bin", fc->bias);
    });
    if (!ok) return false;
    auto optimizer = std::make_unique<torch::optim::Adam>(impl_->net->parameters(),
                          torch::optim::AdamOptions(0.001).weight_decay(impl_->owner.weight_decay()));
    const auto optimizer_path = directory + "/training_optimizer.pt";
    if (std::filesystem::exists(optimizer_path)) {
        try {
            torch::serialize::InputArchive archive;
            archive.load_from(optimizer_path, impl_->device);
            optimizer->load(archive);
        } catch (const c10::Error& error) {
            std::cerr << "[net] optimiser state load failed: "
                      << error.what_without_backtrace() << '\n';
            return false;
        }
    } else if (version >= 2 || optimizer_step > 0) {
        std::cerr << "[net] checkpoint is missing Adam state: " << optimizer_path << '\n';
        return false;
    }
    // Deserialization alone does not validate tensor shapes, finite moments,
    // or missing parameter states. Accepting any of those silently damages the
    // very next update, sometimes long after the apparent successful resume.
    const auto parameters = impl_->net->parameters();
    if (optimizer->param_groups().size() != 1 ||
        optimizer->param_groups().front().params().size() != parameters.size() ||
        optimizer->state().size() > parameters.size() ||
        (optimizer_step == 0 && !optimizer->state().empty())) return false;
    const auto& loaded_options = static_cast<const torch::optim::AdamOptions&>(
        optimizer->param_groups().front().options());
    if (!std::isfinite(loaded_options.lr()) || loaded_options.lr() <= 0.0 ||
        !std::isfinite(loaded_options.eps()) || loaded_options.eps() < 0.0 ||
        !std::isfinite(loaded_options.weight_decay()) || loaded_options.weight_decay() < 0.0)
        return false;
    for (double beta : {std::get<0>(loaded_options.betas()), std::get<1>(loaded_options.betas())})
        if (!std::isfinite(beta) || beta < 0.0 || beta >= 1.0) return false;
    size_t recognized_states = 0;
    std::vector<void*> obsolete_bias_states;
    for (const auto& parameter : parameters) {
        const bool value_parameter = impl_->value_parameters.count(parameter.unsafeGetTensorImpl()) != 0;
        const int64_t expected_step = value_parameter ? value_optimizer_step : optimizer_step;
        const auto found = optimizer->state().find(parameter.unsafeGetTensorImpl());
        if (found == optimizer->state().end()) {
            if (expected_step > 0 && parameter.requires_grad()) return false;
            continue;
        }
        ++recognized_states;
        if (!found->second) return false;
        const auto& state = static_cast<const torch::optim::AdamParamState&>(*found->second);
        if (expected_step == 0 || state.step() != expected_step) return false;
        auto valid_moment = [&](const torch::Tensor& tensor, bool nonnegative) {
            return tensor.defined() && tensor.sizes() == parameter.sizes() &&
                   tensor.scalar_type() == parameter.scalar_type() &&
                   tensor.device() == parameter.device() &&
                   torch::isfinite(tensor).all().item<bool>() &&
                   (!nonnegative || (tensor >= 0).all().item<bool>());
        };
        if (!valid_moment(state.exp_avg(), false) ||
            !valid_moment(state.exp_avg_sq(), true) ||
            (loaded_options.amsgrad() && !valid_moment(state.max_exp_avg_sq(), true)))
            return false;
        if (!parameter.requires_grad()) obsolete_bias_states.push_back(found->first);
    }
    if (recognized_states != optimizer->state().size()) return false;
    // Older checkpoints optimized the redundant conv biases. Their tensor
    // values remain intact; obsolete moments are no longer part of training.
    for (void* key : obsolete_bias_states) optimizer->state().erase(key);
    for (auto& pair : pending) pair.first.copy_(pair.second);
    impl_->optimizer = std::move(optimizer);
    impl_->step_count = optimizer_step;
    impl_->value_step_count = value_optimizer_step;
    impl_->invalidate_inference();
    impl_->for_each_conv([&](const std::string&, torch::nn::Conv2d&, torch::nn::BatchNorm2d& bn) {
        bn->num_batches_tracked.fill_(optimizer_step);
    });
    const auto& options = static_cast<const torch::optim::AdamOptions&>(
        impl_->optimizer->param_groups().front().options());
    impl_->owner.weight_decay_ = options.weight_decay();
    impl_->owner.grad_clip_ = grad_clip;
    if (batch_index) *batch_index = batches;
    if (step) *step = optimizer_step;
    if (moments_m) moments_m->clear();
    if (moments_v) moments_v->clear();
    return true;
}

PureNet::TrainStats CudaNetworkBackend::train_step(
    const std::vector<float>& states, int batch,
    const std::vector<float>& probs, const std::vector<float>& winners,
    double learning_rate, double value_loss_weight,
    const std::vector<float>& value_weights) {
    CudaPrecisionGuard precision(impl_->tf32_training);
    std::optional<at::NoTF32Guard> full_precision;
    if (!impl_->tf32_training) full_precision.emplace();
    impl_->net->train();
    for (auto& group : impl_->optimizer->param_groups())
        static_cast<torch::optim::AdamOptions&>(group.options())
            .lr(learning_rate);
    // Undefined gradients make Adam skip both moments and weight decay for
    // an unsupervised value branch, including after a supervised update.
    impl_->optimizer->zero_grad(true);
    const auto batch_norm_snapshot = impl_->snapshot_batch_norm();
    const double value_weight_total = value_weights.empty() ? static_cast<double>(batch) :
        std::accumulate(value_weights.begin(), value_weights.end(), 0.0);
    const bool value_supervised = value_loss_weight > 0.0 && value_weight_total > 0.0;

    try {
        auto state_tensor =
            torch::from_blob(const_cast<float*>(states.data()),
                             {batch, NetConfig::kInputChannels, 15, 15},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .to(impl_->device).contiguous(torch::MemoryFormat::ChannelsLast);
        auto policy_target =
            torch::from_blob(const_cast<float*>(probs.data()), {batch, kCells},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .to(impl_->device);
        auto value_target =
            torch::from_blob(const_cast<float*>(winners.data()), {batch},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .to(impl_->device);

        auto outputs = [&] {
            TrainingAutocast autocast(impl_->owner.mixed_precision());
            return impl_->net->forward(state_tensor);
        }();
        auto& log_policy = outputs.first;
        auto& values = outputs.second;
        const auto value_errors = (values.reshape({batch}) - value_target).square();
        auto value_loss = value_errors.mean();
        if (!value_weights.empty()) {
            // Normalize in host double precision: tiny confidence weights must not
            // dilute the value objective, and a sum of large finite weights must
            // not overflow fp32. Zero total weight cleanly disables this head.
            std::vector<float> normalized(value_weights.size(), 0.0f);
            if (value_weight_total > 0.0)
                for (size_t i = 0; i < normalized.size(); ++i)
                    normalized[i] = static_cast<float>(value_weights[i] / value_weight_total);
            auto weights = torch::from_blob(normalized.data(), {batch},
                                            torch::TensorOptions().dtype(torch::kFloat32))
                               .to(impl_->device);
            value_loss = (weights * value_errors).sum();
        }
        const auto policy_loss =
            -torch::mean(torch::sum(policy_target * log_policy, 1));
        // Multiplying by zero still creates zero gradients. Adam then applies
        // weight decay and destroys the untrained value features over many
        // policy-only updates. Keep this branch out of autograd altogether.
        const auto loss = value_supervised ? policy_loss + value_loss * value_loss_weight : policy_loss;
        loss.backward();
        std::vector<torch::Tensor> gradients;
        for (const auto& parameter : impl_->net->parameters())
            if (parameter.grad().defined()) gradients.push_back(parameter.grad());
        const auto grad_norm = torch::stack(at::_foreach_norm(gradients)).norm();
        torch::NoGradGuard metrics_guard;
        const auto entropy = -torch::mean(
            torch::sum(torch::exp(log_policy.detach()) * log_policy.detach(), 1));

        const torch::Tensor metrics =
            torch::stack({loss.detach(), entropy.detach(), policy_loss.detach(),
                          value_loss.detach(), grad_norm.detach()})
                .to(torch::kCPU)
                .contiguous();
        const float* result = metrics.data_ptr<float>();
        if (!std::isfinite(result[0]) || !std::isfinite(result[1]) ||
            !std::isfinite(result[2]) || !std::isfinite(result[3]) || !std::isfinite(result[4])) {
            throw std::runtime_error("non-finite training loss/gradient; update rejected");
        }
        const double clip = impl_->owner.grad_clip();
        if (clip > 0.0)
            at::_foreach_mul_(gradients, (clip / (grad_norm + 1e-6)).clamp_max(1.0));
        if (impl_->fused_adam) impl_->fused_step(value_supervised);
        else impl_->optimizer->step();
        impl_->invalidate_inference();
        ++impl_->step_count;
        if (value_supervised) ++impl_->value_step_count;
        return {result[0], result[1], result[2], result[3], result[4]};
    } catch (...) {
        impl_->restore_batch_norm(batch_norm_snapshot);
        impl_->optimizer->zero_grad();
        throw;
    }
}

}  // namespace gomoku

#endif  // GOMOK_HAS_TORCH
