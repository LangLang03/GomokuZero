#include "network_cuda.h"
#include "net_config.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef GOMOK_HAS_TORCH
#include <torch/torch.h>

namespace gomoku {
namespace {

constexpr int kCells = NetConfig::kCells;
constexpr int kChannels = NetConfig::kChannels;
constexpr int kResBlocks = NetConfig::kResBlocks;

// Uniform init on [-bound, bound] with bound = gain * sqrt(6 / fan_in),
// which is the He/Glorot family without depending on the enum-typed
// initialiser API (whose spelling changes between LibTorch releases).
torch::Tensor uniform_tensor(torch::IntArrayRef sizes, double fan_in,
                             double gain, torch::Generator& generator) {
    const double bound = gain * std::sqrt(6.0 / std::max(1.0, fan_in));
    return torch::rand(sizes, generator, torch::TensorOptions().dtype(torch::kFloat32))
               .mul(2.0 * bound)
               .sub(bound);
}


// Two 3x3 convolutions with a skip connection, each followed by batch norm.
struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int channels, double residual_gain)
        : conv_a(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
          bn_a(channels),
          conv_b(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
          bn_b(channels) {
        register_module("conv_a", conv_a);
        register_module("bn_a", bn_a);
        register_module("conv_b", conv_b);
        register_module("bn_b", bn_b);
        auto generator = torch::make_generator<torch::CPUGeneratorImpl>(
            static_cast<uint64_t>(channels) * 7919u + 17u);
        torch::NoGradGuard no_grad;
        const double fan_in = static_cast<double>(channels) * 9.0;
        conv_a->weight.copy_(uniform_tensor(conv_a->weight.sizes(), fan_in,
                                            1.0, generator));
        conv_a->bias.zero_();
        conv_b->weight.copy_(uniform_tensor(conv_b->weight.sizes(), fan_in,
                                            residual_gain, generator));
        conv_b->bias.zero_();
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
            auto block = ResBlock(kChannels, 0.1);
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

        auto generator =
            torch::make_generator<torch::CPUGeneratorImpl>(20240816u);
        torch::NoGradGuard no_grad;
        const double stem_fan_in =
            static_cast<double>(NetConfig::kInputChannels) * 9.0;
        stem->weight.copy_(
            uniform_tensor(stem->weight.sizes(), stem_fan_in, 1.0, generator));
        stem->bias.zero_();
        for (auto& head : {policy_conv, value_conv}) {
            const double fan_in = static_cast<double>(kChannels) *
                                  static_cast<double>(head->weight.size(2) *
                                                      head->weight.size(3));
            head->weight.copy_(
                uniform_tensor(head->weight.sizes(), fan_in, 1.0, generator));
            head->bias.zero_();
        }
        for (auto* fc : {&policy_fc, &value_fc1, &value_fc2}) {
            const double fan_in = static_cast<double>((*fc)->weight.size(1));
            const double fan_out = static_cast<double>((*fc)->weight.size(0));
            (*fc)->weight.copy_(uniform_tensor((*fc)->weight.sizes(),
                                               fan_in + fan_out, 1.0,
                                               generator));
            (*fc)->bias.zero_();
        }
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        torch::Tensor h = torch::relu(stem_bn(stem(x)));
        for (auto& block : trunk) h = block->forward(h);

        torch::Tensor p = torch::relu(policy_bn(policy_conv(h)));
        p = p.reshape({-1, 2 * kCells});
        p = torch::log_softmax(policy_fc(p), 1);

        torch::Tensor v = torch::relu(value_bn(value_conv(h)));
        v = v.reshape({-1, kCells});
        v = torch::relu(value_fc1(v));
        v = torch::tanh(value_fc2(v));
        return {p, v};
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
        : owner(network), net(CudaNet()) {
        try {
            available = torch::cuda::is_available();
            if (!available) return;
            device = torch::Device(torch::kCUDA, 0);
            // MCTS batch sizes vary, so cuDNN autotuning is a net loss here.
            at::globalContext().setBenchmarkCuDNN(false);
            at::globalContext().setAllowTF32CuDNN(true);
            at::globalContext().setAllowTF32CuBLAS(true);
            net->to(device);
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

    PureNet& owner;
    CudaNet net;
    torch::Device device{torch::kCPU};
    std::unique_ptr<torch::optim::Adam> optimizer;
    bool available = false;
    // Set once any optimisation step has run, so a backend switch that would
    // discard training state can be refused rather than silently corrupting.
    bool trained = false;
};

CudaNetworkBackend::CudaNetworkBackend(PureNet& owner, double learning_rate)
    : impl_(std::make_unique<Impl>(owner, learning_rate)) {}

CudaNetworkBackend::~CudaNetworkBackend() = default;

bool CudaNetworkBackend::available() const { return impl_->available; }

void CudaNetworkBackend::set_weight_decay(double weight_decay) {
    if (!impl_->available || !impl_->optimizer) return;
    for (auto& group : impl_->optimizer->param_groups())
        static_cast<torch::optim::AdamOptions&>(group.options())
            .weight_decay(weight_decay);
}

void CudaNetworkBackend::sync_to_cpu() {
    if (!impl_->available) return;
    PureNet& owner = impl_->owner;

    // Export the raw parameterisation (convolution weights plus batch-norm
    // gamma/beta/running statistics) and let the native loader fold it. Going
    // through the same checkpoint format the trainer writes keeps one code
    // path responsible for the conversion.
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path staging =
        fs::temp_directory_path(error) / "gomoku_cuda_to_cpu";
    if (error) return;
    if (!save_training_state(staging.string(), 0, 0, {}, {})) return;
    if (owner.load_native(staging.string())) owner.loaded_ = true;
    fs::remove_all(staging, error);
}

void CudaNetworkBackend::sync_from_cpu() {
    if (!impl_->available) return;
    PureNet& owner = impl_->owner;

    // The native side only ever holds folded weights, so fold-then-load is the
    // only meaningful direction of data flow. Refuse instead of silently
    // dropping batch-norm statistics and optimiser state.
    if (impl_->trained) {
        throw std::logic_error(
            "cannot switch to CUDA after training: the native side holds only "
            "folded inference weights, so batch-norm statistics and optimiser "
            "moments would be lost. Reload the checkpoint and keep one backend "
            "for the whole run.");
    }

    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path staging =
        fs::temp_directory_path(error) / "gomoku_cpu_to_cuda";
    if (error) return;
    if (!owner.export_inference(staging.string())) return;
    load(staging.string());
    fs::remove_all(staging, error);
}

bool CudaNetworkBackend::load(const std::string& directory) {
    if (!impl_->available) return false;
    torch::NoGradGuard no_grad;
    bool ok = true;
    impl_->for_each_conv([&](const std::string& name, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d& bn) {
        if (!ok) return;
        const size_t weight_n = static_cast<size_t>(conv->weight.numel());
        const size_t bias_n = static_cast<size_t>(conv->bias.numel());
        std::vector<float> w, b;
        if (!read_bin(directory + "/" + name + "_weight.bin", w, weight_n) ||
            !read_bin(directory + "/" + name + "_bias.bin", b, bias_n)) {
            ok = false;
            return;
        }
        auto weight = torch::from_blob(w.data(), conv->weight.sizes(),
                                       torch::TensorOptions().dtype(torch::kFloat32));
        auto bias = torch::from_blob(b.data(), conv->bias.sizes(),
                                     torch::TensorOptions().dtype(torch::kFloat32));
        conv->weight.copy_(weight.to(impl_->device));
        conv->bias.copy_(bias.to(impl_->device));
        // The stored tensors already have batch norm folded in. Setting
        // running_var to 1 - eps makes the fold factor exactly
        // 1/sqrt((1-eps)+eps) = 1, so re-saving reproduces the same numbers
        // instead of scaling them by 1/sqrt(1+eps) on every round trip.
        bn->weight.fill_(1.0);
        bn->bias.fill_(0.0);
        bn->running_mean.fill_(0.0);
        bn->running_var.fill_(1.0 - bn->options.eps());
    });
    impl_->for_each_fc([&](const std::string& name, torch::nn::Linear& fc) {
        if (!ok) return;
        const size_t weight_n = static_cast<size_t>(fc->weight.numel());
        const size_t bias_n = static_cast<size_t>(fc->bias.numel());
        std::vector<float> w, b;
        if (!read_bin(directory + "/" + name + "_weight.bin", w, weight_n) ||
            !read_bin(directory + "/" + name + "_bias.bin", b, bias_n)) {
            ok = false;
            return;
        }
        auto weight = torch::from_blob(w.data(), fc->weight.sizes(),
                                       torch::TensorOptions().dtype(torch::kFloat32));
        auto bias = torch::from_blob(b.data(), fc->bias.sizes(),
                                     torch::TensorOptions().dtype(torch::kFloat32));
        fc->weight.copy_(weight.to(impl_->device));
        fc->bias.copy_(bias.to(impl_->device));
    });
    return ok;
}

bool CudaNetworkBackend::save(const std::string& directory) const {
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
    // TF32 keeps only 10 mantissa bits. Across a 25-convolution trunk that
    // accumulates into a visible divergence from the native CPU path, so
    // inference stays in full fp32 and only training opts into TF32.
    at::NoTF32Guard no_tf32;
    impl_->net->eval();
    auto cpu_states = torch::from_blob(
        const_cast<float*>(states), {batch, NetConfig::kInputChannels, 15, 15},
        torch::TensorOptions().dtype(torch::kFloat32));
    auto [log_policy, values] =
        impl_->net->forward(cpu_states.to(impl_->device));
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

    std::ofstream meta(directory + "/training.meta");
    if (!meta.is_open()) return false;
    meta << "version 1\n";
    meta << "batch_index " << batch_index << "\n";
    meta << "optimizer_step " << step << "\n";
    meta << "conv_count " << tensor_index << "\n";
    meta << "fc_count " << fc_index << "\n";
    meta << "moments_m " << moments_m.size() << "\n";
    meta << "moments_v " << moments_v.size() << "\n";
    meta.close();
    if (!moments_m.empty())
        ok = write_bin(directory + "/training_moments_m.bin", moments_m);
    if (ok && !moments_v.empty())
        ok = write_bin(directory + "/training_moments_v.bin", moments_v);
    if (!ok) return false;

    // The inference export travels with the checkpoint so MCTS and `human`
    // can consume the same directory.
    return save(directory);
}

bool CudaNetworkBackend::load_training_state(
    const std::string& directory, int* batch_index, int64_t* step,
    std::vector<float>* moments_m, std::vector<float>* moments_v) {
    if (!impl_->available) return false;
    std::ifstream meta(directory + "/training.meta");
    if (!meta.is_open()) return false;
    std::string key;
    int version = 0, batches = 0, conv_count = 0, fc_count = 0;
    int64_t optimizer_step = 0;
    size_t m_size = 0, v_size = 0;
    meta >> key >> version;
    if (version != 1) return false;
    meta >> key >> batches;
    meta >> key >> optimizer_step;
    meta >> key >> conv_count;
    meta >> key >> fc_count;
    meta >> key >> m_size;
    meta >> key >> v_size;
    meta.close();

    torch::NoGradGuard no_grad;
    bool ok = true;
    size_t tensor_index = 0;
    impl_->for_each_conv([&](const std::string& name, torch::nn::Conv2d& conv,
                             torch::nn::BatchNorm2d& bn) {
        if (!ok) return;
        const std::string prefix =
            "training_conv" + std::to_string(tensor_index++) + "_";
        auto load_into = [&](torch::Tensor& target,
                             const std::string& suffix) {
            std::vector<float> data;
            if (!read_bin(directory + "/" + prefix + suffix, data,
                          static_cast<size_t>(target.numel())))
                return false;
            target.copy_(torch::from_blob(data.data(), target.sizes(),
                                          torch::TensorOptions().dtype(
                                              torch::kFloat32))
                             .to(impl_->device));
            return true;
        };
        ok = load_into(conv->weight, "weight.bin") &&
             load_into(conv->bias, "bias.bin") &&
             load_into(bn->weight, "bn_gamma.bin") &&
             load_into(bn->bias, "bn_beta.bin") &&
             load_into(bn->running_mean, "bn_mean.bin") &&
             load_into(bn->running_var, "bn_var.bin");
    });
    size_t fc_index = 0;
    impl_->for_each_fc([&](const std::string& name, torch::nn::Linear& fc) {
        if (!ok) return;
        const std::string prefix =
            "training_fc" + std::to_string(fc_index++) + "_";
        auto load_into = [&](torch::Tensor& target,
                             const std::string& suffix) {
            std::vector<float> data;
            if (!read_bin(directory + "/" + prefix + suffix, data,
                          static_cast<size_t>(target.numel())))
                return false;
            target.copy_(torch::from_blob(data.data(), target.sizes(),
                                          torch::TensorOptions().dtype(
                                              torch::kFloat32))
                             .to(impl_->device));
            return true;
        };
        ok = load_into(fc->weight, "weight.bin") &&
             load_into(fc->bias, "bias.bin");
    });
    if (!ok) return false;

    if (batch_index != nullptr) *batch_index = batches;
    if (step != nullptr) *step = optimizer_step;
    if (moments_m != nullptr && m_size > 0) {
        std::vector<float> data;
        if (read_bin(directory + "/training_moments_m.bin", data, m_size))
            *moments_m = std::move(data);
    }
    if (moments_v != nullptr && v_size > 0) {
        std::vector<float> data;
        if (read_bin(directory + "/training_moments_v.bin", data, v_size))
            *moments_v = std::move(data);
    }
    return true;
}

PureNet::TrainStats CudaNetworkBackend::train_step(
    const std::vector<float>& states, int batch,
    const std::vector<float>& probs, const std::vector<float>& winners,
    double learning_rate, double value_loss_weight) {
    impl_->net->train();
    for (auto& group : impl_->optimizer->param_groups())
        static_cast<torch::optim::AdamOptions&>(group.options())
            .lr(learning_rate);
    impl_->optimizer->zero_grad();

    auto state_tensor =
        torch::from_blob(const_cast<float*>(states.data()),
                         {batch, NetConfig::kInputChannels, 15, 15},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(impl_->device);
    auto policy_target =
        torch::from_blob(const_cast<float*>(probs.data()), {batch, kCells},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(impl_->device);
    auto value_target =
        torch::from_blob(const_cast<float*>(winners.data()), {batch},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(impl_->device);

    auto [log_policy, values] = impl_->net->forward(state_tensor);
    const auto value_loss =
        torch::mse_loss(values.reshape({batch}), value_target);
    const auto policy_loss =
        -torch::mean(torch::sum(policy_target * log_policy, 1));
    const auto loss = value_loss * value_loss_weight + policy_loss;
    loss.backward();
    const double clip = impl_->owner.grad_clip();
    if (clip > 0.0)
        torch::nn::utils::clip_grad_norm_(impl_->net->parameters(), clip);
    impl_->optimizer->step();
    impl_->trained = true;
    const auto entropy = -torch::mean(
        torch::sum(torch::exp(log_policy) * log_policy, 1));

    const torch::Tensor metrics =
        torch::stack({loss.detach(), entropy.detach(), policy_loss.detach(),
                      value_loss.detach()})
            .to(torch::kCPU)
            .contiguous();
    const float* result = metrics.data_ptr<float>();
    return {result[0], result[1], result[2], result[3]};
}

}  // namespace gomoku

#endif  // GOMOK_HAS_TORCH
