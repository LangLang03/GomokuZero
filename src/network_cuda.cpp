#include "network_cuda.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <torch/torch.h>

namespace gomoku {
namespace {

class CudaNetImpl : public torch::nn::Module {
public:
    CudaNetImpl()
        : conv1(torch::nn::Conv2dOptions(4, 32, 3).padding(1)),
          conv2(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
          conv3(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
          act_conv1(torch::nn::Conv2dOptions(128, 4, 1)),
          act_fc1(4 * 225, 225),
          val_conv1(torch::nn::Conv2dOptions(128, 2, 1)),
          val_fc1(2 * 225, 64),
          val_fc2(64, 1) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("act_conv1", act_conv1);
        register_module("act_fc1", act_fc1);
        register_module("val_conv1", val_conv1);
        register_module("val_fc1", val_fc1);
        register_module("val_fc2", val_fc2);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        auto trunk = torch::relu(conv1->forward(input));
        trunk = torch::relu(conv2->forward(trunk));
        trunk = torch::relu(conv3->forward(trunk));

        auto policy = torch::relu(act_conv1->forward(trunk));
        policy = policy.reshape({-1, 4 * 225});
        policy = torch::log_softmax(act_fc1->forward(policy), 1);

        auto value = torch::relu(val_conv1->forward(trunk));
        value = value.reshape({-1, 2 * 225});
        value = torch::relu(val_fc1->forward(value));
        value = torch::tanh(val_fc2->forward(value));
        return {policy, value};
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Conv2d act_conv1{nullptr};
    torch::nn::Linear act_fc1{nullptr};
    torch::nn::Conv2d val_conv1{nullptr};
    torch::nn::Linear val_fc1{nullptr}, val_fc2{nullptr};
};
TORCH_MODULE(CudaNet);

bool save_tensor(const std::string& path, const torch::Tensor& tensor) {
    const torch::Tensor cpu = tensor.detach().to(torch::kCPU).contiguous();
    if (cpu.numel() > std::numeric_limits<uint32_t>::max()) return false;
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    const uint32_t count = static_cast<uint32_t>(cpu.numel());
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
               static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(file);
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
                net->parameters(), torch::optim::AdamOptions(learning_rate));
        } catch (const c10::Error& error) {
            std::cerr << "[net] CUDA initialization failed: "
                      << error.what_without_backtrace() << '\n';
            available = false;
            optimizer.reset();
        }
    }

    void copy_to_parameter(torch::Tensor parameter,
                           const std::vector<float>& source) {
        torch::NoGradGuard no_grad;
        auto tensor = torch::from_blob(
            const_cast<float*>(source.data()), parameter.sizes(),
            torch::TensorOptions().dtype(torch::kFloat32));
        parameter.copy_(tensor.to(device));
    }

    void copy_from_parameter(const torch::Tensor& parameter,
                             std::vector<float>& destination) {
        const torch::Tensor cpu = parameter.detach().to(torch::kCPU).contiguous();
        destination.resize(static_cast<size_t>(cpu.numel()));
        std::memcpy(destination.data(), cpu.data_ptr<float>(),
                    destination.size() * sizeof(float));
    }

    PureNet& owner;
    CudaNet net;
    torch::Device device{torch::kCPU};
    std::unique_ptr<torch::optim::Adam> optimizer;
    bool available = false;
};

CudaNetworkBackend::CudaNetworkBackend(PureNet& owner, double learning_rate)
    : impl_(std::make_unique<Impl>(owner, learning_rate)) {}

CudaNetworkBackend::~CudaNetworkBackend() = default;

bool CudaNetworkBackend::available() const { return impl_->available; }

void CudaNetworkBackend::sync_from_cpu() {
    if (!available()) return;
    PureNet& owner = impl_->owner;
    impl_->copy_to_parameter(impl_->net->conv1->weight, owner.c1_.w);
    impl_->copy_to_parameter(impl_->net->conv1->bias, owner.c1_.b);
    impl_->copy_to_parameter(impl_->net->conv2->weight, owner.c2_.w);
    impl_->copy_to_parameter(impl_->net->conv2->bias, owner.c2_.b);
    impl_->copy_to_parameter(impl_->net->conv3->weight, owner.c3_.w);
    impl_->copy_to_parameter(impl_->net->conv3->bias, owner.c3_.b);
    impl_->copy_to_parameter(impl_->net->act_conv1->weight, owner.act_c_.w);
    impl_->copy_to_parameter(impl_->net->act_conv1->bias, owner.act_c_.b);
    impl_->copy_to_parameter(impl_->net->act_fc1->weight, owner.act_fc_.w);
    impl_->copy_to_parameter(impl_->net->act_fc1->bias, owner.act_fc_.b);
    impl_->copy_to_parameter(impl_->net->val_conv1->weight, owner.val_c_.w);
    impl_->copy_to_parameter(impl_->net->val_conv1->bias, owner.val_c_.b);
    impl_->copy_to_parameter(impl_->net->val_fc1->weight, owner.val_fc1_.w);
    impl_->copy_to_parameter(impl_->net->val_fc1->bias, owner.val_fc1_.b);
    impl_->copy_to_parameter(impl_->net->val_fc2->weight, owner.val_fc2_.w);
    impl_->copy_to_parameter(impl_->net->val_fc2->bias, owner.val_fc2_.b);
}

void CudaNetworkBackend::sync_to_cpu() {
    if (!available()) return;
    PureNet& owner = impl_->owner;
    impl_->copy_from_parameter(impl_->net->conv1->weight, owner.c1_.w);
    impl_->copy_from_parameter(impl_->net->conv1->bias, owner.c1_.b);
    impl_->copy_from_parameter(impl_->net->conv2->weight, owner.c2_.w);
    impl_->copy_from_parameter(impl_->net->conv2->bias, owner.c2_.b);
    impl_->copy_from_parameter(impl_->net->conv3->weight, owner.c3_.w);
    impl_->copy_from_parameter(impl_->net->conv3->bias, owner.c3_.b);
    impl_->copy_from_parameter(impl_->net->act_conv1->weight, owner.act_c_.w);
    impl_->copy_from_parameter(impl_->net->act_conv1->bias, owner.act_c_.b);
    impl_->copy_from_parameter(impl_->net->act_fc1->weight, owner.act_fc_.w);
    impl_->copy_from_parameter(impl_->net->act_fc1->bias, owner.act_fc_.b);
    impl_->copy_from_parameter(impl_->net->val_conv1->weight, owner.val_c_.w);
    impl_->copy_from_parameter(impl_->net->val_conv1->bias, owner.val_c_.b);
    impl_->copy_from_parameter(impl_->net->val_fc1->weight, owner.val_fc1_.w);
    impl_->copy_from_parameter(impl_->net->val_fc1->bias, owner.val_fc1_.b);
    impl_->copy_from_parameter(impl_->net->val_fc2->weight, owner.val_fc2_.w);
    impl_->copy_from_parameter(impl_->net->val_fc2->bias, owner.val_fc2_.b);
    owner.rebuild_quantized_weights();
}

bool CudaNetworkBackend::save(const std::string& directory) const {
    if (!available()) return false;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) return false;
    const auto save_layer = [&](const char* name,
                                const torch::Tensor& weight,
                                const torch::Tensor& bias) {
        return save_tensor(directory + "/" + name + "_weight.bin", weight) &&
               save_tensor(directory + "/" + name + "_bias.bin", bias);
    };
    return save_layer("conv1", impl_->net->conv1->weight,
                      impl_->net->conv1->bias) &&
           save_layer("conv2", impl_->net->conv2->weight,
                      impl_->net->conv2->bias) &&
           save_layer("conv3", impl_->net->conv3->weight,
                      impl_->net->conv3->bias) &&
           save_layer("act_conv1", impl_->net->act_conv1->weight,
                      impl_->net->act_conv1->bias) &&
           save_layer("act_fc1", impl_->net->act_fc1->weight,
                      impl_->net->act_fc1->bias) &&
           save_layer("val_conv1", impl_->net->val_conv1->weight,
                      impl_->net->val_conv1->bias) &&
           save_layer("val_fc1", impl_->net->val_fc1->weight,
                      impl_->net->val_fc1->bias) &&
           save_layer("val_fc2", impl_->net->val_fc2->weight,
                      impl_->net->val_fc2->bias);
}

void CudaNetworkBackend::forward(const float* states, int batch,
                                 float* policy, float* value,
                                 bool probabilities) const {
    torch::InferenceMode inference_mode;
    impl_->net->eval();
    auto cpu_states = torch::from_blob(
        const_cast<float*>(states), {batch, 4, 15, 15},
        torch::TensorOptions().dtype(torch::kFloat32));
    auto [log_policy, values] = impl_->net->forward(
        cpu_states.to(impl_->device));
    torch::Tensor output_policy = probabilities ? log_policy.exp() : log_policy;
    output_policy = output_policy.to(torch::kCPU).contiguous();
    values = values.reshape({batch}).to(torch::kCPU).contiguous();
    std::memcpy(policy, output_policy.data_ptr<float>(),
                static_cast<size_t>(batch) * 225 * sizeof(float));
    std::memcpy(value, values.data_ptr<float>(),
                static_cast<size_t>(batch) * sizeof(float));
}

PureNet::TrainStats CudaNetworkBackend::train_step(
    const std::vector<float>& states, int batch,
    const std::vector<float>& probs, const std::vector<float>& winners,
    double learning_rate) {
    impl_->net->train();
    for (auto& group : impl_->optimizer->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options())
            .lr(learning_rate);
    }
    impl_->optimizer->zero_grad();

    auto state_tensor = torch::from_blob(
        const_cast<float*>(states.data()), {batch, 4, 15, 15},
        torch::TensorOptions().dtype(torch::kFloat32)).to(impl_->device);
    auto policy_target = torch::from_blob(
        const_cast<float*>(probs.data()), {batch, 225},
        torch::TensorOptions().dtype(torch::kFloat32)).to(impl_->device);
    auto value_target = torch::from_blob(
        const_cast<float*>(winners.data()), {batch},
        torch::TensorOptions().dtype(torch::kFloat32)).to(impl_->device);

    auto [log_policy, values] = impl_->net->forward(state_tensor);
    const auto value_loss = torch::mse_loss(values.reshape({batch}),
                                            value_target);
    const auto policy_loss = -torch::mean(torch::sum(
        policy_target * log_policy, 1));
    const auto loss = value_loss + policy_loss;
    loss.backward();
    torch::nn::utils::clip_grad_norm_(impl_->net->parameters(), 10.0);
    impl_->optimizer->step();
    const auto entropy = -torch::mean(torch::sum(
        torch::exp(log_policy) * log_policy, 1));

    const torch::Tensor metrics = torch::stack(
        {loss.detach(), entropy.detach(), policy_loss.detach(),
         value_loss.detach()}).to(torch::kCPU).contiguous();
    const float* result = metrics.data_ptr<float>();
    return {result[0], result[1], result[2], result[3]};
}

}  // namespace gomoku
