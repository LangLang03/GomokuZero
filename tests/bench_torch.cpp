// LibTorch CPU/CUDA forward benchmark.
// Usage: bench_torch <model.pt> <B> [iters]
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

struct GomokuNet : torch::nn::Module {
    GomokuNet() {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(4, 32, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        act_conv1 = register_module("act_conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 4, 1)));
        act_fc1 = register_module("act_fc1", torch::nn::Linear(4 * 15 * 15, 15 * 15));
        val_conv1 = register_module("val_conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 2, 1)));
        val_fc1 = register_module("val_fc1", torch::nn::Linear(2 * 15 * 15, 64));
        val_fc2 = register_module("val_fc2", torch::nn::Linear(64, 1));
    }
    std::vector<torch::Tensor> forward(torch::Tensor state) {
        torch::Tensor trunk = torch::relu(conv1->forward(state));
        trunk = torch::relu(conv2->forward(trunk));
        trunk = torch::relu(conv3->forward(trunk));
        torch::Tensor policy = torch::relu(act_conv1->forward(trunk)).flatten(1);
        torch::Tensor log_policy = torch::log_softmax(act_fc1->forward(policy), 1);
        torch::Tensor value = torch::relu(val_conv1->forward(trunk)).flatten(1);
        value = torch::relu(val_fc1->forward(value));
        return {log_policy, torch::tanh(val_fc2->forward(value))};
    }
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr},
        act_conv1{nullptr}, val_conv1{nullptr};
    torch::nn::Linear act_fc1{nullptr}, val_fc1{nullptr}, val_fc2{nullptr};
};

static double bench(const char* name, GomokuNet& net, torch::Tensor x,
                    int iters, bool include_copy) {
    auto run = [&](torch::Tensor in) {
        auto out = net.forward(in);
        if (include_copy) out[0].to(torch::kCPU);
        return out[0];
    };
    for (int i = 0; i < 5; ++i) {
        torch::Tensor w_in = include_copy ? x.to(torch::kCUDA) : x;
        run(w_in);
    }
    torch::cuda::synchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        torch::Tensor in = include_copy ? x.to(torch::kCUDA) : x;
        run(in);
    }
    torch::cuda::synchronize();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("%-30s: %8.2f ms/iter -> %8.1f us/state\n",
                name, ms, ms * 1000.0 / x.size(0));
    std::fflush(stdout);
    return ms;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: bench_torch <model.pt> <B>\n"); return 2; }
    const int B = std::atoi(argv[2]);
    torch::jit::Module w = torch::jit::load(argv[1]);
    w.to(torch::kCPU);
    GomokuNet net;
    // Match parameters by their full module name.
    std::map<std::string, torch::Tensor> src;
    for (auto it : w.named_parameters()) src[it.name] = it.value;
    for (auto& it : net.named_parameters()) {
        auto f = src.find(it.key());
        if (f != src.end()) it.value().set_data(f->second);
    }

    net.eval();
    torch::NoGradGuard guard;
    torch::Tensor x = torch::zeros({B, 4, 15, 15});
    int iters = B == 1 ? 200 : 20;
    bench("libtorch CPU", net, x, iters, false);
    net.to(torch::kCUDA);
    torch::Tensor xc = x.to(torch::kCUDA);
    bench("libtorch CUDA (data on GPU)", net, xc, iters, false);
    bench("libtorch CUDA + H2D copy", net, x, iters, true);
    return 0;
}
