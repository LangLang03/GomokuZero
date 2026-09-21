#pragma once
#include <memory>
#include <string>
#include <vector>
#include "network.h"

namespace gomoku {

// LibTorch stays out of public headers through this pimpl.
class CudaNetworkBackend {
public:
    CudaNetworkBackend(PureNet& owner, double learning_rate);
    ~CudaNetworkBackend();

    bool available() const;
    bool load(const std::string& directory);
    // The stored tensors have batch norm folded into the convolution, which
    // is exactly the form the native forward pass consumes.
    void sync_from_cpu();
    void sync_to_cpu();
    bool save(const std::string& directory) const;

    // Mirrors PureNet's optimiser settings so the CUDA and native training
    // paths optimise the same objective.
    void set_weight_decay(double weight_decay);

    void forward(const float* states, int batch, float* policy, float* value,
                 bool probabilities) const;
    PureNet::TrainStats train_step(const std::vector<float>& states, int batch,
                                   const std::vector<float>& probs,
                                   const std::vector<float>& winners,
                                   double learning_rate,
                                   double value_loss_weight = 1.0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gomoku
