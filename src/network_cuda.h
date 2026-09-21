#pragma once
#include <cstdint>
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

    // Full training-state access, so a checkpoint can carry the raw batch-norm
    // parameterisation and the optimiser moments rather than only the folded
    // inference weights.
    bool save_training_state(const std::string& directory, int batch_index,
                             int64_t step, const std::vector<float>& moments_m,
                             const std::vector<float>& moments_v) const;
    bool load_training_state(const std::string& directory, int* batch_index,
                             int64_t* step, std::vector<float>* moments_m,
                             std::vector<float>* moments_v);
    bool has_training_state(const std::string& directory) const;

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
