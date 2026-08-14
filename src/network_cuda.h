#pragma once
#include <memory>
#include <string>
#include <vector>
#include "network.h"

namespace gomoku {

// Torch types are hidden behind Impl so the rest of the project does not need
// to parse the large LibTorch headers.
class CudaNetworkBackend {
public:
    CudaNetworkBackend(PureNet& owner, double learning_rate);
    ~CudaNetworkBackend();

    bool available() const;
    void sync_from_cpu();
    void sync_to_cpu();
    bool save(const std::string& directory) const;

    void forward(const float* states, int batch, float* policy, float* value,
                 bool probabilities) const;
    PureNet::TrainStats train_step(const std::vector<float>& states, int batch,
                                   const std::vector<float>& probs,
                                   const std::vector<float>& winners,
                                   double learning_rate);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gomoku
