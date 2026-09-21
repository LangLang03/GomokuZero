#include "network_cuda.h"
#include <cstdint>
#include <stdexcept>

namespace gomoku {

struct CudaNetworkBackend::Impl {};

CudaNetworkBackend::CudaNetworkBackend(PureNet&, double)
    : impl_(std::make_unique<Impl>()) {}
CudaNetworkBackend::~CudaNetworkBackend() = default;
bool CudaNetworkBackend::available() const { return false; }
bool CudaNetworkBackend::load(const std::string&) { return false; }
bool CudaNetworkBackend::has_training_state(const std::string&) const {
    return false;
}
bool CudaNetworkBackend::save_training_state(
    const std::string&, int, int64_t, const std::vector<float>&,
    const std::vector<float>&) const {
    return false;
}
bool CudaNetworkBackend::load_training_state(const std::string&, int*,
                                             int64_t*, std::vector<float>*,
                                             std::vector<float>*) {
    return false;
}
void CudaNetworkBackend::set_weight_decay(double) {}
void CudaNetworkBackend::sync_from_cpu() {}
void CudaNetworkBackend::sync_to_cpu() {}
bool CudaNetworkBackend::save(const std::string&) const { return false; }
void CudaNetworkBackend::forward(const float*, int, float*, float*, bool) const {
    throw std::logic_error("CUDA backend is unavailable");
}
PureNet::TrainStats CudaNetworkBackend::train_step(
    const std::vector<float>&, int, const std::vector<float>&,
    const std::vector<float>&, double, double) {
    throw std::logic_error("CUDA backend is unavailable");
}

}  // namespace gomoku
