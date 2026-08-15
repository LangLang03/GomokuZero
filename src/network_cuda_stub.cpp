#include "network_cuda.h"
#include <stdexcept>

namespace gomoku {

struct CudaNetworkBackend::Impl {};

CudaNetworkBackend::CudaNetworkBackend(PureNet&, double)
    : impl_(std::make_unique<Impl>()) {}
CudaNetworkBackend::~CudaNetworkBackend() = default;
bool CudaNetworkBackend::available() const { return false; }
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
