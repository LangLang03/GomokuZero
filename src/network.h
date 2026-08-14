#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "gradient.h"
#include "optimizer.h"

namespace gomoku {

class CudaNetworkBackend;

// Policy/value network with CUDA and native CPU backends.
// Layout:
//   conv3x3(4->32) relu -> conv3x3(32->64) relu -> conv3x3(64->128) relu
//   policy: conv1x1(128->4) relu -> fc(4*225 -> 225) -> log_softmax
//   value:  conv1x1(128->2) relu -> fc(2*225 -> 64) relu -> fc(64->1) -> tanh
// Raw weights use the format written by tools/export_model.py.
class PureNet {
public:
    PureNet(double lr = 2e-3);
    ~PureNet();
    PureNet(const PureNet&) = delete;
    PureNet& operator=(const PureNet&) = delete;

    // Reads the 16 tensors under dir.
    bool load(const std::string& dir);
    bool save(const std::string& dir) const;

    struct TrainStats { float loss, entropy, policy_loss, value_loss; };

    // states [B][4][15][15], probs [B][225], winners [B]
    TrainStats train_step(const std::vector<float>& states, int B,
                          const std::vector<float>& probs,
                          const std::vector<float>& winners,
                          double lr);

    // [4][15][15] -> log_policy[225], value
    void forward_one(const std::vector<float>& state,
                     std::vector<float>& log_policy, float& value) const;

    // states [B][4][15][15] -> log_policy [B][225], value [B]
    void forward_batch(const float* states, int B,
                       float* log_policy, float* value) const;

    // Probability output used by MCTS.
    void forward_batch_policy(const float* states, int B,
                              float* policy, float* value) const;

    // Only one thread may submit a batch at a time.
    void set_inference_threads(int threads);
    int inference_threads() const { return inference_threads_; }
    bool using_cuda() const { return cuda_enabled_; }
    bool cuda_available() const;
    void set_cuda_enabled(bool enabled);
    void set_fast_inference(bool enabled) { fast_inference_enabled_ = enabled; }
    void set_quantized_inference(bool enabled) {
        quantized_inference_enabled_ = enabled;
    }

    // Idempotent.
    void register_all_params();

private:
    ConvLayer c1_{4, 32, 3, 3}, c2_{32, 64, 3, 3}, c3_{64, 128, 3, 3};
    ConvLayer act_c_{128, 4, 1, 1};
    FCLayer act_fc_{4 * 225, 225};
    ConvLayer val_c_{128, 2, 1, 1};
    FCLayer val_fc1_{2 * 225, 64}, val_fc2_{64, 1};

    Adam adam_;
    bool loaded_ = false;
    bool adam_registered_ = false;
    int inference_threads_ = 1;
    bool fast_inference_enabled_ = true;
    bool quantized_inference_enabled_ = false;
    std::vector<int8_t> quantized_c2_, quantized_c3_;
    std::vector<float> quantized_c2_scales_;
    std::vector<float> quantized_c3_scales_;
    class InferencePool;
    std::unique_ptr<InferencePool> inference_pool_;
    std::unique_ptr<CudaNetworkBackend> cuda_backend_;
    bool cuda_enabled_ = false;

    friend class CudaNetworkBackend;

    void init_params();
    void rebuild_quantized_weights();
    void forward_impl(const float* state, float* policy,
                      float& value, bool probabilities) const;
    void forward_batch_impl(const float* states, int B, float* policy,
                            float* value, bool probabilities) const;
};

}  // namespace gomoku
