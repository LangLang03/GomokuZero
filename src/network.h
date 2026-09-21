#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "gradient.h"
#include "net_config.h"
#include "optimizer.h"

namespace gomoku {

class CudaNetworkBackend;

// Policy/value network.
//
//   state [4,15,15]
//     stem      conv3x3 4 -> C, bn, relu
//     trunk     kResBlocks x [ conv3x3 C->C, bn, relu, conv3x3 C->C, bn, add, relu ]
//     policy    conv1x1 C->2, bn, relu, reshape(2*225), fc 450->225, log_softmax
//     value     conv1x1 C->1, bn, relu, reshape(225), fc 225->64, relu,
//               fc 64->1, tanh
//
// Training runs on the LibTorch backend. The native path stays available for
// inference so MCTS keeps working without CUDA (and so the two forward
// implementations can be checked against each other).
class PureNet {
public:
    explicit PureNet(double lr = 2e-3);
    ~PureNet();
    PureNet(const PureNet&) = delete;
    PureNet& operator=(const PureNet&) = delete;

    // Reads the tensor set under dir, or returns false if it is missing or
    // belongs to a different architecture.
    bool load(const std::string& dir);

    // Inference export: batch-normalisation folded into the convolutions.
    // This is what MCTS, `human` and deployment read.
    bool export_inference(const std::string& dir) const;

    // Training checkpoint: raw convolution weights, batch-norm gamma/beta and
    // running statistics, dense layers, optimiser moments and the batch index.
    // Folding is exact for inference but destroys the parameterisation, so a
    // folded-only file cannot resume training faithfully.
    bool save_training_state(const std::string& dir, int batch_index) const;
    bool load_training_state(const std::string& dir, int* batch_index_out);
    bool has_training_state(const std::string& dir) const;

    struct TrainStats { float loss, entropy, policy_loss, value_loss; };

    // states [B][4][15][15], probs [B][225], winners [B].
    // Requires the CUDA backend; throws std::logic_error otherwise.
    TrainStats train_step(const std::vector<float>& states, int B,
                          const std::vector<float>& probs,
                          const std::vector<float>& winners,
                          double lr,
                          double value_loss_weight = 1.0);

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

    // Last batch index from a training checkpoint, or 0.
    int resume_batch_index() const { return resume_batch_index_; }

    // Optimiser settings, mirrored into the LibTorch optimiser so the
    // objective does not depend on the backend.
    void set_weight_decay(double weight_decay);
    double weight_decay() const;
    void set_grad_clip(double max_norm);
    double grad_clip() const { return grad_clip_; }

    bool using_cuda() const { return cuda_enabled_; }
    bool cuda_available() const;
    void set_cuda_enabled(bool enabled);

    // Parameter count, for logging and tests.
    size_t parameter_count() const;

    static constexpr int kChannels = NetConfig::kChannels;
    static constexpr int kResBlocks = NetConfig::kResBlocks;
    static constexpr int kCells = NetConfig::kCells;

private:
    friend class CudaNetworkBackend;

    // A convolution followed by batch normalisation.
    struct ConvBlock {
        ConvLayer conv;
        std::vector<float> bn_gamma, bn_beta, bn_mean, bn_var;
        std::vector<float> bn_gamma_grad, bn_beta_grad;
        float bn_eps = 1e-5f;

        ConvBlock(int in_ch, int out_ch, int kh, int kw)
            : conv(in_ch, out_ch, kh, kw) {}
    };

    // Two 3x3 convolutions with a skip connection.
    struct ResBlock {
        ConvBlock a, b;
        explicit ResBlock(int channels)
            : a(channels, channels, 3, 3), b(channels, channels, 3, 3) {}
    };

    // Batch-normalisation folded into its convolution, so native inference is
    // a plain conv + bias + ReLU per layer.
    struct FoldedConv {
        std::vector<float> w;  // [out][in][kh][kw]
        std::vector<float> b;  // [out]
    };

    std::vector<std::unique_ptr<ResBlock>> trunk_;
    ConvBlock stem_, policy_conv_, value_conv_;
    FCLayer policy_fc_, value_fc1_, value_fc2_;

    // Folded weights for the native CPU path.
    FoldedConv stem_folded_;
    std::vector<FoldedConv> trunk_a_folded_, trunk_b_folded_;
    FoldedConv policy_folded_, value_folded_;

    Adam adam_;
    bool loaded_ = false;
    // Remembered so switching to the native CPU path can re-read the same
    // checkpoint instead of trying to mirror live CUDA tensors.
    std::string last_load_dir_;
    int resume_batch_index_ = 0;
    int inference_threads_ = 1;
    double grad_clip_ = 2.0;
    std::unique_ptr<CudaNetworkBackend> cuda_backend_;
    bool cuda_enabled_ = false;

    void init_params();
    void rebuild_folded_weights();
    // Reads the folded tensors into the native buffers only. Used by load()
    // and by the CUDA backend when it mirrors parameters back for --cpu.
    bool load_native(const std::string& dir);
    void forward_impl(const float* state, float* policy,
                      float& value, bool probabilities) const;
    void forward_batch_impl(const float* states, int B, float* policy,
                            float* value, bool probabilities) const;
};

}  // namespace gomoku
