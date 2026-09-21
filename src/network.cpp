#include "network.h"
#include "network_cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>

namespace gomoku {

namespace {

constexpr int SIDE = NetConfig::kBoardSize;
constexpr int CELLS = NetConfig::kCells;

bool write_bin_file(const std::string& path, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    const uint32_t count = static_cast<uint32_t>(data.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(f);
}

bool read_bin_file(const std::string& path, std::vector<float>& out,
                   size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f || count != expected) return false;
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(f);
}

// out = relu(conv3x3(in) + bias), zero padded, same size.
void folded_conv3x3(const float* in, int in_ch, const std::vector<float>& w,
                    const std::vector<float>& b, int out_ch,
                    std::vector<float>& out, bool relu) {
    out.assign(static_cast<size_t>(out_ch) * CELLS, 0.0f);
    for (int oc = 0; oc < out_ch; ++oc) {
        float* dst = out.data() + static_cast<size_t>(oc) * CELLS;
        std::fill(dst, dst + CELLS, b[static_cast<size_t>(oc)]);
        const float* kernel_base = w.data() + static_cast<size_t>(oc) * in_ch * 9;
        for (int ic = 0; ic < in_ch; ++ic) {
            const float* src = in + static_cast<size_t>(ic) * CELLS;
            const float* kernel = kernel_base + static_cast<size_t>(ic) * 9;
            for (int kr = 0; kr < 3; ++kr) {
                const int row_begin = std::max(0, 1 - kr);
                const int row_end = std::min(SIDE, SIDE + 1 - kr);
                for (int kc = 0; kc < 3; ++kc) {
                    const float weight = kernel[kr * 3 + kc];
                    const int col_begin = std::max(0, 1 - kc);
                    const int col_end = std::min(SIDE, SIDE + 1 - kc);
                    for (int row = row_begin; row < row_end; ++row) {
                        float* d = dst + row * SIDE + col_begin;
                        const float* s =
                            src + (row + kr - 1) * SIDE + col_begin + kc - 1;
                        for (int col = 0; col + 8 <= col_end - col_begin; col += 8) {
                            for (int lane = 0; lane < 8; ++lane)
                                d[col + lane] += s[col + lane] * weight;
                        }
                        for (int col = (col_end - col_begin) & ~7;
                             col < col_end - col_begin; ++col)
                            d[col] += s[col] * weight;
                    }
                }
            }
        }
        if (relu)
            for (int i = 0; i < CELLS; ++i) dst[i] = std::max(0.0f, dst[i]);
    }
}

// out = relu(conv1x1(in) + bias).
void folded_conv1x1(const float* in, int in_ch, const std::vector<float>& w,
                    const std::vector<float>& b, int out_ch,
                    std::vector<float>& out, bool relu) {
    out.assign(static_cast<size_t>(out_ch) * CELLS, 0.0f);
    for (int oc = 0; oc < out_ch; ++oc) {
        float* dst = out.data() + static_cast<size_t>(oc) * CELLS;
        std::fill(dst, dst + CELLS, b[static_cast<size_t>(oc)]);
        const float* kernel = w.data() + static_cast<size_t>(oc) * in_ch;
        for (int ic = 0; ic < in_ch; ++ic) {
            const float* src = in + static_cast<size_t>(ic) * CELLS;
            const float weight = kernel[ic];
            for (int i = 0; i < CELLS; ++i) dst[i] += src[i] * weight;
        }
        if (relu)
            for (int i = 0; i < CELLS; ++i) dst[i] = std::max(0.0f, dst[i]);
    }
}

void dense_forward(const FCLayer& layer, const float* in, float* out,
                   bool relu) {
    for (int o = 0; o < layer.out_n; ++o) {
        float acc = layer.b[static_cast<size_t>(o)];
        const float* row = layer.w.data() + static_cast<size_t>(o) * layer.in_n;
        for (int i = 0; i < layer.in_n; ++i) acc += in[i] * row[i];
        out[o] = relu ? std::max(0.0f, acc) : acc;
    }
}

// Temporary instrument: prints a per-layer summary so a divergence between
// the native and LibTorch forward passes can be localised. Enabled at
// runtime by GOMOK_DUMP_FORWARD, compiled out otherwise.
void dump_summary(const char* tag, const std::vector<float>& values) {
    double sum = 0.0;
    float maximum = -1e30f, minimum = 1e30f;
    for (float x : values) {
        sum += x;
        maximum = std::max(maximum, x);
        minimum = std::min(minimum, x);
    }
    std::fprintf(stderr,
                 "[dump] %-12s n=%zu sum=%.6f min=%.6f max=%.6f first=%.6f\n",
                 tag, values.size(), sum, minimum, maximum,
                 values.empty() ? 0.0f : values[0]);
}

void log_softmax(const float* logits, float* out, int n, bool probabilities) {
    const float maximum = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += std::exp(logits[i] - maximum);
    if (probabilities) {
        const float inverse = 1.0f / sum;
        for (int i = 0; i < n; ++i)
            out[i] = std::exp(logits[i] - maximum) * inverse;
    } else {
        const float log_sum = maximum + std::log(sum);
        for (int i = 0; i < n; ++i) out[i] = logits[i] - log_sum;
    }
}

}  // namespace

PureNet::PureNet(double lr)
    : stem_(NetConfig::kInputChannels, NetConfig::kChannels, 3, 3),
      policy_conv_(NetConfig::kChannels, 2, 1, 1),
      value_conv_(NetConfig::kChannels, 1, 1, 1),
      policy_fc_(NetConfig::kPolicyFcIn, NetConfig::kPolicyOut),
      value_fc1_(NetConfig::kValueFcIn, NetConfig::kValueHidden),
      value_fc2_(NetConfig::kValueHidden, 1),
      adam_(lr) {
    init_params();
    cuda_backend_ = std::make_unique<CudaNetworkBackend>(*this, lr);
    cuda_enabled_ = cuda_backend_->available();
    if (cuda_enabled_) {
        cuda_backend_->sync_from_cpu();
        std::fprintf(stderr, "[net] CUDA backend enabled (%zu parameters)\n",
                     parameter_count());
    } else {
        std::fprintf(stderr,
                     "[net] CUDA unavailable; native CPU inference only "
                     "(training needs the LibTorch backend)\n");
    }
}

PureNet::~PureNet() = default;

void PureNet::init_params() {
    trunk_.clear();
    trunk_.reserve(NetConfig::kResBlocks);
    for (int i = 0; i < NetConfig::kResBlocks; ++i)
        trunk_.push_back(std::make_unique<ResBlock>(NetConfig::kChannels));

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    // He-style initialisation for the rectified convolutions, scaled down for
    // the residual branches so the trunk starts near the identity.
    // ConvLayer/FCLayer only carry their shape; the storage has to be
    // allocated here before anything can be written into it.
    auto init_conv = [&rng](ConvLayer& layer, float gain) {
        const size_t weights = static_cast<size_t>(layer.out_ch) * layer.in_ch *
                               layer.kh * layer.kw;
        layer.w.assign(weights, 0.0f);
        layer.b.assign(static_cast<size_t>(layer.out_ch), 0.0f);
        const float fan_in = static_cast<float>(layer.in_ch * layer.kh * layer.kw);
        const float bound = gain * std::sqrt(6.0f / std::max(1.0f, fan_in));
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (float& v : layer.w) v = dist(rng);
        layer.gw.assign(weights, 0.0f);
        layer.gb.assign(static_cast<size_t>(layer.out_ch), 0.0f);
    };
    auto init_fc = [&rng](FCLayer& layer) {
        const size_t weights =
            static_cast<size_t>(layer.out_n) * static_cast<size_t>(layer.in_n);
        layer.w.assign(weights, 0.0f);
        layer.b.assign(static_cast<size_t>(layer.out_n), 0.0f);
        const float bound = std::sqrt(6.0f / static_cast<float>(layer.in_n));
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (float& v : layer.w) v = dist(rng);
        layer.gw.assign(weights, 0.0f);
        layer.gb.assign(static_cast<size_t>(layer.out_n), 0.0f);
    };

    init_conv(stem_.conv, 1.0f);
    for (auto& block : trunk_) {
        init_conv(block->a.conv, 1.0f);
        init_conv(block->b.conv, 0.1f);  // residual branch starts small
    }
    init_conv(policy_conv_.conv, 1.0f);
    init_conv(value_conv_.conv, 1.0f);
    init_fc(policy_fc_);
    init_fc(value_fc1_);
    init_fc(value_fc2_);

    auto init_bn = [](ConvBlock& block) {
        block.bn_gamma.assign(static_cast<size_t>(block.conv.out_ch), 1.0f);
        block.bn_beta.assign(static_cast<size_t>(block.conv.out_ch), 0.0f);
        block.bn_mean.assign(static_cast<size_t>(block.conv.out_ch), 0.0f);
        block.bn_var.assign(static_cast<size_t>(block.conv.out_ch), 1.0f);
        block.bn_gamma_grad.assign(static_cast<size_t>(block.conv.out_ch), 0.0f);
        block.bn_beta_grad.assign(static_cast<size_t>(block.conv.out_ch), 0.0f);
    };
    init_bn(stem_);
    for (auto& block : trunk_) { init_bn(block->a); init_bn(block->b); }
    init_bn(policy_conv_);
    init_bn(value_conv_);

    loaded_ = false;
    rebuild_folded_weights();
}

void PureNet::rebuild_folded_weights() {
    // gamma / sqrt(var + eps) folds into the convolution weights, and
    // (beta - gamma * mean / sqrt(var + eps)) into its bias.
    auto fold = [](const ConvBlock& block, FoldedConv& out, bool relu) {
        (void)relu;
        const int out_ch = block.conv.out_ch;
        const int in_ch = block.conv.in_ch;
        const int kernel = block.conv.kh * block.conv.kw;
        out.w.assign(block.conv.w.size(), 0.0f);
        out.b.assign(static_cast<size_t>(out_ch), 0.0f);
        for (int oc = 0; oc < out_ch; ++oc) {
            const float inv_std =
                1.0f / std::sqrt(block.bn_var[static_cast<size_t>(oc)] +
                                 block.bn_eps);
            const float scale = block.bn_gamma[static_cast<size_t>(oc)] * inv_std;
            const float* src = block.conv.w.data() +
                               static_cast<size_t>(oc) * in_ch * kernel;
            float* dst = out.w.data() + static_cast<size_t>(oc) * in_ch * kernel;
            for (int i = 0; i < in_ch * kernel; ++i) dst[i] = src[i] * scale;
            out.b[static_cast<size_t>(oc)] =
                block.bn_beta[static_cast<size_t>(oc)] -
                block.bn_gamma[static_cast<size_t>(oc)] *
                    block.bn_mean[static_cast<size_t>(oc)] * inv_std +
                block.conv.b[static_cast<size_t>(oc)] * scale;
        }
    };
    fold(stem_, stem_folded_, true);
    trunk_a_folded_.resize(trunk_.size());
    trunk_b_folded_.resize(trunk_.size());
    for (size_t i = 0; i < trunk_.size(); ++i) {
        fold(trunk_[i]->a, trunk_a_folded_[i], true);
        fold(trunk_[i]->b, trunk_b_folded_[i], false);
    }
    fold(policy_conv_, policy_folded_, true);
    fold(value_conv_, value_folded_, true);
}

namespace {

// One thread-local scratch set for the native forward pass. Every buffer
// that dense_forward writes into must be sized up front.
struct ForwardScratch {
    std::vector<float> a, b, c;
    std::vector<float> logits = std::vector<float>(NetConfig::kPolicyOut);
    std::vector<float> hidden = std::vector<float>(NetConfig::kValueHidden);
    std::vector<float> policy_planes, value_planes;
};

}  // namespace

void PureNet::forward_impl(const float* state, float* policy, float& value,
                           bool probabilities) const {
    thread_local ForwardScratch scratch;
    static const bool dump_enabled = std::getenv("GOMOK_DUMP_FORWARD") != nullptr;
    const int channels = NetConfig::kChannels;

    folded_conv3x3(state, NetConfig::kInputChannels, stem_folded_.w,
                   stem_folded_.b, channels, scratch.a, true);
    if (dump_enabled) dump_summary("stem", scratch.a);



    scratch.b.resize(static_cast<size_t>(channels) * CELLS);
    for (size_t i = 0; i < trunk_.size(); ++i) {
        folded_conv3x3(scratch.a.data(), channels, trunk_a_folded_[i].w,
                       trunk_a_folded_[i].b, channels, scratch.b, true);
        folded_conv3x3(scratch.b.data(), channels, trunk_b_folded_[i].w,
                       trunk_b_folded_[i].b, channels, scratch.c, false);
        // residual add, then ReLU
        for (size_t k = 0; k < scratch.c.size(); ++k)
            scratch.a[k] = std::max(0.0f, scratch.c[k] + scratch.a[k]);

    }

    folded_conv1x1(scratch.a.data(), channels, policy_folded_.w,
                   policy_folded_.b, 2, scratch.policy_planes, true);
    if (dump_enabled) dump_summary("policy_conv", scratch.policy_planes);

    dense_forward(policy_fc_, scratch.policy_planes.data(), scratch.logits.data(),
                  false);
    log_softmax(scratch.logits.data(), policy, NetConfig::kPolicyOut,
                probabilities);

    folded_conv1x1(scratch.a.data(), channels, value_folded_.w,
                   value_folded_.b, 1, scratch.value_planes, true);
    if (dump_enabled) dump_summary("value_conv", scratch.value_planes);
    dense_forward(value_fc1_, scratch.value_planes.data(), scratch.hidden.data(),
                  true);
    float value_logit = 0.0f;
    dense_forward(value_fc2_, scratch.hidden.data(), &value_logit, false);
    value = std::tanh(value_logit);
}

void PureNet::forward_batch_impl(const float* states, int B, float* policy,
                                 float* value, bool probabilities) const {
    if (B <= 0) return;
    if (cuda_enabled_) {
        cuda_backend_->forward(states, B, policy, value, probabilities);
        return;
    }
    for (int b = 0; b < B; ++b) {
        float v = 0.0f;
        forward_impl(states + static_cast<size_t>(b) * 4 * CELLS,
                     policy + static_cast<size_t>(b) * NetConfig::kPolicyOut,
                     v, probabilities);
        value[b] = v;
    }
}

// Forward API.
void PureNet::forward_one(const std::vector<float>& state,
                          std::vector<float>& log_policy, float& value) const {
    if (state.size() != static_cast<size_t>(4 * CELLS))
        throw std::invalid_argument("state must contain 4*15*15 floats");
    log_policy.resize(NetConfig::kPolicyOut);
    forward_batch_impl(state.data(), 1, log_policy.data(), &value, false);
}

namespace {

// Rejects null buffers explicitly. A caller that passes an unsized vector's
// data() would otherwise get a segfault deep inside the backend with no hint
// about which argument was wrong.
void require_buffers(const float* states, float* policy, float* value) {
    if (states == nullptr || policy == nullptr || value == nullptr)
        throw std::invalid_argument(
            "forward_batch requires non-null state, policy and value buffers");
}

}  // namespace

void PureNet::forward_batch(const float* states, int B, float* log_policy,
                            float* value) const {
    if (B <= 0) return;
    require_buffers(states, log_policy, value);
    forward_batch_impl(states, B, log_policy, value, false);
}

void PureNet::forward_batch_policy(const float* states, int B, float* policy,
                                   float* value) const {
    if (B <= 0) return;
    require_buffers(states, policy, value);
    forward_batch_impl(states, B, policy, value, true);
}

PureNet::TrainStats PureNet::train_step(
    const std::vector<float>& states, int B, const std::vector<float>& probs,
    const std::vector<float>& winners, double lr, double value_loss_weight) {
    if (B <= 0 || states.size() != static_cast<size_t>(B) * 4 * CELLS ||
        probs.size() != static_cast<size_t>(B) * NetConfig::kPolicyOut ||
        winners.size() != static_cast<size_t>(B))
        throw std::invalid_argument("invalid training batch dimensions");
    if (!cuda_enabled_)
        throw std::logic_error(
            "training requires the LibTorch backend; rebuild with "
            "-DGOMOK_ENABLE_CUDA=ON or run with --cpu removed");
    return cuda_backend_->train_step(states, B, probs, winners, lr,
                                     value_loss_weight);
}

bool PureNet::load_native(const std::string& dir) {
    // Two on-disk forms are accepted.
    //
    //   Inference export (no training.meta): batch normalisation is already
    //   folded into the convolution, which is directly what the native forward
    //   pass consumes.
    //
    //   Training checkpoint (training.meta present): the convolution keeps its
    //   own weights and the batch norm keeps gamma/beta/running statistics
    //   separately, because folding destroys the parameterisation. The folded
    //   buffers are then rebuilt from those raw tensors.
    const bool training_format = has_training_state(dir);

    auto conv_slots = [&](size_t index) -> std::pair<ConvBlock*, FoldedConv*> {
        if (index == 0) return {&stem_, &stem_folded_};
        const size_t block = (index - 1) / 2;
        const bool second = ((index - 1) % 2) == 1;
        if (block >= trunk_.size()) return {nullptr, nullptr};
        return second ? std::make_pair(&trunk_[block]->b, &trunk_b_folded_[block])
                      : std::make_pair(&trunk_[block]->a, &trunk_a_folded_[block]);
    };

    if (training_format) {
        auto read_conv_raw = [&](size_t index, const std::string& prefix) {
            auto [block, folded] = conv_slots(index);
            if (block == nullptr) return false;
            ConvLayer& conv = block->conv;
            const size_t weights = static_cast<size_t>(conv.out_ch) * conv.in_ch *
                                   conv.kh * conv.kw;
            const size_t channels = static_cast<size_t>(conv.out_ch);
            return read_bin_file(dir + "/" + prefix + "weight.bin", conv.w,
                                 weights) &&
                   read_bin_file(dir + "/" + prefix + "bias.bin", conv.b,
                                 channels) &&
                   read_bin_file(dir + "/" + prefix + "bn_gamma.bin",
                                 block->bn_gamma, channels) &&
                   read_bin_file(dir + "/" + prefix + "bn_beta.bin",
                                 block->bn_beta, channels) &&
                   read_bin_file(dir + "/" + prefix + "bn_mean.bin",
                                 block->bn_mean, channels) &&
                   read_bin_file(dir + "/" + prefix + "bn_var.bin",
                                 block->bn_var, channels) &&
                   (conv.gw.assign(weights, 0.0f), true) &&
                   (conv.gb.assign(channels, 0.0f), true) &&
                   (folded != nullptr);
        };
        size_t index = 0;
        if (!read_conv_raw(index++, "training_conv0_")) return false;
        for (size_t i = 0; i < trunk_.size(); ++i) {
            if (!read_conv_raw(index++, "training_conv" + std::to_string(index - 1) + "_"))
                return false;
            if (!read_conv_raw(index++, "training_conv" + std::to_string(index - 1) + "_"))
                return false;
        }
        if (!read_conv_raw(index++, "training_conv" + std::to_string(index - 1) + "_"))
            return false;
        if (!read_conv_raw(index++, "training_conv" + std::to_string(index - 1) + "_"))
            return false;

        auto read_fc_raw = [&](size_t fc_index, FCLayer& layer) {
            const std::string prefix =
                "training_fc" + std::to_string(fc_index) + "_";
            return read_bin_file(dir + "/" + prefix + "weight.bin", layer.w,
                                 static_cast<size_t>(layer.out_n) * layer.in_n) &&
                   read_bin_file(dir + "/" + prefix + "bias.bin", layer.b,
                                 static_cast<size_t>(layer.out_n));
        };
        if (!read_fc_raw(0, policy_fc_)) return false;
        if (!read_fc_raw(1, value_fc1_)) return false;
        if (!read_fc_raw(2, value_fc2_)) return false;

        // Rebuild the folded buffers from the raw tensors, which is also what
        // keeps the native forward pass correct for a later --cpu switch.
        rebuild_folded_weights();
        loaded_ = true;
        return true;
    }

    auto read_folded = [&](const std::string& name, FoldedConv& folded,
                           ConvLayer& conv) {
        const size_t weights = static_cast<size_t>(conv.out_ch) * conv.in_ch *
                               conv.kh * conv.kw;
        const size_t biases = static_cast<size_t>(conv.out_ch);
        if (!read_bin_file(dir + "/" + name + "_weight.bin", folded.w, weights))
            return false;
        if (!read_bin_file(dir + "/" + name + "_bias.bin", folded.b, biases))
            return false;
        // Keep the structured copy in step so a later save() round-trips.
        conv.w = folded.w;
        conv.b = folded.b;
        conv.gw.assign(weights, 0.0f);
        conv.gb.assign(biases, 0.0f);
        return true;
    };
    auto identity_bn = [](ConvBlock& block) {
        const size_t channels = static_cast<size_t>(block.conv.out_ch);
        block.bn_gamma.assign(channels, 1.0f);
        block.bn_beta.assign(channels, 0.0f);
        block.bn_mean.assign(channels, 0.0f);
        block.bn_var.assign(channels, 1.0f - block.bn_eps);
    };
    auto read_fc = [&](const std::string& name, FCLayer& layer) {
        return read_bin_file(dir + "/" + name + "_weight.bin", layer.w,
                             static_cast<size_t>(layer.out_n) * layer.in_n) &&
               read_bin_file(dir + "/" + name + "_bias.bin", layer.b,
                             static_cast<size_t>(layer.out_n));
    };

    if (!read_folded("stem", stem_folded_, stem_.conv)) return false;
    identity_bn(stem_);
    for (size_t i = 0; i < trunk_.size(); ++i) {
        const std::string base = "res" + std::to_string(i);
        if (!read_folded(base + "_a", trunk_a_folded_[i], trunk_[i]->a.conv))
            return false;
        if (!read_folded(base + "_b", trunk_b_folded_[i], trunk_[i]->b.conv))
            return false;
        identity_bn(trunk_[i]->a);
        identity_bn(trunk_[i]->b);
    }
    if (!read_folded("policy_conv1", policy_folded_, policy_conv_.conv))
        return false;
    if (!read_folded("val_conv1", value_folded_, value_conv_.conv)) return false;
    identity_bn(policy_conv_);
    identity_bn(value_conv_);
    if (!read_fc("act_fc1", policy_fc_)) return false;
    if (!read_fc("val_fc1", value_fc1_)) return false;
    if (!read_fc("val_fc2", value_fc2_)) return false;

    loaded_ = true;
    return true;
}

bool PureNet::load(const std::string& dir) {
    // Training checkpoints carry the raw batch-norm parameterisation, so the
    // hook that generated them is also the one that can consume them.
    if (has_training_state(dir)) {
        int batches = 0;
        if (!load_training_state(dir, &batches)) return false;
        last_load_dir_ = dir;
        std::fprintf(stderr, "[net] resumed training state from %s (batch %d)\n",
                     dir.c_str(), batches);
        return true;
    }

    // Native buffers are always populated first so a later --cpu switch has
    // valid weights; the LibTorch module then becomes the authority.
    if (!load_native(dir)) return false;
    last_load_dir_ = dir;
    loaded_ = true;
    if (cuda_enabled_ && !cuda_backend_->load(dir)) return false;
    std::fprintf(stderr, "[net] loaded %zu parameters from %s%s\n",
                 parameter_count(), dir.c_str(),
                 cuda_enabled_ ? " (CUDA)" : "");
    return true;
}

bool PureNet::has_training_state(const std::string& dir) const {
    std::ifstream meta(dir + "/training.meta");
    return meta.is_open();
}

bool PureNet::export_inference(const std::string& dir) const {
    if (cuda_enabled_) return cuda_backend_->save(dir);
    // Native side: the folded buffers are already the inference form.
    auto write_folded = [&](const std::string& name, const FoldedConv& folded) {
        return write_bin_file(dir + "/" + name + "_weight.bin", folded.w) &&
               write_bin_file(dir + "/" + name + "_bias.bin", folded.b);
    };
    auto write_fc = [&](const std::string& name, const FCLayer& layer) {
        return write_bin_file(dir + "/" + name + "_weight.bin", layer.w) &&
               write_bin_file(dir + "/" + name + "_bias.bin", layer.b);
    };
    if (!write_folded("stem", stem_folded_)) return false;
    for (size_t i = 0; i < trunk_.size(); ++i) {
        const std::string base = "res" + std::to_string(i);
        if (!write_folded(base + "_a", trunk_a_folded_[i])) return false;
        if (!write_folded(base + "_b", trunk_b_folded_[i])) return false;
    }
    if (!write_folded("policy_conv1", policy_folded_)) return false;
    if (!write_folded("val_conv1", value_folded_)) return false;
    if (!write_fc("act_fc1", policy_fc_)) return false;
    if (!write_fc("val_fc1", value_fc1_)) return false;
    if (!write_fc("val_fc2", value_fc2_)) return false;
    return true;
}

bool PureNet::save_training_state(const std::string& dir,
                                  int batch_index) const {
    // Only the LibTorch backend trains, and its optimiser holds the moments, so
    // it owns the checkpoint. The native side has no training state to lose.
    if (cuda_enabled_)
        return cuda_backend_->save_training_state(dir, batch_index, 0, {}, {});
    return export_inference(dir);
}

bool PureNet::load_training_state(const std::string& dir,
                                  int* batch_index_out) {
    int batches = 0;
    if (cuda_enabled_) {
        if (!cuda_backend_->load_training_state(dir, &batches, nullptr, nullptr,
                                                nullptr))
            return false;
    } else if (!load_native(dir)) {
        return false;
    }
    // The checkpoint also carries the folded inference export, so the native
    // buffers stay valid for a --cpu switch.
    load_native(dir);
    resume_batch_index_ = batches;
    if (batch_index_out != nullptr) *batch_index_out = batches;
    return true;
}

size_t PureNet::parameter_count() const {
    size_t total = 0;
    auto add_conv = [&total](const ConvLayer& l) {
        total += l.w.size() + l.b.size();
    };
    auto add_block = [&](const ConvBlock& block) {
        add_conv(block.conv);
        total += block.bn_gamma.size() + block.bn_beta.size();
    };
    add_block(stem_);
    for (const auto& block : trunk_) { add_block(block->a); add_block(block->b); }
    add_block(policy_conv_);
    add_block(value_conv_);
    auto add_fc = [&total](const FCLayer& l) { total += l.w.size() + l.b.size(); };
    add_fc(policy_fc_);
    add_fc(value_fc1_);
    add_fc(value_fc2_);
    return total;
}

void PureNet::set_inference_threads(int threads) {
    inference_threads_ = std::max(1, threads);
}

bool PureNet::cuda_available() const {
    return cuda_backend_ && cuda_backend_->available();
}

void PureNet::set_weight_decay(double weight_decay) {
    adam_.set_weight_decay(weight_decay);
    if (cuda_backend_) cuda_backend_->set_weight_decay(weight_decay);
}

double PureNet::weight_decay() const { return adam_.weight_decay(); }

void PureNet::set_grad_clip(double max_norm) {
    grad_clip_ = max_norm > 0.0 ? max_norm : 0.0;
}

void PureNet::set_cuda_enabled(bool enabled) {
    const bool requested = enabled && cuda_available();
    if (requested == cuda_enabled_) return;
    if (!requested && cuda_enabled_) {
        cuda_enabled_ = false;
        // Live CUDA tensors cannot be mirrored exactly into the folded native
        // layout, so re-read the checkpoint that produced them.
        if (!last_load_dir_.empty()) load_native(last_load_dir_);
    } else if (requested) {
        cuda_backend_->sync_from_cpu();
        cuda_enabled_ = true;
    }
}

}  // namespace gomoku
