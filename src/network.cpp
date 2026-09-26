#include "network.h"
#include "network_cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <filesystem>
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
    // Small tensors can stay in the stream buffer until destruction. Check
    // flush/close explicitly so a failed final write cannot report success.
    f.flush();
    f.close();
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
      value_fc2_(NetConfig::kValueHidden, 1) {
    if (!std::isfinite(lr) || lr <= 0.0)
        throw std::invalid_argument("learning rate must be finite and positive");
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

    uint32_t seed = static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    if (const char* configured = std::getenv("GOMOK_SEED")) {
        size_t consumed = 0;
        const std::string text(configured);
        const auto parsed = std::stoull(text, &consumed);
        if (text.empty() || text.front() == '-' || consumed != text.size())
            throw std::invalid_argument("GOMOK_SEED must be a nonnegative integer");
        seed = static_cast<uint32_t>(parsed);
    }
    std::mt19937 rng(seed);

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
    };
    auto init_fc = [&rng](FCLayer& layer) {
        const size_t weights =
            static_cast<size_t>(layer.out_n) * static_cast<size_t>(layer.in_n);
        layer.w.assign(weights, 0.0f);
        layer.b.assign(static_cast<size_t>(layer.out_n), 0.0f);
        const float bound = std::sqrt(6.0f / static_cast<float>(layer.in_n));
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (float& v : layer.w) v = dist(rng);
    };

    init_conv(stem_.conv, 1.0f);
    for (auto& block : trunk_) {
        init_conv(block->a.conv, 1.0f);
        init_conv(block->b.conv, 1.0f);
    }
    init_conv(policy_conv_.conv, 1.0f);
    init_conv(value_conv_.conv, 1.0f);
    init_fc(policy_fc_);
    init_fc(value_fc1_);
    init_fc(value_fc2_);
    // Before outcome supervision exists, MCTS must receive a neutral value
    // prior. A random final head would drift with the policy-trained trunk
    // despite having no evidence about which player will win. Consume the
    // same random draws above, then zero only this last layer; all other
    // parameters and their seeded initialization remain unchanged.
    std::fill(value_fc2_.w.begin(), value_fc2_.w.end(), 0.0f);
    std::fill(value_fc2_.b.begin(), value_fc2_.b.end(), 0.0f);

    auto init_bn = [](ConvBlock& block) {
        block.bn_gamma.assign(static_cast<size_t>(block.conv.out_ch), 1.0f);
        block.bn_beta.assign(static_cast<size_t>(block.conv.out_ch), 0.0f);
        block.bn_mean.assign(static_cast<size_t>(block.conv.out_ch), 0.0f);
        block.bn_var.assign(static_cast<size_t>(block.conv.out_ch), 1.0f);
    };
    init_bn(stem_);
    for (auto& block : trunk_) {
        init_bn(block->a);
        init_bn(block->b);
        // Scaling the convolution is cancelled by BN during training. Scale
        // gamma so the residual branch really starts small in both modes.
        std::fill(block->b.bn_gamma.begin(), block->b.bn_gamma.end(), 0.1f);
    }
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
    const int workers = std::min(B, inference_threads_);
    auto run = [&](int worker) {
        for (int b = worker; b < B; b += workers)
            forward_impl(states + static_cast<size_t>(b) * NetConfig::kInputChannels * CELLS,
                         policy + static_cast<size_t>(b) * NetConfig::kPolicyOut,
                         value[b], probabilities);
    };
    if (workers == 1) {
        run(0);
        return;
    }
    std::vector<std::exception_ptr> errors(static_cast<size_t>(workers));
    auto checked_run = [&](int worker) {
        try { run(worker); }
        catch (...) { errors[worker] = std::current_exception(); }
    };
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(workers - 1));
    try {
        for (int worker = 1; worker < workers; ++worker)
            threads.emplace_back(checked_run, worker);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    checked_run(0);
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
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
    const std::vector<float>& winners, double lr, double value_loss_weight,
    const std::vector<float>& value_weights) {
    if (B <= 0 || states.size() != static_cast<size_t>(B) * 4 * CELLS ||
        probs.size() != static_cast<size_t>(B) * NetConfig::kPolicyOut ||
        winners.size() != static_cast<size_t>(B))
        throw std::invalid_argument("invalid training batch dimensions");
    if (!value_weights.empty() && value_weights.size() != static_cast<size_t>(B))
        throw std::invalid_argument("invalid value weight dimensions");
    for (float w : value_weights)
        if (!std::isfinite(w) || w < 0.0f)
            throw std::invalid_argument("invalid value weight");
    if (!std::isfinite(lr) || lr <= 0.0 || !std::isfinite(value_loss_weight) ||
        value_loss_weight < 0.0)
        throw std::invalid_argument("invalid learning rate or value loss weight");
    for (float x : states)
        if (!std::isfinite(x)) throw std::invalid_argument("non-finite state");
    for (int b = 0; b < B; ++b) {
        if (!std::isfinite(winners[b]) || std::abs(winners[b]) > 1.0f)
            throw std::invalid_argument("value target must be in [-1, 1]");
        double sum = 0.0;
        for (int i = 0; i < NetConfig::kPolicyOut; ++i) {
            const float p = probs[static_cast<size_t>(b) * NetConfig::kPolicyOut + i];
            if (!std::isfinite(p) || p < 0.0f)
                throw std::invalid_argument("invalid policy target");
            sum += p;
        }
        if (std::abs(sum - 1.0) > 1e-3)
            throw std::invalid_argument("policy target must sum to one");
    }
    if (!cuda_enabled_)
        throw std::logic_error(
            "training requires the LibTorch backend; rebuild with "
            "-DGOMOK_ENABLE_CUDA=ON or run with --cpu removed");
    return cuda_backend_->train_step(states, B, probs, winners, lr,
                                     value_loss_weight, value_weights);
}

bool PureNet::load_native(const std::string& dir) {
    const bool training_format = has_training_state(dir);
    if (training_format) {
        std::ifstream meta(dir + "/training.meta");
        auto field = [&](const char* expected, auto& value) {
            std::string key;
            return static_cast<bool>(meta >> key >> value) && key == expected;
        };
        int version = 0, batches = 0, conv_count = 0, fc_count = 0;
        int64_t step = 0;
        if (!field("version", version) || (version != 1 && version != 2) ||
            !field("batch_index", batches) || batches < 0 ||
            !field("optimizer_step", step) || step < 0 ||
            !field("conv_count", conv_count) || conv_count != 3 + 2 * NetConfig::kResBlocks ||
            !field("fc_count", fc_count) || fc_count != 3) return false;
    }
    std::vector<std::pair<std::vector<float>*, std::vector<float>>> pending;
    auto prepare = [&](const std::string& file, std::vector<float>& target,
                       size_t size, bool variance = false) {
        std::vector<float> data;
        if (!read_bin_file(dir + "/" + file, data, size)) return false;
        for (float x : data)
            if (!std::isfinite(x) || (variance && x < 0.0f)) return false;
        pending.emplace_back(&target, std::move(data));
        return true;
    };
    std::vector<std::pair<std::string, ConvBlock*>> convs{{"stem", &stem_}};
    for (size_t i = 0; i < trunk_.size(); ++i) {
        convs.emplace_back("res" + std::to_string(i) + "_a", &trunk_[i]->a);
        convs.emplace_back("res" + std::to_string(i) + "_b", &trunk_[i]->b);
    }
    convs.emplace_back("policy_conv1", &policy_conv_);
    convs.emplace_back("val_conv1", &value_conv_);
    for (size_t i = 0; i < convs.size(); ++i) {
        auto& block = *convs[i].second;
        auto& conv = block.conv;
        const auto prefix = training_format ? "training_conv" + std::to_string(i) : convs[i].first;
        const size_t channels = static_cast<size_t>(conv.out_ch);
        if (!prepare(prefix + "_weight.bin", conv.w, channels * conv.in_ch * conv.kh * conv.kw) ||
            !prepare(prefix + "_bias.bin", conv.b, channels)) return false;
        if (training_format &&
            (!prepare(prefix + "_bn_gamma.bin", block.bn_gamma, channels) ||
             !prepare(prefix + "_bn_beta.bin", block.bn_beta, channels) ||
             !prepare(prefix + "_bn_mean.bin", block.bn_mean, channels) ||
             !prepare(prefix + "_bn_var.bin", block.bn_var, channels, true))) return false;
    }
    const std::vector<std::pair<std::string, FCLayer*>> fcs{
        {"act_fc1", &policy_fc_}, {"val_fc1", &value_fc1_}, {"val_fc2", &value_fc2_}};
    for (size_t i = 0; i < fcs.size(); ++i) {
        auto& fc = *fcs[i].second;
        const auto prefix = training_format ? "training_fc" + std::to_string(i) : fcs[i].first;
        if (!prepare(prefix + "_weight.bin", fc.w, static_cast<size_t>(fc.in_n) * fc.out_n) ||
            !prepare(prefix + "_bias.bin", fc.b, fc.out_n)) return false;
    }
    for (auto& item : pending) *item.first = std::move(item.second);
    if (!training_format) {
        for (auto& entry : convs) {
            auto& block = *entry.second;
            const size_t channels = static_cast<size_t>(block.conv.out_ch);
            block.bn_gamma.assign(channels, 1.0f);
            block.bn_beta.assign(channels, 0.0f);
            block.bn_mean.assign(channels, 0.0f);
            block.bn_var.assign(channels, 1.0f - block.bn_eps);
        }
    }
    rebuild_folded_weights();
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

    // Validate and commit through one authority. Reading the directory twice
    // could leave a new native model paired with old CUDA weights when the
    // second read failed (e.g. a concurrently replaced checkpoint).
    if (cuda_available()) {
        if (!cuda_backend_->load(dir)) return false;
        cuda_backend_->sync_to_cpu();
    } else if (!load_native(dir)) return false;
    last_load_dir_ = dir;
    loaded_ = true;
    resume_batch_index_ = 0;
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
    // An existing training.meta would make load() prefer stale raw weights
    // over this export. Refuse mixing both artifact types through this API.
    if (has_training_state(dir)) return false;
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error) return false;
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
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error || batch_index < 0) return false;
    // Only the LibTorch backend trains, and its optimiser holds the moments, so
    // it owns the checkpoint. The native side has no training state to lose.
    // A temporary switch to native inference must not discard the live Adam
    // moments. The CUDA module remains the training authority while disabled.
    if (cuda_available())
        return cuda_backend_->save_training_state(dir, batch_index, 0, {}, {});
    return false;
}

bool PureNet::load_training_state(const std::string& dir,
                                  int* batch_index_out) {
    int batches = 0;
    if (cuda_available()) {
        if (!cuda_backend_->load_training_state(dir, &batches, nullptr, nullptr,
                                                nullptr))
            return false;
        cuda_backend_->sync_to_cpu();
    } else {
        if (!load_native(dir)) return false;
        std::ifstream meta(dir + "/training.meta");
        std::string key;
        int version = 0;
        if (!(meta >> key >> version) || key != "version" ||
            !(meta >> key >> batches) || key != "batch_index") return false;
    }
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

void PureNet::synchronize() const {
    if (cuda_available()) cuda_backend_->synchronize();
}

void PureNet::set_weight_decay(double weight_decay) {
    if (!std::isfinite(weight_decay) || weight_decay < 0.0)
        throw std::invalid_argument("weight decay must be finite and nonnegative");
    weight_decay_ = weight_decay;
    if (cuda_backend_) cuda_backend_->set_weight_decay(weight_decay);
}

double PureNet::weight_decay() const { return weight_decay_; }

void PureNet::set_grad_clip(double max_norm) {
    if (!std::isfinite(max_norm))
        throw std::invalid_argument("gradient clip must be finite");
    grad_clip_ = max_norm > 0.0 ? max_norm : 0.0;
}

void PureNet::set_cuda_enabled(bool enabled) {
    const bool requested = enabled && cuda_available();
    if (requested == cuda_enabled_) return;
    if (!requested && cuda_enabled_) {
        cuda_backend_->sync_to_cpu();
        cuda_enabled_ = false;
    } else if (requested) {
        // CPU inference does not modify weights; the CUDA optimiser remains
        // authoritative and can resume without resetting its moments.
        cuda_enabled_ = true;
    }
}

}  // namespace gomoku
