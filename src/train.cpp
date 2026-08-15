#include "train.h"
#include "logger.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdlib>

namespace gomoku {

static bool read_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    f.close();
    return true;
}

Trainer::Trainer(PureNet& net, int n_playout, int batch_games,
                 int n_threads, float c_puct, float temp, int buffer_size,
                 int batch_size, int epochs, int check_freq, int game_batch_num,
                 const std::string& mix_data, float mix_ratio,
                 const std::string& tag,
                 double lr_multiplier_init)
    : net_(net), n_playout_(n_playout), batch_games_(batch_games),
      n_threads_(n_threads), buffer_size_(buffer_size), batch_size_(batch_size),
      epochs_(epochs), check_freq_(check_freq), game_batch_num_(game_batch_num),
      c_puct_(c_puct), temp_(temp), mix_data_(mix_data), mix_ratio_(mix_ratio),
      tag_(tag), lr_multiplier_(lr_multiplier_init) {
    buffer_.reserve(buffer_size_);
    load_mix_data();
}

void Trainer::load_mix_data() {
    if (mix_data_.empty()) {
        logger.log("[train] no mix data");
        return;
    }
    std::vector<float> all_s, all_p, all_w;
    if (!read_bin(mix_data_ + "_states.bin", all_s) ||
        !read_bin(mix_data_ + "_probs.bin", all_p) ||
        !read_bin(mix_data_ + "_winners.bin", all_w)) {
        logger.log("[train] mix data load failed (%s*)", mix_data_.c_str());
        n_mix_ = 0;
        return;
    }
    int64_t total = static_cast<int64_t>(all_w.size());
    n_val_ = std::max<int64_t>(1, total / 20);
    n_mix_ = total - n_val_;
    const size_t per_state = 4 * 225, per_prob = 225;
    val_states_.assign(all_s.begin(), all_s.begin() + n_val_ * per_state);
    val_probs_.assign(all_p.begin(), all_p.begin() + n_val_ * per_prob);
    val_winners_.assign(all_w.begin(), all_w.begin() + n_val_);
    mix_states_.assign(all_s.begin() + n_val_ * per_state, all_s.end());
    mix_probs_.assign(all_p.begin() + n_val_ * per_prob, all_p.end());
    mix_winners_.assign(all_w.begin() + n_val_, all_w.end());
    logger.log("[train] mix data loaded: %lld train + %lld val samples (%s*)",
               (long long)n_mix_, (long long)n_val_, mix_data_.c_str());
}

void Trainer::add_samples(
    const std::vector<std::vector<BatchedSelfPlay::Sample>>& batches) {
    for (const auto& game : batches) {
        for (const auto& s : game) {
            Exp e;
            e.state.assign(s.state.begin(), s.state.end());
            e.probs.assign(s.probs.begin(), s.probs.end());
            e.z = s.z;
            if (static_cast<int>(buffer_.size()) < buffer_size_) {
                buffer_.push_back(std::move(e));
            } else {
                buffer_[buf_head_] = std::move(e);
                buf_head_ = (buf_head_ + 1) % buffer_size_;
            }
        }
    }
}

void Trainer::policy_update() {
    if (buffer_.size() < static_cast<size_t>(batch_size_)) return;
    std::mt19937 rng(std::random_device{}());

    int n_self = static_cast<int>(buffer_.size());
    int n_mix = 0;
    if (n_mix_ > 0) n_mix = static_cast<int>(batch_size_ * mix_ratio_);
    int n_self_use = batch_size_ - n_mix;
    if (n_self_use > n_self) {
        n_self_use = n_self;
        n_mix = batch_size_ - n_self_use;
    }

    const size_t per_state = 4 * 225, per_prob = 225;
    std::vector<float> s_buf((size_t)batch_size_ * per_state);
    std::vector<float> p_buf((size_t)batch_size_ * per_prob);
    std::vector<float> z_buf(batch_size_);

    std::uniform_int_distribution<int> self_dist(0, n_self - 1);
    for (int i = 0; i < n_self_use; ++i) {
        const Exp& e = buffer_[self_dist(rng)];
        std::memcpy(&s_buf[(size_t)i * per_state], e.state.data(),
                    per_state * sizeof(float));
        std::memcpy(&p_buf[(size_t)i * per_prob], e.probs.data(),
                    per_prob * sizeof(float));
        z_buf[i] = e.z;
    }
    if (n_mix_ > 0 && n_mix > 0) {
        std::uniform_int_distribution<int64_t> mix_dist(0, n_mix_ - 1);
        for (int i = 0; i < n_mix; ++i) {
            int64_t idx = mix_dist(rng);
            std::memcpy(&s_buf[(size_t)(n_self_use + i) * per_state],
                        &mix_states_[(size_t)idx * per_state],
                        per_state * sizeof(float));
            std::memcpy(&p_buf[(size_t)(n_self_use + i) * per_prob],
                        &mix_probs_[(size_t)idx * per_prob],
                        per_prob * sizeof(float));
            z_buf[n_self_use + i] = mix_winners_[idx];
        }
    }

    double lr = 2e-3 * lr_multiplier_;
    PureNet::TrainStats stats{};
    for (int e = 0; e < epochs_; ++e) {
        stats = net_.train_step(s_buf, batch_size_, p_buf, z_buf, lr);
    }

    // Approximate KL by the change in loss, smoothed to reduce noise.
    static float prev_loss = 0;
    static float smoothed_kl = 0.05f;
    const float raw_kl = prev_loss > 0 ? std::abs(stats.loss - prev_loss) : 0.05f;
    smoothed_kl = 0.7f * smoothed_kl + 0.3f * raw_kl;
    const float kl = smoothed_kl;
    prev_loss = stats.loss;
    if (kl > kl_targ_ * 4) {
        lr_multiplier_ /= 1.5;
    } else if (kl > kl_targ_ * 2 && lr_multiplier_ > 0.1) {
        lr_multiplier_ /= 1.5;
    } else if (kl < kl_targ_ / 2 && lr_multiplier_ < 2.0) {
        lr_multiplier_ *= 1.5;
    }

    // Avoid total LR collapse: keep a small minimum learning rate.
    if (lr_multiplier_ < 0.01) lr_multiplier_ = 0.01;
    if (lr_multiplier_ > 2.0) lr_multiplier_ = 2.0;

    logger.log("kl:%.5f,lr_multiplier:%.3f,loss:%.6f,entropy:%.6f",
               kl, lr_multiplier_, stats.loss, stats.entropy);
}

void Trainer::validate(int batch_idx) {
    if (n_val_ == 0) return;
    const size_t per_state = 4 * 225, per_prob = 225;
    const int chunk = 512;
    double ploss_sum = 0, vloss_sum = 0;
    int correct = 0;
    std::vector<float> lp(chunk * 225), val_out(chunk);
    for (int64_t off = 0; off < n_val_; off += chunk) {
        int n = (int)std::min<int64_t>(chunk, n_val_ - off);
        net_.forward_batch(&val_states_[(size_t)off * per_state], n,
                           lp.data(), val_out.data());
        for (int i = 0; i < n; ++i) {
            const float* p = &val_probs_[(size_t)(off + i) * per_prob];
            const float* l = &lp[(size_t)i * 225];
            float z = val_winners_[off + i];
            double pl = 0;
            for (int k = 0; k < 225; ++k) pl -= (double)p[k] * l[k];
            ploss_sum += pl;
            vloss_sum += (val_out[i] - z) * (val_out[i] - z);
            int argmax_l = 0, argmax_p = 0;
            for (int k = 1; k < 225; ++k) {
                if (l[k] > l[argmax_l]) argmax_l = k;
                if (p[k] > p[argmax_p]) argmax_p = k;
            }
            if (argmax_l == argmax_p) ++correct;
        }
    }
    double n = (double)n_val_;
    logger.log("[val] batch %d | val_loss:%.4f (pol:%.4f val:%.4f) | top1_acc:%.3f",
               batch_idx,
               (ploss_sum + vloss_sum) / n,
               ploss_sum / n, vloss_sum / n,
               (double)correct / n);
}

void Trainer::evaluate_and_save(int batch_idx) {
    std::string dir = "best_policy_" + tag_ + "_bin";
    std::string cmd = "mkdir -p " + dir;
    (void)system(cmd.c_str());
    net_.save(dir);
    logger.log("[train] batch %d model saved (best_policy_%s_bin)",
               batch_idx, tag_.c_str());
}

void Trainer::run() {
    BatchedSelfPlay sp(net_, c_puct_, n_playout_, batch_games_, n_threads_);
    auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;

    for (int i = 0; i < game_batch_num_; ++i) {
        auto batches = sp.run_batch(temp_);
        add_samples(batches);
        int n_samples = 0;
        for (auto& g : batches) n_samples += static_cast<int>(g.size());

        policy_update();

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - t_last).count();
        float elapsed = std::chrono::duration<float>(now - t0).count();
        t_last = now;

        float eta = dt * (game_batch_num_ - i - 1) / 3600.0f;
        logger.log("batch i:%d/%d, samples:%d, buffer:%zu, selfplay_time:%.1fs",
                   i + 1, game_batch_num_, n_samples, buffer_.size(), dt);
        if ((i + 1) % 10 == 0) {
            logger.log("[PROGRESS] batch %d/%d | elapsed %.1fh | ETA %.1fh",
                       i + 1, game_batch_num_, elapsed / 3600.0f, eta);
        }
        if ((i + 1) % check_freq_ == 0) evaluate_and_save(i + 1);
        if ((i + 1) % 5 == 0) validate(i + 1);
    }
    evaluate_and_save(game_batch_num_);
}

}  // namespace gomoku
