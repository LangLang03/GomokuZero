#include "train.h"
#include "logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace gomoku {

namespace {

bool read_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f) return false;
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(f);
}

// Index of the cell a board cell maps to after transform `t`.
//
// t & 3 selects a rotation, t & 4 optionally mirrors. The set of eight
// values forms the dihedral group of the square.
int transform_cell(int cell, int t) {
    int row = cell / BOARD_SIZE;
    int col = cell % BOARD_SIZE;
    int r = row, c = col;
    switch (t & 3) {
        case 1: r = col;              c = BOARD_SIZE - 1 - row; break;
        case 2: r = BOARD_SIZE - 1 - row; c = BOARD_SIZE - 1 - col; break;
        case 3: r = BOARD_SIZE - 1 - col; c = row;              break;
        default: break;
    }
    if (t & 4) c = BOARD_SIZE - 1 - c;
    return r * BOARD_SIZE + c;
}

// Applies transform `t` to a [4][15][15] state plane set.
void transform_state(const float* src, float* dst, int t) {
    for (int ch = 0; ch < 4; ++ch) {
        const float* in = src + static_cast<size_t>(ch) * BOARD_CELLS;
        float* out = dst + static_cast<size_t>(ch) * BOARD_CELLS;
        for (int cell = 0; cell < BOARD_CELLS; ++cell)
            out[transform_cell(cell, t)] = in[cell];
    }
}

// Blends `epsilon` of mass into a one-hot target, spread over the legal
// (empty) cells only.
//
// Spreading over all 225 cells would put probability on occupied, illegal
// moves. The shared state planes identify occupancy: channels 0 and 1 hold
// the two sides' stones.
void apply_label_smoothing(std::vector<float>& probs, const float* state,
                           float epsilon) {
    if (epsilon <= 0.0f || state == nullptr) return;
    const size_t n = probs.size();
    if (n == 0) return;

    int legal_count = 0;
    for (size_t cell = 0; cell < n; ++cell) {
        const bool occupied = state[cell] > 0.0f || state[n + cell] > 0.0f;
        if (!occupied) ++legal_count;
    }
    if (legal_count == 0) return;

    const float spread = epsilon / static_cast<float>(legal_count);
    const float keep = 1.0f - epsilon;
    for (size_t cell = 0; cell < n; ++cell) {
        const bool occupied = state[cell] > 0.0f || state[n + cell] > 0.0f;
        probs[cell] = occupied ? 0.0f : probs[cell] * keep + spread;
    }
}

double policy_kl(const std::vector<float>& log_p,
                 const std::vector<float>& log_q, int probes) {
    double total = 0.0;
    for (int i = 0; i < probes; ++i) {
        const float* lp = log_p.data() + static_cast<size_t>(i) * BOARD_CELLS;
        const float* lq = log_q.data() + static_cast<size_t>(i) * BOARD_CELLS;
        double row = 0.0;
        for (int k = 0; k < BOARD_CELLS; ++k)
            row += static_cast<double>(std::exp(lp[k])) * (lp[k] - lq[k]);
        total += row;
    }
    return probes > 0 ? total / probes : 0.0;
}

}  // namespace

Trainer::Trainer(PureNet& net, const SearchConfig& search,
                 const TrainConfig& config)
    : net_(net), search_(search), config_(config) {
    buffer_.reserve(config_.buffer_size);
    net_.set_weight_decay(config_.weight_decay);
    net_.set_grad_clip(static_cast<float>(config_.grad_clip));
    load_mix_data();
}

void Trainer::load_mix_data() {
    if (config_.mix_data.empty()) {
        logger.log("[train] no mix data");
        return;
    }
    std::vector<float> all_s, all_p, all_w;
    if (!read_bin(config_.mix_data + "_states.bin", all_s) ||
        !read_bin(config_.mix_data + "_probs.bin", all_p) ||
        !read_bin(config_.mix_data + "_winners.bin", all_w)) {
        logger.log("[train] mix data load failed (%s*)", config_.mix_data.c_str());
        n_mix_ = 0;
        return;
    }
    const size_t per_state = 4 * BOARD_CELLS;
    const int64_t total = static_cast<int64_t>(all_w.size());
    if (total <= 0 || all_s.size() < static_cast<size_t>(total) * per_state ||
        all_p.size() < static_cast<size_t>(total) * BOARD_CELLS) {
        logger.log("[train] mix data inconsistent (%lld winners, %zu states)",
                   static_cast<long long>(total), all_s.size());
        n_mix_ = 0;
        return;
    }

    // Random split. The files are ordered by game, so a prefix split would
    // make the validation set "the earliest games only".
    std::vector<int64_t> order(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) order[static_cast<size_t>(i)] = i;
    std::mt19937 rng(12345);
    std::shuffle(order.begin(), order.end(), rng);

    const int64_t n_val = std::max<int64_t>(1, total / 20);
    const int64_t n_train = total - n_val;
    n_val_ = n_val;
    n_mix_ = n_train;
    val_states_.assign(static_cast<size_t>(n_val) * per_state, 0.0f);
    val_probs_.assign(static_cast<size_t>(n_val) * BOARD_CELLS, 0.0f);
    val_winners_.assign(static_cast<size_t>(n_val), 0.0f);
    mix_states_.assign(static_cast<size_t>(n_train) * per_state, 0.0f);
    mix_probs_.assign(static_cast<size_t>(n_train) * BOARD_CELLS, 0.0f);
    mix_winners_.assign(static_cast<size_t>(n_train), 0.0f);

    auto copy_sample = [&](int64_t src, float* state, float* prob, float* win) {
        std::memcpy(state, &all_s[static_cast<size_t>(src) * per_state],
                    per_state * sizeof(float));
        std::memcpy(prob, &all_p[static_cast<size_t>(src) * BOARD_CELLS],
                    BOARD_CELLS * sizeof(float));
        *win = all_w[static_cast<size_t>(src)];
    };
    for (int64_t i = 0; i < n_val; ++i)
        copy_sample(order[static_cast<size_t>(i)],
                    &val_states_[static_cast<size_t>(i) * per_state],
                    &val_probs_[static_cast<size_t>(i) * BOARD_CELLS],
                    &val_winners_[static_cast<size_t>(i)]);
    for (int64_t i = 0; i < n_train; ++i)
        copy_sample(order[static_cast<size_t>(n_val + i)],
                    &mix_states_[static_cast<size_t>(i) * per_state],
                    &mix_probs_[static_cast<size_t>(i) * BOARD_CELLS],
                    &mix_winners_[static_cast<size_t>(i)]);

    // Human records are stored as hard one-hot targets.
    for (int64_t i = 0; i < n_train; ++i) {
        std::vector<float> row(
            mix_probs_.begin() + static_cast<size_t>(i) * BOARD_CELLS,
            mix_probs_.begin() + static_cast<size_t>(i + 1) * BOARD_CELLS);
        apply_label_smoothing(
            row, &mix_states_[static_cast<size_t>(i) * per_state],
            config_.label_smooth);
        std::copy(row.begin(), row.end(),
                  mix_probs_.begin() + static_cast<size_t>(i) * BOARD_CELLS);
    }

    // Reserve a small fixed probe batch for the KL diagnostic.
    n_kl_ = static_cast<int>(std::min<int64_t>(256, n_train));
    kl_states_.assign(static_cast<size_t>(n_kl_) * per_state, 0.0f);
    std::memcpy(kl_states_.data(), mix_states_.data(),
                kl_states_.size() * sizeof(float));

    logger.log("[train] mix data loaded: %lld train + %lld val samples, "
               "label_smooth=%.2f (%s*)",
               static_cast<long long>(n_mix_), static_cast<long long>(n_val_),
               config_.label_smooth, config_.mix_data.c_str());
}

void Trainer::add_samples(
    const std::vector<std::vector<BatchedSelfPlay::Sample>>& batches) {
    for (const auto& game : batches) {
        for (const auto& s : game) {
            Exp e;
            e.state.assign(s.state.begin(), s.state.end());
            e.probs.assign(s.probs.begin(), s.probs.end());
            e.z = s.z;
            if (static_cast<int>(buffer_.size()) < config_.buffer_size) {
                buffer_.push_back(std::move(e));
            } else {
                buffer_[buf_head_] = std::move(e);
                buf_head_ = (buf_head_ + 1) % config_.buffer_size;
            }
        }
    }
}

double Trainer::learning_rate_for_batch(int batch_index) const {
    const int total = std::max(1, config_.game_batch_num);
    const int warmup = std::max(0, config_.lr_warmup_batches);
    if (batch_index < warmup && warmup > 0) {
        // Linear warmup from 10% of the base rate.
        const double t = static_cast<double>(batch_index + 1) / warmup;
        return config_.lr_base * (0.1 + 0.9 * t);
    }
    const double denominator =
        static_cast<double>(std::max(1, total - warmup - 1));
    const double progress = std::min(
        1.0, std::max(0.0, static_cast<double>(batch_index - warmup) /
                               denominator));
    const double cosine = 0.5 * (1.0 + std::cos(M_PI * progress));
    const double final_fraction = config_.lr_final_fraction;
    return config_.lr_base * (final_fraction + (1.0 - final_fraction) * cosine);
}

void Trainer::policy_update(int batch_index) {
    if (buffer_.size() < static_cast<size_t>(config_.batch_size)) return;
    std::mt19937 rng(std::random_device{}());

    const int batch_size = config_.batch_size;
    const int n_self = static_cast<int>(buffer_.size());
    int n_mix = 0;
    if (n_mix_ > 0) n_mix = static_cast<int>(batch_size * config_.mix_ratio);
    int n_self_use = batch_size - n_mix;
    if (n_self_use > n_self) {
        n_self_use = n_self;
        n_mix = batch_size - n_self_use;
    }

    const size_t per_state = 4 * BOARD_CELLS, per_prob = BOARD_CELLS;
    std::vector<float> s_buf(static_cast<size_t>(batch_size) * per_state);
    std::vector<float> p_buf(static_cast<size_t>(batch_size) * per_prob);
    std::vector<float> z_buf(batch_size);

    std::uniform_int_distribution<int> self_dist(0, n_self - 1);
    std::uniform_int_distribution<int> sym_dist(0, config_.augment ? 7 : 0);
    for (int i = 0; i < n_self_use; ++i) {
        const Exp& e = buffer_[self_dist(rng)];
        const int t = sym_dist(rng);
        float* state = &s_buf[static_cast<size_t>(i) * per_state];
        float* prob = &p_buf[static_cast<size_t>(i) * per_prob];
        if (t == 0) {
            std::memcpy(state, e.state.data(), per_state * sizeof(float));
            std::memcpy(prob, e.probs.data(), per_prob * sizeof(float));
        } else {
            transform_state(e.state.data(), state, t);
            for (int cell = 0; cell < BOARD_CELLS; ++cell)
                prob[transform_cell(cell, t)] = e.probs[cell];
        }
        z_buf[i] = e.z;
    }
    if (n_mix > 0) {
        std::uniform_int_distribution<int64_t> mix_dist(0, n_mix_ - 1);
        for (int i = 0; i < n_mix; ++i) {
            const int64_t idx = mix_dist(rng);
            const int t = sym_dist(rng);
            float* state = &s_buf[static_cast<size_t>(n_self_use + i) * per_state];
            float* prob = &p_buf[static_cast<size_t>(n_self_use + i) * per_prob];
            const float* src_state =
                &mix_states_[static_cast<size_t>(idx) * per_state];
            const float* src_prob =
                &mix_probs_[static_cast<size_t>(idx) * per_prob];
            if (t == 0) {
                std::memcpy(state, src_state, per_state * sizeof(float));
                std::memcpy(prob, src_prob, per_prob * sizeof(float));
            } else {
                transform_state(src_state, state, t);
                for (int cell = 0; cell < BOARD_CELLS; ++cell)
                    prob[transform_cell(cell, t)] = src_prob[cell];
            }
            z_buf[n_self_use + i] = mix_winners_[static_cast<size_t>(idx)];
        }
    }

    const double lr = learning_rate_for_batch(batch_index);
    PureNet::TrainStats stats{};
    for (int e = 0; e < config_.epochs; ++e)
        stats = net_.train_step(s_buf, batch_size, p_buf, z_buf, lr,
                                config_.value_loss_weight);
    logger.log("batch %d loss:%.6f entropy:%.6f lr:%.6f", batch_index + 1,
               stats.loss, stats.entropy, lr);

    // Capture the post-update policy so the KL diagnostic is a pure
    // observation. train_step() inside the probe would add an optimisation
    // step, advance Adam's moments and shift the BN statistics.
    if (n_kl_ > 0) {
        std::vector<float> value(static_cast<size_t>(n_kl_));
        kl_after_.assign(static_cast<size_t>(n_kl_) * BOARD_CELLS, 0.0f);
        net_.forward_batch(kl_states_.data(), n_kl_, kl_after_.data(),
                           value.data());
    }
}

void Trainer::update_kl_probe() {
    if (n_kl_ <= 0) return;
    // The buffers must be sized here: nothing else allocates them, and passing
    // an empty vector's data() to forward_batch dereferences a null pointer.
    const size_t size = static_cast<size_t>(n_kl_) * BOARD_CELLS;
    kl_before_.assign(size, 0.0f);
    std::vector<float> value(static_cast<size_t>(n_kl_));
    net_.forward_batch(kl_states_.data(), n_kl_, kl_before_.data(),
                       value.data());
}

void Trainer::log_true_kl(int batch_index) const {
    if (n_kl_ <= 0 || kl_before_.empty() || kl_after_.empty()) return;
    const double kl = policy_kl(kl_before_, kl_after_, n_kl_);
    logger.log("batch %d lr:%.6f true_kl:%.6f", batch_index + 1,
               learning_rate_for_batch(batch_index), kl);
}

void Trainer::validate(int batch_idx) {
    if (n_val_ == 0) return;
    const size_t per_state = 4 * BOARD_CELLS, per_prob = BOARD_CELLS;
    const int chunk = 512;
    double ploss_sum = 0, vloss_sum = 0;
    int correct = 0;
    std::vector<float> lp(static_cast<size_t>(chunk) * BOARD_CELLS), val_out(chunk);
    for (int64_t off = 0; off < n_val_; off += chunk) {
        const int n = static_cast<int>(std::min<int64_t>(chunk, n_val_ - off));
        net_.forward_batch(&val_states_[static_cast<size_t>(off) * per_state], n,
                           lp.data(), val_out.data());
        for (int i = 0; i < n; ++i) {
            const float* p = &val_probs_[static_cast<size_t>(off + i) * per_prob];
            const float* l = &lp[static_cast<size_t>(i) * BOARD_CELLS];
            const float z = val_winners_[off + i];
            double pl = 0;
            for (int k = 0; k < BOARD_CELLS; ++k)
                pl -= static_cast<double>(p[k]) * l[k];
            ploss_sum += pl;
            vloss_sum += (val_out[i] - z) * (val_out[i] - z);
            int argmax_l = 0, argmax_p = 0;
            for (int k = 1; k < BOARD_CELLS; ++k) {
                if (l[k] > l[argmax_l]) argmax_l = k;
                if (p[k] > p[argmax_p]) argmax_p = k;
            }
            if (argmax_l == argmax_p) ++correct;
        }
    }
    const double n = static_cast<double>(n_val_);
    logger.log("[val] batch %d | val_loss:%.4f (pol:%.4f val:%.4f) | top1_acc:%.3f",
               batch_idx, (ploss_sum + vloss_sum) / n, ploss_sum / n,
               vloss_sum / n, static_cast<double>(correct) / n);
}

void Trainer::evaluate_and_save(int batch_idx) {
    const std::string dir = "best_policy_" + config_.tag + "_bin";
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error) {
        logger.log("[error] cannot create %s: %s", dir.c_str(),
                   error.message().c_str());
        throw std::runtime_error("checkpoint directory is not writable: " + dir);
    }

    // A full training checkpoint carries the raw batch-norm parameterisation
    // and the optimiser moments, which a folded inference export cannot
    // express. It also writes the inference tensors into the same directory.
    if (!net_.save_training_state(dir, batch_idx)) {
        logger.log("[error] checkpoint save failed (%s)", dir.c_str());
        throw std::runtime_error("checkpoint save failed: " + dir);
    }
    logger.log("[train] batch %d checkpoint saved (%s)", batch_idx, dir.c_str());
}

void Trainer::run_arena(int batch_idx) {
    if (config_.eval_games <= 0 || config_.baseline_dir.empty()) return;

    // The baseline stays on whatever backend is active. play_match() merges
    // both models' leaves into batched calls, so putting the baseline on the
    // native CPU path would stall every round on its much slower forward pass.
    PureNet baseline(2e-3);
    if (!baseline.load(config_.baseline_dir)) {
        logger.log("[arena] baseline load failed (%s); skipping",
                   config_.baseline_dir.c_str());
        return;
    }
    baseline.set_cuda_enabled(net_.using_cuda());
    baseline.set_inference_threads(net_.inference_threads());

    SearchConfig arena = search_;
    arena.n_playout = std::max(1, config_.eval_playout);
    const auto started = std::chrono::steady_clock::now();
    const MatchResult result =
        play_match(net_, baseline, arena, config_.eval_games);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    logger.log("[arena] batch %d | vs %s | %dW %dL %dD | score %.3f | "
               "first-player win rate %.2f | %.0fs",
               batch_idx, config_.baseline_dir.c_str(), result.wins,
               result.losses, result.draws, result.score(),
               result.first_player_win_rate(), seconds);
    if (result.first_player_win_rate() > 0.95 && result.games >= 8) {
        logger.log("[arena] note: the first player won almost every game, so "
                   "this score reflects the first-move advantage rather than "
                   "the difference between the two models");
    }
}

void Trainer::run() {
    BatchedSelfPlay sp(net_, search_);
    auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;

    // Resuming must continue the schedule, not restart it. The learning rate is
    // a function of the absolute batch index, so picking up from the checkpoint
    // keeps the decay where it left off and avoids re-running warmup.
    const int start_batch = std::min(net_.resume_batch_index(),
                                     config_.game_batch_num);
    if (start_batch > 0) {
        logger.log("[train] resuming at batch %d/%d (lr %.6f)", start_batch,
                   config_.game_batch_num, learning_rate_for_batch(start_batch));
    }

    for (int i = start_batch; i < config_.game_batch_num; ++i) {
        // Negative override: use the configured temperature schedule.
        auto batches = sp.run_batch(-1.0f);
        add_samples(batches);
        int n_samples = 0;
        for (auto& g : batches) n_samples += static_cast<int>(g.size());

        update_kl_probe();
        policy_update(i);
        if ((i + 1) % 5 == 0) log_true_kl(i);

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - t_last).count();
        const float elapsed = std::chrono::duration<float>(now - t0).count();
        t_last = now;

        const float eta = dt * (config_.game_batch_num - i - 1) / 3600.0f;
        logger.log("batch i:%d/%d, samples:%d, buffer:%zu, lr:%.6f, "
                   "selfplay_time:%.1fs",
                   i + 1, config_.game_batch_num, n_samples, buffer_.size(),
                   learning_rate_for_batch(i), dt);
        if ((i + 1) % 10 == 0) {
            logger.log("[PROGRESS] batch %d/%d | elapsed %.1fh | ETA %.1fh",
                       i + 1, config_.game_batch_num, elapsed / 3600.0f, eta);
        }
        if ((i + 1) % config_.check_freq == 0) evaluate_and_save(i + 1);
        if ((i + 1) % 5 == 0) validate(i + 1);
        if (config_.eval_freq > 0 && (i + 1) % config_.eval_freq == 0)
            run_arena(i + 1);
    }
    evaluate_and_save(config_.game_batch_num);
}

}  // namespace gomoku
