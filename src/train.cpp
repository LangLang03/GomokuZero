#include "train.h"
#include "logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gomoku {
namespace {
constexpr size_t kState = 4 * BOARD_CELLS;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string format_duration(double seconds) {
    const long long total = static_cast<long long>(std::max(0.0, std::round(seconds)));
    const long long days = total / 86400;
    const long long hours = (total % 86400) / 3600;
    const long long minutes = (total % 3600) / 60;
    const long long secs = total % 60;
    char text[48];
    if (days > 0)
        std::snprintf(text, sizeof(text), "%lldd %02lld:%02lld:%02lld", days, hours, minutes, secs);
    else if (hours > 0)
        std::snprintf(text, sizeof(text), "%02lld:%02lld:%02lld", hours, minutes, secs);
    else
        std::snprintf(text, sizeof(text), "%02lld:%02lld", minutes, secs);
    return text;
}
uint64_t stream_seed(uint64_t seed, uint64_t phase, uint64_t index) {
    uint64_t x = seed ^ (phase * 0x9e3779b97f4a7c15ULL) ^ index;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
bool nonnegative(double x) { return std::isfinite(x) && x >= 0; }
// Persist the numerical training contract. Extending the requested number of
// batches/epochs is allowed; silently changing the saved experiment is not.
std::string training_contract(const TrainConfig& c, const SearchConfig& s,
                              bool selfplay_started, bool include_tree_reuse = true) {
    std::ostringstream out;
    out << std::setprecision(17) << c.batch_size
        << ' ' << c.lr_base << ' ' << c.lr_final_fraction
        << ' ' << c.pretrain_warmup_steps << ' ' << c.weight_decay << ' ' << c.grad_clip
        << ' ' << c.value_loss_weight << ' ' << c.label_smooth
        << ' ' << c.mix_value_weight << ' ' << c.validation_fraction << ' ' << c.augment
        << ' ' << c.mixed_precision << ' ' << c.validation_samples << ' ' << c.infer_game_groups << ' ' << c.early_stop_patience;
    // Search/replay settings have not been used by a supervised-only checkpoint.
    // Establish their contract when the first self-play batch is saved.
    if (selfplay_started) out << ' ' << c.epochs << ' ' << c.buffer_size << ' ' << c.mix_ratio
        << ' ' << c.lr_warmup_batches << ' ' << s.n_playout << ' ' << s.batch_games << ' ' << s.c_puct
        << ' ' << s.temp_high << ' ' << s.temp_low << ' ' << s.temp_switch_move
        << ' ' << s.dirichlet_eps << ' ' << s.dirichlet_alpha << ' ' << s.max_candidates
        << ' ' << s.virtual_loss << ' ' << s.unseen_move_smooth;
    for (const auto& option : {std::pair<const char*, const char*>{"GOMOK_TF32", "1"},
                               {"GOMOK_INFERENCE_TF32", "1"}, {"GOMOK_FUSED_ADAM", "1"}}) {
        const char* value = std::getenv(option.first);
        out << ' ' << option.first << '=' << (value ? value : option.second);
    }
    if (selfplay_started && include_tree_reuse) {
        const char* value = std::getenv("GOMOK_TREE_REUSE");
        out << " GOMOK_TREE_REUSE="
            << ((value == nullptr || std::strcmp(value, "0") != 0) ? 1 : 0);
    }
    if (!c.mix_data.empty()) {
        out << ' ' << std::quoted(std::filesystem::weakly_canonical(c.mix_data).string());
        for (const char* suffix : {"_states.bin", "_probs.bin", "_winners.bin", "_game_ids.bin", "_meta.json"}) {
            const auto path = c.mix_data + suffix;
            if (std::filesystem::exists(path)) out << ' ' << std::filesystem::file_size(path)
                << ':' << std::filesystem::last_write_time(path).time_since_epoch().count();
            else out << " missing";
        }
    }
    return out.str();
}
struct Batch {
    std::vector<float> states, policies, outcomes, weights;
    explicit Batch(int n) : states(static_cast<size_t>(n) * kState),
        policies(static_cast<size_t>(n) * BOARD_CELLS), outcomes(n), weights(n) {}
    void put(int row, const float* state, const float* policy, float z,
             float weight, int symmetry, float smoothing) {
        transform_sample(state, policy, states.data() + static_cast<size_t>(row) * kState,
                         policies.data() + static_cast<size_t>(row) * BOARD_CELLS,
                         symmetry, smoothing);
        outcomes[row] = z;
        weights[row] = weight;
    }
};
}  // namespace

void validate_train_config(const TrainConfig& c, const SearchConfig& s) {
    require(c.buffer_size > 0 && c.batch_size > 0 && c.epochs > 0, "buffer, batch-size and epochs must be positive");
    require(c.game_batch_num >= 0 && c.pretrain_epochs >= 0, "games and pretrain must be nonnegative");
    require(c.game_batch_num > 0 || c.pretrain_epochs > 0, "request at least one training phase");
    require(c.check_freq > 0 && c.validation_freq >= 0 && c.kl_freq >= 0, "invalid diagnostic/checkpoint frequency");
    require(c.validation_samples >= 0 && c.early_stop_patience >= 0, "invalid validation samples or patience");
    require(c.lr_base > 0 && std::isfinite(c.lr_base), "lr must be finite and positive");
    require(c.lr_final_fraction > 0 && c.lr_final_fraction <= 1, "lr-final-fraction must be in (0,1]");
    require(c.lr_warmup_batches >= 0 && c.pretrain_warmup_steps >= 0, "warmup must be nonnegative");
    require(nonnegative(c.weight_decay) && nonnegative(c.grad_clip) && nonnegative(c.value_loss_weight), "invalid optimizer parameter");
    require(nonnegative(c.label_smooth) && c.label_smooth < 1, "label-smooth must be in [0,1)");
    require(nonnegative(c.mix_ratio) && c.mix_ratio <= 1, "mix-ratio must be in [0,1]");
    require(c.mix_value_weight == -1 || (nonnegative(c.mix_value_weight) && c.mix_value_weight <= 1), "mix-value-weight must be in [0,1]");
    require(nonnegative(c.validation_fraction) && c.validation_fraction < 1, "val-fraction must be in [0,1)");
    require(c.pretrain_epochs == 0 || !c.mix_data.empty(), "pretrain requires --mix data");
    require(c.mix_ratio != 1 || !c.mix_data.empty(), "mix-ratio 1 requires --mix data");
    require(c.early_stop_patience == 0 || c.validation_fraction > 0, "early stopping requires validation");
    require(c.eval_games >= 0 && c.eval_playout > 0 && c.eval_freq > 0, "invalid arena settings");
    require(c.eval_games == 0 || !c.baseline_dir.empty(), "eval-games requires --baseline");
    require(c.eval_games % 2 == 0, "eval-games must be even for paired openings");
    require(!c.tag.empty() && c.tag.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos,
            "tag must contain only letters, digits, '_' or '-'");
    validate_search_config(s);
}

double scheduled_learning_rate(double base, double final_fraction,
                               int64_t warmup, int64_t total, int64_t step) {
    total = std::max<int64_t>(1, total);
    warmup = std::min(std::max<int64_t>(0, warmup), total - 1);
    if (step < warmup) return base * (0.1 + 0.9 * static_cast<double>(step + 1) / warmup);
    const double progress = std::clamp(static_cast<double>(step - warmup) /
        std::max<int64_t>(1, total - warmup - 1), 0.0, 1.0);
    return base * (final_fraction + (1 - final_fraction) *
                   0.5 * (1 + std::cos(std::acos(-1.0) * progress)));
}

Trainer::Trainer(PureNet& net, const SearchConfig& search, const TrainConfig& config)
    : net_(net), search_(search), config_(config),
      schedule_batches_(config.game_batch_num), schedule_pretrain_epochs_(config.pretrain_epochs) {
    validate_train_config(config_, search_);
    if (!net_.using_cuda()) throw std::runtime_error("training requires the LibTorch/CUDA backend; native CPU supports inference only");
    buffer_.reserve(config_.buffer_size);
    net_.set_mixed_precision(config_.mixed_precision);
    net_.set_weight_decay(config_.weight_decay);
    net_.set_grad_clip(config_.grad_clip);
    restore_trainer_state();
    load_mix_data();
}

void Trainer::load_mix_data() {
    if (config_.mix_data.empty()) return;
    data_.load(config_.mix_data, config_.validation_fraction, config_.seed, config_.infer_game_groups);
    mix_value_weight_ = config_.mix_value_weight >= 0 ? config_.mix_value_weight : data_.value_weight();
    validation_probe_ = data_.validation_indices();
    training_probe_ = data_.train_indices();
    std::mt19937_64 rng(stream_seed(config_.seed, 1, 0));
    std::shuffle(validation_probe_.begin(), validation_probe_.end(), rng);
    std::shuffle(training_probe_.begin(), training_probe_.end(), rng);
    if (config_.validation_samples > 0 && validation_probe_.size() > static_cast<size_t>(config_.validation_samples))
        validation_probe_.resize(config_.validation_samples);
    training_probe_.resize(std::min(training_probe_.size(), validation_probe_.size()));
    n_kl_ = config_.kl_freq > 0 ? static_cast<int>(std::min<size_t>(64, data_.train_indices().size())) : 0;
    kl_states_.resize(static_cast<size_t>(n_kl_) * kState);
    for (int i = 0; i < n_kl_; ++i)
        std::memcpy(kl_states_.data() + static_cast<size_t>(i) * kState,
                    data_.state(data_.train_indices()[i]), kState * sizeof(float));
    logger.log("[data] %zu games, %zu train, %zu independent val (%zu D4 duplicates purged); mmap; value_weight=%.3f; seed=%llu",
               data_.groups(), data_.train_indices().size(), data_.validation_indices().size(),
               data_.purged_validation(), mix_value_weight_, static_cast<unsigned long long>(config_.seed));
    if (!data_.metadata_present()) logger.log("[data] no rule metadata: external value supervision defaults to zero; rebuild records to document their rules");
}

std::string Trainer::checkpoint_dir() const { return "best_policy_" + config_.tag + "_bin"; }

void Trainer::restore_trainer_state() {
    if (config_.resume_dir.empty()) return;
    const auto meta_path = std::filesystem::path(config_.resume_dir) / "trainer.meta";
    if (!std::filesystem::exists(meta_path)) {
        logger.log("[train] legacy checkpoint: no replay/schedule state; replay starts empty");
        return;
    }
    std::ifstream meta(meta_path);
    std::string magic;
    int version = 0, saved_batch = 0;
    uint64_t saved_seed = 0;
    size_t count = 0;
    meta >> magic >> version >> saved_batch >> pretrain_completed_ >> schedule_batches_
         >> schedule_pretrain_epochs_ >> saved_seed >> best_validation_loss_ >> stale_epochs_ >> count >> buf_head_;
    if (!meta || magic != "GOMOKU_TRAINER" || version != 2 || saved_batch != net_.resume_batch_index() ||
        pretrain_completed_ < 0 || schedule_batches_ < 0 || schedule_pretrain_epochs_ < 0 ||
        stale_epochs_ < 0 || !std::isfinite(best_validation_loss_) || best_validation_loss_ < 0 ||
        count > static_cast<size_t>(config_.buffer_size) || (count == 0 ? buf_head_ != 0 : buf_head_ >= count))
        throw std::runtime_error("invalid/incompatible trainer.meta (requires version 2) or replay capacity");
    std::string saved_contract;
    meta >> std::quoted(saved_contract);
    if (!meta) throw std::runtime_error("cannot read trainer configuration");
    const bool legacy_contract =
        saved_contract == training_contract(config_, search_, saved_batch > 0, false);
    if (saved_contract != training_contract(config_, search_, saved_batch > 0) &&
        !legacy_contract)
        throw std::runtime_error("resume training configuration or dataset changed; pass the original training/search options and unchanged data");
    if (legacy_contract && saved_batch > 0) {
        const char* reuse_setting = std::getenv("GOMOK_TREE_REUSE");
        logger.log("[train] checkpoint predates subtree reuse; applying GOMOK_TREE_REUSE=%d",
                   reuse_setting && std::strcmp(reuse_setting, "0") == 0 ? 0 : 1);
    }
    if (saved_batch == 0 && config_.game_batch_num > 0)
        schedule_batches_ = config_.game_batch_num;
    if (schedule_pretrain_epochs_ == 0 && pretrain_completed_ == 0)
        schedule_pretrain_epochs_ = config_.pretrain_epochs;
    if (saved_seed != config_.seed) throw std::runtime_error("resume seed disagrees with trainer.meta; pass the original --seed");
    const auto replay_path = std::filesystem::path(config_.resume_dir) / "trainer.replay";
    constexpr size_t floats_per_sample = kState + BOARD_CELLS + 1;
    if (!std::filesystem::exists(replay_path) || std::filesystem::file_size(replay_path) != count * floats_per_sample * sizeof(float))
        throw std::runtime_error("checkpoint replay file is missing or truncated");
    buffer_.resize(count);
    std::ifstream replay(replay_path, std::ios::binary);
    size_t record_index = 0;
    for (auto& e : buffer_) {
        replay.read(reinterpret_cast<char*>(e.state.data()), kState * sizeof(float));
        replay.read(reinterpret_cast<char*>(e.probs.data()), BOARD_CELLS * sizeof(float));
        replay.read(reinterpret_cast<char*>(&e.z), sizeof(float));
        if (!replay) throw std::runtime_error("cannot restore replay record");
        validate_training_record(e.state.data(), e.probs.data(), e.z, record_index++);
    }
    if (!replay) throw std::runtime_error("cannot restore replay buffer");
    logger.log("[train] restored %zu replay samples, %d supervised epochs, schedule horizon %d; extending --games keeps the saved LR floor",
               count, pretrain_completed_, schedule_batches_);
}

void Trainer::save_checkpoint(const std::string& dir, int batch_idx) {
    // Publish only a complete model + optimizer + trainer state. The prior
    // generation remains recoverable if a write or rename fails.
    const std::string staged = dir + ".staging";
    const std::string backup = dir + ".previous";
    if (std::filesystem::exists(staged)) std::filesystem::remove_all(staged);
    std::filesystem::create_directories(staged);
    if (!net_.save_training_state(staged, batch_idx)) throw std::runtime_error("checkpoint save failed: " + staged);
    {
        std::ofstream meta(staged + "/trainer.meta");
        meta << "GOMOKU_TRAINER 2\n" << batch_idx << ' ' << pretrain_completed_ << ' '
             << schedule_batches_ << ' ' << schedule_pretrain_epochs_ << ' ' << config_.seed << '\n'
             << std::setprecision(17) << best_validation_loss_ << ' ' << stale_epochs_ << ' '
             << buffer_.size() << ' ' << buf_head_ << '\n'
             << std::quoted(training_contract(config_, search_, batch_idx > 0)) << '\n';
        meta.flush();
        meta.close();
        if (!meta) throw std::runtime_error("cannot save trainer metadata");
    }
    {
        std::ofstream replay(staged + "/trainer.replay", std::ios::binary);
        for (const auto& e : buffer_) {
            replay.write(reinterpret_cast<const char*>(e.state.data()), kState * sizeof(float));
            replay.write(reinterpret_cast<const char*>(e.probs.data()), BOARD_CELLS * sizeof(float));
            replay.write(reinterpret_cast<const char*>(&e.z), sizeof(float));
        }
        replay.flush();
        replay.close();
        if (!replay) throw std::runtime_error("cannot save replay buffer");
    }
    if (std::filesystem::exists(backup)) std::filesystem::remove_all(backup);
    if (std::filesystem::exists(dir)) std::filesystem::rename(dir, backup);
    try { std::filesystem::rename(staged, dir); }
    catch (...) {
        if (std::filesystem::exists(backup)) std::filesystem::rename(backup, dir);
        throw;
    }
    if (std::filesystem::exists(backup)) std::filesystem::remove_all(backup);
    logger.log("[checkpoint] batch %d, pretrain epoch %d, replay %zu saved: %s", batch_idx, pretrain_completed_, buffer_.size(), dir.c_str());
}

void Trainer::add_samples(const std::vector<std::vector<Exp>>& batches) {
    for (const auto& game : batches) for (const auto& sample : game) {
        if (buffer_.size() < static_cast<size_t>(config_.buffer_size)) buffer_.push_back(sample);
        else { buffer_[buf_head_] = sample; buf_head_ = (buf_head_ + 1) % buffer_.size(); }
    }
}
double Trainer::learning_rate_for_batch(int batch_index) const {
    return scheduled_learning_rate(config_.lr_base, config_.lr_final_fraction,
                                   config_.lr_warmup_batches, schedule_batches_, batch_index);
}

void Trainer::policy_update(int batch_index) {
    const auto start = Clock::now();
    const int n = config_.batch_size;
    const auto& pool = data_.train_indices();
    const int external = pool.empty() ? 0 : static_cast<int>(n * config_.mix_ratio);
    const int self = n - external;
    if (self > 0 && buffer_.empty()) throw std::runtime_error("cannot train without replay samples");
    std::mt19937_64 rng(stream_seed(config_.seed, 2, batch_index));
    std::uniform_int_distribution<int> sym(0, config_.augment ? 7 : 0);
    Batch batch(n);
    const bool probe = n_kl_ > 0 && (batch_index + 1) % config_.kl_freq == 0;
    std::vector<float> probe_value(n_kl_);
    if (probe) {
        kl_before_.resize(static_cast<size_t>(n_kl_) * BOARD_CELLS);
        net_.forward_batch(kl_states_.data(), n_kl_, kl_before_.data(), probe_value.data());
    }
    double loss = 0, ploss = 0, vloss = 0, entropy = 0, norm = 0;
    for (int step = 0; step < config_.epochs; ++step) {
        // Each optimizer update draws new records and D4 transforms. Reusing
        // one minibatch for every update increased fitting to a tiny subset.
        for (int i = 0; i < self; ++i) {
            const auto& e = buffer_[std::uniform_int_distribution<size_t>(0, buffer_.size() - 1)(rng)];
            batch.put(i, e.state.data(), e.probs.data(), e.z, 1, sym(rng), 0);
        }
        for (int i = self; i < n; ++i) {
            const size_t index = pool[std::uniform_int_distribution<size_t>(0, pool.size() - 1)(rng)];
            batch.put(i, data_.state(index), data_.policy(index), data_.outcome(index),
                      mix_value_weight_, sym(rng), config_.label_smooth);
        }
        const auto stats = net_.train_step(batch.states, n, batch.policies, batch.outcomes,
            learning_rate_for_batch(batch_index), config_.value_loss_weight, batch.weights);
        loss += stats.loss; ploss += stats.policy_loss; vloss += stats.value_loss;
        entropy += stats.entropy; norm += stats.grad_norm;
    }
    const double steps = config_.epochs;
    logger.log("[update] batch %d loss:%.5f policy:%.5f value:%.5f entropy:%.5f grad_norm:%.3f lr:%.7f samples/s:%.1f replay_draws:%d external_draws:%d",
        batch_index + 1, loss / steps, ploss / steps, vloss / steps, entropy / steps,
        norm / steps, learning_rate_for_batch(batch_index), n * steps / elapsed(start), self * config_.epochs, external * config_.epochs);
    if (probe) {
        std::vector<float> after(kl_before_.size());
        net_.forward_batch(kl_states_.data(), n_kl_, after.data(), probe_value.data());
        double kl = 0;
        for (size_t i = 0; i < after.size(); ++i)
            kl += std::exp(kl_before_[i]) * (kl_before_[i] - after[i]);
        logger.log("[diagnostic] batch %d true_kl:%.7f", batch_index + 1, kl / n_kl_);
    }
}

Trainer::Metrics Trainer::measure(const std::vector<size_t>& indices) {
    Metrics m;
    const int chunk = std::min(256, config_.batch_size);
    Batch batch(chunk);
    std::vector<float> lp(static_cast<size_t>(chunk) * BOARD_CELLS), values(chunk);
    for (size_t off = 0; off < indices.size(); off += chunk) {
        const int n = static_cast<int>(std::min<size_t>(chunk, indices.size() - off));
        for (int i = 0; i < n; ++i)
            std::memcpy(batch.states.data() + static_cast<size_t>(i) * kState,
                        data_.state(indices[off + i]), kState * sizeof(float));
        net_.forward_batch(batch.states.data(), n, lp.data(), values.data());
        for (int i = 0; i < n; ++i) {
            const size_t index = indices[off + i];
            const float* target = data_.policy(index);
            const float* logits = lp.data() + static_cast<size_t>(i) * BOARD_CELLS;
            for (int c = 0; c < BOARD_CELLS; ++c) {
                m.policy -= target[c] * logits[c];
                m.entropy -= std::exp(logits[c]) * logits[c];
            }
            const double diff = values[i] - data_.outcome(index);
            m.value += diff * diff;
            m.accuracy += std::max_element(logits, logits + BOARD_CELLS) - logits ==
                          std::max_element(target, target + BOARD_CELLS) - target;
            ++m.samples;
        }
    }
    if (m.samples) {
        m.policy /= m.samples; m.value /= m.samples;
        m.accuracy /= m.samples; m.entropy /= m.samples;
    }
    return m;
}

void Trainer::record_metrics(const char* phase, int index, const Metrics& train, const Metrics& val) {
    const auto dir = std::filesystem::path("runs") / config_.tag;
    std::filesystem::create_directories(dir);
    const auto path = dir / "metrics.csv";
    const bool fresh = !std::filesystem::exists(path) ||
        (std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == 0);
    std::ofstream file(path, std::ios::app);
    if (fresh) file << "phase,index,train_samples,val_samples,train_policy,val_policy,train_value,val_value,train_top1,val_top1,train_entropy,val_entropy,value_weight\n";
    file << std::setprecision(9) << phase << ',' << index << ',' << train.samples << ',' << val.samples
         << ',' << train.policy << ',' << val.policy << ',' << train.value << ',' << val.value
         << ',' << train.accuracy << ',' << val.accuracy << ',' << train.entropy << ',' << val.entropy
         << ',' << (mix_value_weight_ > 0 ? config_.value_loss_weight : 0) << '\n';
    file.flush();
    file.close();
    if (!file) throw std::runtime_error("cannot write training metrics");
}
double Trainer::validate(const char* phase, int index) {
    if (validation_probe_.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto train = measure(training_probe_);
    const auto val = measure(validation_probe_);
    const double weight = mix_value_weight_ > 0 ? config_.value_loss_weight : 0;
    logger.log("[val] %s %d train_policy:%.5f val_policy:%.5f gap:%+.5f train_top1:%.4f val_top1:%.4f value_mse:%.5f value_enabled:%d samples:%zu",
        phase, index, train.policy, val.policy, val.policy - train.policy, train.accuracy,
        val.accuracy, val.value, mix_value_weight_ > 0 ? 1 : 0, val.samples);
    record_metrics(phase, index, train, val);
    return val.objective(weight);
}

void Trainer::run_pretrain() {
    if (pretrain_completed_ >= config_.pretrain_epochs) return;
    if (config_.early_stop_patience > 0 && stale_epochs_ >= config_.early_stop_patience) {
        logger.log("[pretrain] saved patience exhausted; supervised phase remains stopped");
        return;
    }
    if (net_.resume_batch_index() > 0) throw std::runtime_error("cannot rerun supervised pretraining after self-play; use --pretrain 0 on this checkpoint");
    const auto& pool = data_.train_indices();
    const int batch_size = config_.batch_size;
    const int64_t steps_per_epoch = (pool.size() + batch_size - 1) / batch_size;
    const int64_t total_steps = steps_per_epoch * std::max(1, schedule_pretrain_epochs_);
    const int starting_epoch = pretrain_completed_;
    const auto pretrain_started = Clock::now();
    logger.log("[pretrain] %zu records; %lld steps/epoch including partial final minibatch; completed:%d target:%d",
               pool.size(), static_cast<long long>(steps_per_epoch), pretrain_completed_, config_.pretrain_epochs);
    for (int epoch = pretrain_completed_; epoch < config_.pretrain_epochs; ++epoch) {
        std::vector<size_t> order = pool;
        std::mt19937_64 rng(stream_seed(config_.seed, 3, epoch));
        std::shuffle(order.begin(), order.end(), rng);
        std::uniform_int_distribution<int> sym(0, config_.augment ? 7 : 0);
        const auto start = Clock::now();
        double sum_loss = 0;
        size_t seen = 0;
        for (size_t off = 0; off < pool.size(); off += batch_size) {
            const int n = static_cast<int>(std::min<size_t>(batch_size, pool.size() - off));
            Batch batch(n);
            for (int i = 0; i < n; ++i) {
                const size_t index = order[off + i];
                batch.put(i, data_.state(index), data_.policy(index), data_.outcome(index),
                          mix_value_weight_, sym(rng), config_.label_smooth);
            }
            const int64_t step = epoch * steps_per_epoch + off / batch_size;
            const double lr = scheduled_learning_rate(config_.lr_base, config_.lr_final_fraction,
                config_.pretrain_warmup_steps, total_steps, step);
            const auto stats = net_.train_step(batch.states, n, batch.policies, batch.outcomes,
                                               lr, config_.value_loss_weight, batch.weights);
            sum_loss += stats.loss * n;
            seen += n;
            if ((off / batch_size + 1) % 100 == 0) {
                const int64_t steps_done =
                    static_cast<int64_t>(epoch - starting_epoch) * steps_per_epoch +
                    off / batch_size + 1;
                const int64_t steps_left =
                    static_cast<int64_t>(config_.pretrain_epochs - starting_epoch) *
                    steps_per_epoch - steps_done;
                const double phase_elapsed = elapsed(pretrain_started);
                const double eta = steps_done > 0
                    ? phase_elapsed / steps_done * std::max<int64_t>(0, steps_left)
                    : 0.0;
                logger.log("[pretrain] epoch %d step %lld/%lld loss:%.5f grad_norm:%.3f lr:%.7f samples/s:%.1f",
                    epoch + 1, static_cast<long long>(off / batch_size + 1), static_cast<long long>(steps_per_epoch),
                    stats.loss, stats.grad_norm, lr, seen / elapsed(start));
                logger.log("[pretrain-progress] epoch %d/%d | elapsed %s | ETA %s",
                    epoch + 1, config_.pretrain_epochs, format_duration(phase_elapsed).c_str(),
                    format_duration(eta).c_str());
            }
        }
        pretrain_completed_ = epoch + 1;
        logger.log("[pretrain] epoch %d samples:%zu loss:%.5f samples/s:%.1f", epoch + 1, seen, sum_loss / seen, seen / elapsed(start));
        const double score = validate("pretrain", epoch + 1);
        if (std::isfinite(score)) {
            if (score < best_validation_loss_ - 1e-5) {
                best_validation_loss_ = score;
                stale_epochs_ = 0;
                save_checkpoint("best_policy_" + config_.tag + "_val_bin", 0);
            } else ++stale_epochs_;
        }
        save_checkpoint(checkpoint_dir(), 0);
        const bool stopped_early =
            config_.early_stop_patience > 0 && stale_epochs_ >= config_.early_stop_patience;
        const int completed_epochs = epoch + 1 - starting_epoch;
        const int remaining_epochs = stopped_early ? 0 : config_.pretrain_epochs - epoch - 1;
        const int64_t steps_done = static_cast<int64_t>(completed_epochs) * steps_per_epoch;
        const double phase_elapsed = elapsed(pretrain_started);
        const double phase_eta = steps_done > 0
            ? phase_elapsed / steps_done * remaining_epochs * steps_per_epoch : 0.0;
        logger.log("[pretrain-progress] epoch %d/%d | elapsed %s | ETA %s%s",
                   epoch + 1, config_.pretrain_epochs, format_duration(phase_elapsed).c_str(),
                   format_duration(phase_eta).c_str(),
                   stopped_early ? " (stopped by validation)" : "");
        if (stopped_early) {
            logger.log("[pretrain] validation failed to improve for %d epochs; stopped. Best supervised model: best_policy_%s_val_bin", stale_epochs_, config_.tag.c_str());
            break;
        }
    }
}

double Trainer::run_arena(int batch_idx) {
    if (config_.eval_games == 0) return 0.0;
    if (!baseline_) throw std::logic_error("arena baseline was not prepared");
    SearchConfig arena = search_;
    arena.n_playout = config_.eval_playout;
    const auto start = Clock::now();
    const auto result = play_match(net_, *baseline_, arena, config_.eval_games);
    const double seconds = elapsed(start);
    logger.log("[arena] batch %d %dW %dL %dD score:%.4f first_player_win_rate:%.3f games:%d pairs:%dW/%dL/%dD seconds:%.1f",
               batch_idx, result.wins, result.losses, result.draws, result.score(),
               result.first_player_win_rate(), result.games, result.pair_wins, result.pair_losses, result.pair_draws, seconds);
    if (result.first_player_win_rate() > 0.95 && result.games >= 8)
        logger.log("[arena] high first-player bias; this small match cannot establish a strength improvement");
    return seconds;
}

void Trainer::run() {
    const int start_batch = net_.resume_batch_index();
    if (start_batch > config_.game_batch_num)
        throw std::runtime_error("--games is an absolute batch target and is below the checkpoint batch");
    if (config_.eval_games > 0) {
        const auto baseline_path = std::filesystem::weakly_canonical(config_.baseline_dir);
        if (baseline_path == std::filesystem::weakly_canonical(checkpoint_dir()) ||
            baseline_path == std::filesystem::weakly_canonical("best_policy_" + config_.tag + "_val_bin"))
            throw std::runtime_error("arena baseline must not be a checkpoint output directory");
        baseline_ = std::make_unique<PureNet>();
        if (!baseline_->load(config_.baseline_dir))
            throw std::runtime_error("cannot load arena baseline before training: " + config_.baseline_dir);
    }
    validate("initial", start_batch);
    run_pretrain();
    BatchedSelfPlay sp(net_, search_);
    sp.set_batch_counter(start_batch);
    double avg_batch_seconds = 0.0;
    double avg_arena_seconds = 0.0;
    const auto selfplay_started = Clock::now();
    const auto update_average = [](double sample, double& average) {
        average = average > 0.0 ? 0.75 * average + 0.25 * sample : sample;
    };
    for (int i = start_batch; i < config_.game_batch_num; ++i) {
        const auto batch_start = Clock::now();
        const auto batches = sp.run_batch();
        const double play_seconds = elapsed(batch_start);
        add_samples(batches);
        size_t count = 0;
        for (const auto& game : batches) count += game.size();
        policy_update(i);
        logger.log("[train] batch:%d/%d new_samples:%zu replay:%zu selfplay_seconds:%.3f update_seconds:%.3f",
                   i + 1, config_.game_batch_num, count, buffer_.size(), play_seconds,
                   elapsed(batch_start) - play_seconds);
        const bool arena_due = config_.eval_games > 0 && (i + 1) % config_.eval_freq == 0;
        if ((i + 1) % config_.check_freq == 0 || arena_due) save_checkpoint(checkpoint_dir(), i + 1);
        if (config_.validation_freq > 0 && (i + 1) % config_.validation_freq == 0) validate("selfplay", i + 1);

        // Measure ordinary batch work separately from arena matches, then
        // forecast future scheduled matches explicitly. This avoids a sharp
        // ETA jump when a 40-game arena is much slower than one training batch.
        update_average(elapsed(batch_start), avg_batch_seconds);
        if (arena_due) update_average(run_arena(i + 1), avg_arena_seconds);
        const int completed = i + 1;
        const int remaining_batches = config_.game_batch_num - completed;
        const int remaining_arenas = config_.eval_games > 0
            ? config_.game_batch_num / config_.eval_freq - completed / config_.eval_freq
            : 0;
        const double eta_seconds = remaining_batches * avg_batch_seconds +
                                   remaining_arenas * avg_arena_seconds;
        const double elapsed_seconds = elapsed(selfplay_started);
        if (remaining_arenas > 0) {
            logger.log("[progress] selfplay batch %d/%d | elapsed %s | ETA %s "
                       "(avg batch %.1fs, includes %d arena matches)",
                       completed, config_.game_batch_num,
                       format_duration(elapsed_seconds).c_str(), format_duration(eta_seconds).c_str(),
                       avg_batch_seconds, remaining_arenas);
        } else {
            logger.log("[progress] selfplay batch %d/%d | elapsed %s | ETA %s | avg batch %.1fs",
                       completed, config_.game_batch_num,
                       format_duration(elapsed_seconds).c_str(), format_duration(eta_seconds).c_str(),
                       avg_batch_seconds);
        }
    }
    if (config_.game_batch_num > start_batch) validate("final", config_.game_batch_num);
    save_checkpoint(checkpoint_dir(), config_.game_batch_num);
}
}  // namespace gomoku
