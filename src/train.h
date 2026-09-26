#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "dataset.h"
#include "network.h"
#include "selfplay.h"

namespace gomoku {

struct TrainConfig {
    int buffer_size = 10000;
    int batch_size = 512;
    int epochs = 5;             // fresh minibatches per self-play batch
    int check_freq = 50;
    int game_batch_num = 1000;  // absolute final self-play batch, also on resume
    double lr_base = 2e-3;
    double lr_final_fraction = 0.1;
    int lr_warmup_batches = 2;
    double weight_decay = 1e-4;
    double grad_clip = 2.0;
    double value_loss_weight = 1.0;
    float label_smooth = 0.1f;
    std::string mix_data;
    float mix_ratio = 0.5f;
    float mix_value_weight = -1.0f;  // negative: dataset metadata (missing => 0)
    double validation_fraction = 0.1;
    bool infer_game_groups = false;
    bool augment = true;
    bool mixed_precision = false;  // opt-in BF16 training; inference remains FP32
    uint64_t seed = 1;
    int pretrain_epochs = 0;  // absolute final supervised epoch, also on resume
    int pretrain_warmup_steps = 20;
    int early_stop_patience = 0;  // supervised epochs; 0 disables early stopping
    int validation_freq = 5;
    int validation_samples = 2048;  // 0 evaluates the whole held-out partition
    int kl_freq = 5;                // 0 disables the observational probe
    int eval_games = 0;
    int eval_freq = 50;
    int eval_playout = 400;
    std::string baseline_dir;
    std::string tag = "cpp";
    std::string resume_dir;
};

// Validate before constructing a network or allocating training buffers.
void validate_train_config(const TrainConfig& config, const SearchConfig& search);
double scheduled_learning_rate(double base, double final_fraction,
                               int64_t warmup, int64_t total, int64_t step);

class Trainer {
public:
    Trainer(PureNet& net, const SearchConfig& search, const TrainConfig& config);
    void run();
    double learning_rate_for_batch(int batch_index) const;
private:
    PureNet& net_;
    SearchConfig search_;
    TrainConfig config_;
    using Exp = BatchedSelfPlay::Sample;
    std::vector<Exp> buffer_;
    size_t buf_head_ = 0;
    TrainingData data_;
    std::unique_ptr<PureNet> baseline_;
    float mix_value_weight_ = 0;
    std::vector<size_t> validation_probe_, training_probe_;
    std::vector<float> kl_states_, kl_before_;
    int n_kl_ = 0;
    int pretrain_completed_ = 0;
    int schedule_batches_ = 0;
    int schedule_pretrain_epochs_ = 0;
    int stale_epochs_ = 0;
    double best_validation_loss_ = std::numeric_limits<double>::max();

    struct Metrics {
        double policy = 0, value = 0, accuracy = 0, entropy = 0;
        size_t samples = 0;
        double objective(double value_weight) const { return policy + value_weight * value; }
    };
    void load_mix_data();
    void restore_trainer_state();
    void save_checkpoint(const std::string& dir, int batch_idx);
    void add_samples(const std::vector<std::vector<Exp>>& batches);
    void policy_update(int batch_index);
    void run_pretrain();
    Metrics measure(const std::vector<size_t>& indices);
    double validate(const char* phase, int index);
    void record_metrics(const char* phase, int index, const Metrics& train, const Metrics& val);
    double run_arena(int batch_idx);
    std::string checkpoint_dir() const;
};
}  // namespace gomoku
