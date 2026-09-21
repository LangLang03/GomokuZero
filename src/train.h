#pragma once
#include <string>
#include <vector>
#include "network.h"
#include "selfplay.h"

namespace gomoku {

// Everything the training loop needs. Defaults keep the historical shapes
// where they still make sense, but the learning-rate feedback hack is gone.
struct TrainConfig {
    int buffer_size = 10000;    // replay capacity
    int batch_size = 512;       // samples per optimisation step
    int epochs = 5;             // optimisation steps per self-play batch
    int check_freq = 50;        // batches between checkpoints
    int game_batch_num = 1000;  // self-play batches in total

    // Learning rate: a deterministic warmup + cosine decay. The previous
    // implementation drove lr from |loss change|, which is not a KL
    // divergence and collapsed the rate to its floor within a few batches.
    double lr_base = 2e-3;
    double lr_final_fraction = 0.1;  // end at lr_base * this
    int lr_warmup_batches = 2;

    double weight_decay = 1e-4;
    double grad_clip = 2.0;
    double value_loss_weight = 1.0;

    // Probability mass blended into one-hot human targets. The RenjuNet
    // policies are hard one-hot (entropy 0.000), which forces the policy to
    // assign ~0 to every other legal move.
    float label_smooth = 0.1f;

    // Optional external data (RenjuNet style <prefix>_{states,probs,winners}.bin).
    std::string mix_data;
    float mix_ratio = 0.5f;

    // Replay samples and human records are augmented with the eight board
    // symmetries on the fly (see policy_update), so no extra storage.
    bool augment = true;

    // Real strength measurement: play `eval_games` games against a baseline
    // checkpoint every `eval_freq` batches. Validation loss on human records
    // only measures agreement with one-hot labels, not playing strength.
    // The arena advances several games in lockstep and merges their leaf
    // evaluations into batched inference calls, measured at 3.05 s/game on the
    // 12-block trunk at 400 playouts (a single game at a time costs 7.85 s).
    // Search budget now matches deployment, so the result reflects real
    // playing strength rather than the quality of the raw policy prior.
    int eval_games = 0;       // 0 disables the arena
    int eval_freq = 50;
    int eval_playout = 400;
    std::string baseline_dir;

    std::string tag = "cpp";
};

// Batched self-play and policy updates.
class Trainer {
public:
    Trainer(PureNet& net, const SearchConfig& search, const TrainConfig& config);

    void run();

    // Learning-rate schedule for a zero-based batch index.
    double learning_rate_for_batch(int batch_index) const;

private:
    PureNet& net_;
    SearchConfig search_;
    TrainConfig config_;

    // Replay entry. May be stored symmetrically transformed.
    struct Exp {
        std::vector<float> state;   // 4*15*15
        std::vector<float> probs;   // 225
        float z;
    };
    std::vector<Exp> buffer_;
    int buf_head_ = 0;

    // Optional float32 training and validation data.
    std::vector<float> mix_states_, mix_probs_, mix_winners_;
    int64_t n_mix_ = 0;
    std::vector<float> val_states_, val_probs_, val_winners_;
    int64_t n_val_ = 0;

    // Fixed probe batch for the (diagnostic) KL estimate. The distributions
    // are captured around the normal optimisation step so the diagnostic has
    // no effect on the training trajectory.
    std::vector<float> kl_states_;
    std::vector<float> kl_before_, kl_after_;
    int n_kl_ = 0;

    void load_mix_data();
    void add_samples(
        const std::vector<std::vector<BatchedSelfPlay::Sample>>& batches);
    void policy_update(int batch_index);
    void update_kl_probe();
    void log_true_kl(int batch_index) const;
    void evaluate_and_save(int batch_idx);
    void run_arena(int batch_idx);
    void validate(int batch_idx);
};

}  // namespace gomoku
