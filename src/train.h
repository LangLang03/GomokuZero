#pragma once
#include <string>
#include <vector>
#include "network.h"
#include "selfplay.h"

namespace gomoku {

// Batched self-play and policy updates.
class Trainer {
public:
    Trainer(PureNet& net, int n_playout, int batch_games,
            int n_threads, float c_puct, float temp, int buffer_size,
            int batch_size, int epochs, int check_freq, int game_batch_num,
            const std::string& mix_data, float mix_ratio,
            const std::string& tag = "cpp",
            double lr_multiplier_init = 1.0);

    void run();

private:
    PureNet& net_;
    int n_playout_, batch_games_, n_threads_, buffer_size_, batch_size_, epochs_;
    int check_freq_, game_batch_num_;
    float c_puct_, temp_;
    std::string mix_data_;
    float mix_ratio_;
    std::string tag_;  // checkpoint suffix

    // Replay entry.
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

    double lr_multiplier_ = 1.0;
    const double kl_targ_ = 0.05;

    void load_mix_data();
    void add_samples(const std::vector<std::vector<BatchedSelfPlay::Sample>>& batches);
    void policy_update();
    void evaluate_and_save(int batch_idx);
    void validate(int batch_idx);
};

}  // namespace gomoku
