#pragma once
#include <array>
#include <vector>
#include "game.h"
#include "network.h"

namespace gomoku {

// Self-play and MCTS behaviour. Kept in one struct so the trainer, the
// benchmark command and the tests stay consistent.
struct SearchConfig {
    float c_puct = 5.0f;
    int n_playout = 400;
    int batch_games = 64;
    int n_threads = -1;  // <0: derive from hardware

    // Temperature schedule: full exploration for the opening, then sharper.
    float temp_high = 1.0f;
    float temp_low = 0.5f;
    int temp_switch_move = 30;

    // Root Dirichlet noise. Without it MCTS follows the raw prior and
    // self-play games collapse onto very few distinct lines.
    float dirichlet_eps = 0.25f;
    float dirichlet_alpha = 0.3f;

    // Node expansion cap. Positions outside this window would otherwise be
    // trained as "never play here"; see unseen_move_smooth.
    int max_candidates = 60;

    // Virtual loss applied to a node while a playout is in flight. Larger
    // values spread concurrent threads over distinct branches instead of
    // having them collide on the same child and retry.
    float virtual_loss = 3.0f;

    // Probability mass reserved for legal moves that MCTS never visited, so
    // the policy target does not assert p = 0 for unexplored legal moves.
    float unseen_move_smooth = 0.05f;

    // Returns the sampling temperature for a given ply.
    float temperature_at(int move_number) const {
        return move_number < temp_switch_move ? temp_high : temp_low;
    }
};

// Batched lock-free MCTS with one inference coordinator.
class BatchedSelfPlay {
public:
    struct Sample {
        std::array<float, 4 * BOARD_CELLS> state{};
        std::array<float, BOARD_CELLS> probs{};
        float z = 0.0f;
    };

    BatchedSelfPlay(PureNet& net, const SearchConfig& config);
    // Backwards-compatible shorthand: no root noise, flat temperature.
    BatchedSelfPlay(PureNet& net, float c_puct, int n_playout,
                    int batch_games, int n_threads = -1);
    ~BatchedSelfPlay() = default;

    // Plays one batch of games. The temperature schedule in the config is
    // applied per move; an explicit non-negative value overrides it.
    std::vector<std::vector<Sample>> run_batch(float temp = -1.0f);

    const SearchConfig& config() const { return config_; }

private:
    PureNet& net_;
    SearchConfig config_;
    float c_puct_;
    int n_playout_;
    int batch_games_;
    int leaf_batch_size_;
    int inference_threads_;
    int mcts_threads_;
};

}  // namespace gomoku
