#pragma once
#include <array>
#include <cstdint>
#include <functional>
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

    // Move-sampling temperature: explore the opening, then play more sharply.
    // Policy training targets always use normalized raw visit counts.
    float temp_high = 1.0f;
    float temp_low = 0.5f;
    int temp_switch_move = 30;

    // Root Dirichlet noise. Without it MCTS follows the raw prior and
    // self-play games collapse onto very few distinct lines.
    float dirichlet_eps = 0.25f;
    float dirichlet_alpha = 0.3f;

    // Expansion cap. Root noise is drawn across all legal moves before
    // pruning. A promoted root also retains previously visited children, so
    // its child count may reach twice this limit.
    int max_candidates = 60;

    // Deterministic opening pairs in the arena; colours swap on the identical
    // position. Zero retains empty-board evaluation.
    int eval_opening_moves = 4;

    // Target size of one CUDA inference call; the scheduler derives the
    // maximum in-flight leaves per game from this and batch_games.
    // Zero uses 32. Tune against the selected network and GPU.
    int inference_batch_target = 0;

    // Virtual loss discourages concurrent selections of an in-flight branch.
    // Bounded in-flight work keeps it from dominating completed search values.
    float virtual_loss = 8.0f;

    // Optional uniform smoothing over legal moves. Zero keeps the standard
    // search target. Forced tactical actions always retain their exact target.
    float unseen_move_smooth = 0.0f;

    // Returns the sampling temperature for a given ply.
    float temperature_at(int move_number) const {
        return move_number < temp_switch_move ? temp_high : temp_low;
    }
};

// Shared by CLI/trainer and every search entry point; throws on invalid or
// unrepresentable search budgets before allocating trees or starting training.
void validate_search_config(const SearchConfig& config);

// Outcome of an arena match between two models.
struct MatchResult {
    int games = 0;
    int wins = 0;     // first model
    int losses = 0;   // second model
    int draws = 0;
    // Colour split, tracked because free-style gomoku gives the first player a
    // winning advantage once the search is deep enough. A match where every
    // game is won by whoever moved first looks like a 50% score no matter how
    // the two models compare, so the score alone is not a strength signal.
    int black_wins = 0;
    int white_wins = 0;
    // One pair uses an identical opening with model colours swapped. Compare
    // its combined candidate score with 1: greater wins, less loses, equal ties.
    // Only complete pairs count when the API receives an odd game count.
    int pair_wins = 0;
    int pair_losses = 0;
    int pair_draws = 0;

    double score() const {
        // Wins count 1, draws 0.5, matching the usual Elo convention.
        return games > 0 ? (wins + 0.5 * draws) / games : 0.0;
    }

    // Fraction of decided games won by the first player.
    double first_player_win_rate() const {
        const int decided = black_wins + white_wins;
        return decided > 0 ? static_cast<double>(black_wins) / decided : 0.0;
    }
};

// Single best move for `board`, using the same MCTS as self-play and the
// arena. Returns -1 when no move is available.
int search_best_move(PureNet& net, const Board& board,
                     const SearchConfig& config);

// Backend-independent single-position search. The evaluator returns policy
// probabilities and a value in [-1, 1] from the leaf's side-to-move perspective.
using PositionEvaluator = std::function<void(
    const Board&, std::array<float, BOARD_CELLS>&, float&)>;
int search_best_move(const Board& board, const SearchConfig& config,
                     const PositionEvaluator& evaluate);

// Inspect the same root visit target used by self-play without a neural backend.
struct SearchResult {
    int move = -1;
    std::array<float, BOARD_CELLS> policy{};
    int completed_playouts = 0;
    int evaluations = 0;
};
SearchResult search_position(const Board& board, const SearchConfig& config,
                             const PositionEvaluator& evaluate);

// Policy/value output uses [batch][BOARD_CELLS] and [batch] layouts. Input
// states use [batch][4][BOARD_SIZE][BOARD_SIZE], matching Board::encode_state.
using BatchPositionEvaluator = std::function<void(const float*, int, float*, float*)>;

// Plays `games` games of `candidate` versus `opponent` on reproducible paired
// openings with swapped colours. Even game counts keep the colour split fair.
// Human-record validation loss alone does not establish playing strength.
MatchResult play_match(PureNet& candidate, PureNet& opponent,
                       const SearchConfig& config, int games);
MatchResult play_match(const BatchPositionEvaluator& candidate,
                       const BatchPositionEvaluator& opponent,
                       const SearchConfig& config, int games);


// Batched lock-free MCTS with one inference coordinator.
class BatchedSelfPlay {
public:
    struct Sample {
        std::array<float, 4 * BOARD_CELLS> state{};
        std::array<float, BOARD_CELLS> probs{};
        float z = 0.0f;
    };

    BatchedSelfPlay(PureNet& net, const SearchConfig& config);
    // Backend-independent path for deterministic concurrency tests and other
    // inference engines. accelerated selects the GPU-sized batch schedule.
    BatchedSelfPlay(const SearchConfig& config, BatchPositionEvaluator evaluate,
                    bool accelerated = false);
    // Backwards-compatible shorthand: no root noise, flat temperature.
    BatchedSelfPlay(PureNet& net, float c_puct, int n_playout,
                    int batch_games, int n_threads = -1);
    ~BatchedSelfPlay() = default;

    // Plays one batch of games. The temperature schedule in the config is
    // applied per move; an explicit non-negative value overrides it.
    std::vector<std::vector<Sample>> run_batch(float temp = -1.0f);

    const SearchConfig& config() const { return config_; }
    // Restore the number of generated batches when resuming a seeded run.
    void set_batch_counter(uint64_t completed_batches) {
        batch_counter_ = completed_batches;
    }

private:
    BatchPositionEvaluator evaluate_;
    bool accelerated_;
    SearchConfig config_;
    float c_puct_;
    int n_playout_;
    int batch_games_;
    int leaf_batch_size_;
    int inference_threads_;
    int mcts_threads_;
    // Incremented per run_batch() call. A fixed GOMOK_SEED must still yield a
    // different stream for each batch, otherwise every batch replays exactly
    // the same self-play games and the replay buffer fills with duplicates.
    uint64_t batch_counter_ = 0;
};

}  // namespace gomoku
