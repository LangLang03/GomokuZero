#pragma once
#include <vector>
#include "game.h"
#include "network.h"

namespace gomoku {

// Lock-free MCTS over a batch of independent games. Search workers claim and
// update tree nodes concurrently; one coordinator combines inference requests.
class BatchedSelfPlay {
public:
    struct Sample {
        std::array<float, 4 * BOARD_CELLS> state{};
        std::array<float, BOARD_CELLS> probs{};
        float z = 0.0f;
    };

    BatchedSelfPlay(PureNet& net, float c_puct, int n_playout,
                    int batch_games, int n_threads = -1);  // inference workers
    ~BatchedSelfPlay() = default;

    // play one batch of games to completion, returns samples per game
    std::vector<std::vector<Sample>> run_batch(float temp);

private:
    PureNet& net_;             // used for self-play inference + training
    float c_puct_;
    int n_playout_;
    int batch_games_;
    int leaf_batch_size_;
    int inference_threads_;
    int mcts_threads_;
};

}  // namespace gomoku
