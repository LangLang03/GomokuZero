#pragma once
#include <array>
#include <vector>
#include <cstdint>

namespace gomoku {

constexpr int BOARD_SIZE = 15;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int N_IN_ROW = 5;

// Board state with compact array storage and O(1) undo (for zero-copy MCTS).
class Board {
public:
    Board();

    // piece at cell: 0 empty, 1 black, 2 white
    inline int at(int r, int c) const { return cells_[r * BOARD_SIZE + c]; }
    inline int at(int idx) const { return cells_[idx]; }

    void reset();
    void play(int move);            // place current player's stone at move
    void undo();                    // revert last play (O(1))

    inline int current_player() const { return current_; }  // 1 or 2
    inline int last_move() const { return last_; }
    inline int move_count() const { return move_count_; }
    inline int move_at(int ply) const { return history_[ply]; }
    inline bool is_empty(int move) const { return cells_[move] == 0; }
    uint64_t position_key() const;

    // full board scan for legal moves (used rarely); hot path uses avail list
    std::vector<int> legal_moves() const;

    // cells with a stone on them (for candidate-move generation)
    std::vector<int> occupied_cells() const;

    // win check from last move only (4 directions), O(1)-ish
    // returns winner (1/2) or 0 if no winner yet
    inline int winner() const { return winner_; }
    bool game_over() const { return winner_ != 0 || move_count_ == BOARD_CELLS; }

    // encode 4-channel state from current player's perspective -> float[4][15][15]
    void encode_state(float* out) const;

private:
    std::array<int8_t, BOARD_CELLS> cells_;
    std::array<int16_t, BOARD_CELLS> history_;  // move stack for undo
    int move_count_;
    int current_;   // player to move
    int last_;      // last move index, -1 if none
    int winner_;    // 0 none, 1/2 winner
    uint64_t position_hash_;  // incremental stone-only Zobrist hash

    int compute_winner_from_last() const;
};

}  // namespace gomoku
