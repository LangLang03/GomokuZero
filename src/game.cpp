#include "game.h"
#include <cstring>

namespace gomoku {

namespace {

uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t stone_key(int move, int player) {
    return mix64(static_cast<uint64_t>(move) * 2ULL +
                 static_cast<uint64_t>(player - 1));
}

}  // namespace

Board::Board() { reset(); }

void Board::reset() {
    cells_.fill(0);
    move_count_ = 0;
    current_ = 1;
    last_ = -1;
    winner_ = 0;
    position_hash_ = 0;
}

void Board::play(int move) {
    position_hash_ ^= stone_key(move, current_);
    cells_[move] = static_cast<int8_t>(current_);
    history_[move_count_] = static_cast<int16_t>(move);
    ++move_count_;
    last_ = move;
    current_ = (current_ == 1) ? 2 : 1;
    winner_ = compute_winner_from_last();
}

void Board::undo() {
    if (move_count_ == 0) return;
    --move_count_;
    int move = history_[move_count_];
    current_ = (current_ == 1) ? 2 : 1;
    position_hash_ ^= stone_key(move, current_);
    cells_[move] = 0;
    last_ = (move_count_ > 0) ? history_[move_count_ - 1] : -1;
    winner_ = 0;
}

uint64_t Board::position_key() const {
    uint64_t key = position_hash_;
    key ^= mix64(0x1000ULL + static_cast<uint64_t>(last_ + 1));
    key ^= mix64(0x2000ULL + static_cast<uint64_t>(current_));
    return key;
}

std::vector<int> Board::legal_moves() const {
    std::vector<int> moves;
    moves.reserve(BOARD_CELLS - move_count_);
    for (int i = 0; i < BOARD_CELLS; ++i)
        if (cells_[i] == 0) moves.push_back(i);
    return moves;
}

std::vector<int> Board::occupied_cells() const {
    std::vector<int> occ;
    occ.reserve(move_count_);
    for (int i = 0; i < BOARD_CELLS; ++i)
        if (cells_[i] != 0) occ.push_back(i);
    return occ;
}

// directions: horizontal, vertical, diag \, diag /
static const int DR[4] = {0, 1, 1, 1};
static const int DC[4] = {1, 0, 1, -1};

int Board::compute_winner_from_last() const {
    if (last_ < 0) return 0;
    int r = last_ / BOARD_SIZE;
    int c = last_ % BOARD_SIZE;
    int player = cells_[last_];
    if (player == 0) return 0;
    for (int d = 0; d < 4; ++d) {
        int count = 1;
        for (int s = 1; s < N_IN_ROW; ++s) {
            int nr = r + DR[d] * s, nc = c + DC[d] * s;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (cells_[nr * BOARD_SIZE + nc] != player) break;
            ++count;
        }
        for (int s = 1; s < N_IN_ROW; ++s) {
            int nr = r - DR[d] * s, nc = c - DC[d] * s;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (cells_[nr * BOARD_SIZE + nc] != player) break;
            ++count;
        }
        if (count >= N_IN_ROW) return player;
    }
    return 0;
}

void Board::encode_state(float* out) const {
    // channels: [0]=current player stones, [1]=opponent stones,
    //           [2]=last move, [3]=all ones if current player is black
    std::memset(out, 0, 4 * BOARD_CELLS * sizeof(float));
    int opp = (current_ == 1) ? 2 : 1;
    for (int ply = 0; ply < move_count_; ++ply) {
        const int i = history_[ply];
        const int v = cells_[i];
        if (v == current_)
            out[i] = 1.0f;
        else if (v == opp)
            out[BOARD_CELLS + i] = 1.0f;
    }
    if (last_ >= 0)
        out[2 * BOARD_CELLS + last_] = 1.0f;
    if (current_ == 1)
        for (int i = 0; i < BOARD_CELLS; ++i)
            out[3 * BOARD_CELLS + i] = 1.0f;
}

}  // namespace gomoku
