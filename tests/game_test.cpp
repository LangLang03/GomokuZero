// Board rules and state encoding.
#include "game.h"
#include <cstdio>
#include <utility>
#include <vector>

using namespace gomoku;

static int fails = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s\n", msg);                                    \
            ++fails;                                                           \
        }                                                                      \
    } while (0)

static void play(Board& b, int r, int c) { b.play(r * BOARD_SIZE + c); }

int main() {
    {
        Board board;
        const uint64_t empty_key = board.position_key();
        board.play(7 * BOARD_SIZE + 7);
        const uint64_t one_move_key = board.position_key();
        CHECK(one_move_key != empty_key, "position key changes after play");
        board.undo();
        CHECK(board.position_key() == empty_key,
              "position key roundtrip after undo");
    }
    {
        Board b;
        CHECK(b.move_count() == 0, "fresh board move count");
        CHECK(b.current_player() == 1, "black starts");
        CHECK(b.last_move() == -1, "no last move");
        CHECK(b.winner() == 0, "no winner");
        CHECK(!b.game_over(), "not game over");
        CHECK(b.legal_moves().size() == BOARD_CELLS, "all moves legal");
    }

    {
        Board b;
        play(b, 7, 7);
        CHECK(b.current_player() == 2, "white after black");
        play(b, 7, 8);
        CHECK(b.current_player() == 1, "back to black");
        CHECK(b.at(7, 7) == 1 && b.at(7, 8) == 2, "stones placed");
        b.undo();
        CHECK(b.current_player() == 2 && b.move_count() == 1, "undo player");
        CHECK(b.last_move() == 7 * BOARD_SIZE + 7, "undo last move");
        b.undo();
        CHECK(b.move_count() == 0 && b.last_move() == -1, "undo all");
    }

    {   // Horizontal.
        Board b;
        for (int c = 0; c < 4; ++c) {
            play(b, 7, c);
            play(b, 14, c);
        }
        play(b, 7, 4);
        CHECK(b.winner() == 1, "horizontal win");
        CHECK(b.game_over(), "game over on win");
    }

    {   // Vertical.
        Board b;
        for (int r = 1; r <= 4; ++r) {
            play(b, 0, r - 1);
            play(b, r, 7);
        }
        play(b, 0, 4);
        play(b, 5, 7);
        CHECK(b.winner() == 2, "vertical win");
    }

    {   // Main diagonal.
        Board b;
        for (int i = 0; i < 4; ++i) {
            play(b, i, i);
            play(b, 14, 14 - i);
        }
        play(b, 4, 4);
        CHECK(b.winner() == 1, "diag backslash win");
    }

    {   // Anti-diagonal.
        Board b;
        for (int i = 0; i < 4; ++i) {
            play(b, i, 4 - i);
            play(b, 14, i);
        }
        play(b, 4, 0);
        CHECK(b.winner() == 1, "diag slash win");
    }

    {
        Board b;
        for (int c = 0; c < 4; ++c) {
            play(b, 7, c);
            play(b, 14, c);
        }
        CHECK(b.winner() == 0, "four in a row not a win");
    }

    {
        Board b;
        for (int c = 0; c < 4; ++c) {
            play(b, 7, c);
            play(b, 14, c);
        }
        play(b, 7, 4);
        CHECK(b.winner() == 1, "win before undo");
        b.undo();
        CHECK(b.winner() == 0, "undo clears win");
        CHECK(b.move_count() == 8, "undo keeps count");
    }

    {   // Black to move after white at (7,8).
        Board b;
        play(b, 7, 7);
        play(b, 7, 8);
        std::vector<float> s(4 * BOARD_CELLS);
        b.encode_state(s.data());
        CHECK(s[7 * BOARD_SIZE + 7] == 1.0f, "ch0 own stone");
        CHECK(s[BOARD_CELLS + 7 * BOARD_SIZE + 8] == 1.0f, "ch1 opp stone");
        CHECK(s[2 * BOARD_CELLS + 7 * BOARD_SIZE + 8] == 1.0f, "ch2 last move");
        int ones = 0;
        for (int i = 0; i < BOARD_CELLS; ++i) ones += s[3 * BOARD_CELLS + i] == 1.0f;
        CHECK(ones == BOARD_CELLS, "ch3 player marker all ones");
        CHECK(s[0 + 7 * BOARD_SIZE + 8] == 0.0f && s[BOARD_CELLS + 7 * BOARD_SIZE + 7] == 0.0f,
              "no cross-channel leakage");
    }

    {   // Channel ownership follows the side to move.
        Board b;
        play(b, 7, 7);
        play(b, 7, 8);
        play(b, 6, 6);
        std::vector<float> s(4 * BOARD_CELLS);
        b.encode_state(s.data());
        CHECK(s[7 * BOARD_SIZE + 8] == 1.0f, "ch0 now white's stone");
        CHECK(s[BOARD_CELLS + 7 * BOARD_SIZE + 7] == 1.0f, "ch1 now black's stone");
        CHECK(s[2 * BOARD_CELLS + 6 * BOARD_SIZE + 6] == 1.0f, "ch2 last move");
        int ones = 0;
        for (int i = 0; i < BOARD_CELLS; ++i) ones += s[3 * BOARD_CELLS + i] == 1.0f;
        CHECK(ones == 0, "white to move: marker plane zero");
    }

    if (fails == 0) {
        std::printf("game_test passed\n");
        return 0;
    }
    std::printf("game_test: %d failures\n", fails);
    return 1;
}
