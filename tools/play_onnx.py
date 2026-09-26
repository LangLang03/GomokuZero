#!/usr/bin/env python3
"""Terminal Gomoku: play vs the ONNX policy/value net.

Runs on ONNX Runtime only (no PyTorch needed). Board is 15x15, you are black
(move first), AI is white. Enter moves as "row,col" (0-14), e.g. "7,7".

Usage: python3 play_onnx.py [--model model.onnx] [--playout 200] [--c-puct 5]
                            [--device auto|cuda|cpu]
"""
import argparse
from gomoku import BOARD, Board, ONNXEngine


def print_board(board):
    print()
    print("    " + " ".join(f"{c:2d}" for c in range(BOARD)))
    for r in range(BOARD - 1, -1, -1):
        print(f"{r:3d} " + " ".join(
            "X " if board.states.get(r * BOARD + c) == 1 else
            "O " if board.states.get(r * BOARD + c) == 2 else ". " for c in range(BOARD)))


def main():
    parser = argparse.ArgumentParser(description="Play Gomoku vs ONNX AI in terminal")
    parser.add_argument("--model", default="best_policy_cpp_5000.onnx")
    parser.add_argument("--playout", type=int, default=200)
    parser.add_argument("--c-puct", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    engine = ONNXEngine(args.model, args.c_puct, args.playout, args.device)
    board = Board()
    print(f"五子棋 · ONNX AI（你是黑棋 X，AI 白棋 O；provider: {engine.provider}）")
    print("输入格式: 行,列 (0-14)，如 7,7")

    while True:
        print_board(board)
        if board.current_player == 1:
            line = input("你的落子 (行,列): ").strip()
            if line.lower() in ("q", "quit", "exit"):
                break
            try:
                r, c = map(int, line.split(","))
                move = r * BOARD + c
            except (ValueError, IndexError):
                print("无效输入，格式如 7,7")
                continue
            if not (0 <= r < BOARD and 0 <= c < BOARD) or move not in board.availables:
                print("该位置不可用")
                continue
            board.play(move)
        else:
            print("AI 思考中...")
            move = engine.get_move(board)
            r, c = divmod(move, BOARD)
            print(f"AI 落子: {r},{c}")
            board.play(move)

        end, winner = board.game_end()
        if end:
            print_board(board)
            if winner == 1:
                print("你赢了！")
            elif winner == 2:
                print("AI 赢了")
            else:
                print("平局")
            break


if __name__ == "__main__":
    main()
