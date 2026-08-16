#!/usr/bin/env python3
"""Terminal Gomoku: play vs the ONNX policy/value net.

Runs on ONNX Runtime only (no PyTorch needed). Board is 15x15, you are black
(move first), AI is white. Enter moves as "row,col" (0-14), e.g. "7,7".

Usage: python3 play_onnx.py [--model model.onnx] [--playout 200] [--c-puct 5]
"""
import argparse
import copy
import random
import sys

import numpy as np
import onnxruntime as ort

BOARD = 15
N_IN_ROW = 5


class Board:
    def __init__(self):
        self.states = {}
        self.current_player = 1
        self.availables = list(range(BOARD * BOARD))
        self.last_move = -1

    def play(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = 2 if self.current_player == 1 else 1
        self.last_move = move

    def current_state(self):
        square = np.zeros((4, BOARD, BOARD), dtype=np.float32)
        if self.states:
            moves = list(self.states.keys())
            players = list(self.states.values())
            cur = [m for m, p in zip(moves, players) if p == self.current_player]
            opp = [m for m, p in zip(moves, players) if p != self.current_player]
            for m in cur:
                square[0, m // BOARD, m % BOARD] = 1.0
            for m in opp:
                square[1, m // BOARD, m % BOARD] = 1.0
            if self.last_move >= 0:
                square[2, self.last_move // BOARD, self.last_move % BOARD] = 1.0
        if len(self.states) % 2 == 0:
            square[3, :, :] = 1.0
        return square

    def winner(self):
        if self.last_move < 0:
            return 0
        r, c = divmod(self.last_move, BOARD)
        p = self.states.get(self.last_move, -1)
        if p == -1:
            return 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt = 1
            for s in range(1, N_IN_ROW):
                rr, cc = r + dr * s, c + dc * s
                if not (0 <= rr < BOARD and 0 <= cc < BOARD):
                    break
                if self.states.get(rr * BOARD + cc) != p:
                    break
                cnt += 1
            for s in range(1, N_IN_ROW):
                rr, cc = r - dr * s, c - dc * s
                if not (0 <= rr < BOARD and 0 <= cc < BOARD):
                    break
                if self.states.get(rr * BOARD + cc) != p:
                    break
                cnt += 1
            if cnt >= N_IN_ROW:
                return p
        return 0

    def game_end(self):
        win = self.winner()
        if win:
            return True, win
        if not self.availables:
            return True, -1
        return False, -1


def nearby_moves(board, radius=2, max_cand=40):
    """Return legal moves near existing stones (mirrors C++ MCTS pruning)."""
    candidates = set()
    for mv in board.states:
        r, c = divmod(mv, BOARD)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD and 0 <= nc < BOARD:
                    idx = nr * BOARD + nc
                    if idx in board.availables:
                        candidates.add(idx)
    if not candidates:
        candidates.update(board.availables)
    return candidates


def makes_five(board, move, player):
    """Return True if placing `player` at `move` would make five in a row."""
    r, c = divmod(move, BOARD)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for s in range(1, N_IN_ROW):
            rr, cc = r + dr * s, c + dc * s
            if not (0 <= rr < BOARD and 0 <= cc < BOARD):
                break
            if board.states.get(rr * BOARD + cc) != player:
                break
            cnt += 1
        for s in range(1, N_IN_ROW):
            rr, cc = r - dr * s, c - dc * s
            if not (0 <= rr < BOARD and 0 <= cc < BOARD):
                break
            if board.states.get(rr * BOARD + cc) != player:
                break
            cnt += 1
        if cnt >= N_IN_ROW:
            return True
    return False


class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

    def select(self, c_puct):
        parent_visits = self.parent.visits if self.parent else 1
        return max(self.children.items(),
                   key=lambda kv: kv[1].value + c_puct * kv[1].prior *
                   np.sqrt(parent_visits) / (1 + kv[1].visits))

    def update(self, leaf_value):
        self.visits += 1
        self.value += (leaf_value - self.value) / self.visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS:
    def __init__(self, policy_fn, c_puct=5, n_playout=200):
        self.policy_fn = policy_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.root = MCTSNode()

    def _playout(self, board):
        node = self.root
        state = copy.deepcopy(board)
        while node.children:
            action, node = node.select(self.c_puct)
            state.play(action)
        action_probs, leaf_value = self.policy_fn(state)
        end, winner = state.game_end()
        if end:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.current_player else -1.0
        else:
            candidates = nearby_moves(state)
            # Force immediate wins and mandatory blocks.
            current = state.current_player
            opponent = 2 if current == 1 else 1
            forced = None
            for mv in candidates:
                if makes_five(state, mv, current):
                    forced = mv
                    break
            if forced is None:
                threat = None
                multiple = False
                for mv in candidates:
                    if makes_five(state, mv, opponent):
                        if threat is None:
                            threat = mv
                        else:
                            multiple = True
                            break
                if not multiple and threat is not None:
                    forced = threat
            if forced is not None:
                node.expand([(forced, 1.0)])
            else:
                filtered = [(a, p) for a, p in action_probs if a in candidates]
                filtered.sort(key=lambda x: x[1], reverse=True)
                node.expand(filtered[:40] if filtered else action_probs)
        node.update_recursive(-leaf_value)

    def get_move(self, board, temp=1e-3):
        for _ in range(self.n_playout):
            self._playout(copy.deepcopy(board))
        acts = list(self.root.children.keys())
        if not acts:
            return random.choice(board.availables)
        visits = np.array([self.root.children[a].visits for a in acts], dtype=np.float64)
        logits = np.log(visits + 1e-10) / temp
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        move = int(np.random.choice(acts, p=probs))
        self.root = MCTSNode()
        return move


class ONNXEngine:
    def __init__(self, model_path, c_puct=5, n_playout=200):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.c_puct = c_puct
        self.n_playout = n_playout

    def policy_value(self, board):
        state = board.current_state().reshape(1, 4, BOARD, BOARD)
        log_policy, value = self.session.run(None, {"state": state})
        probs = np.exp(log_policy[0])
        legal = board.availables
        return list(zip(legal, probs[legal])), float(value[0][0])

    def get_move(self, board):
        mcts = MCTS(self.policy_value, self.c_puct, self.n_playout)
        return mcts.get_move(board)


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
    args = parser.parse_args()

    engine = ONNXEngine(args.model, args.c_puct, args.playout)
    board = Board()
    print("五子棋 · ONNX AI（你是黑棋 X，AI 白棋 O）")
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
            if move not in board.availables:
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
