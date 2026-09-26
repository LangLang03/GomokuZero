"""Shared freestyle board, state encoding and search for Python tools.

The four planes use the same current-player perspective as src/game.cpp.
A child's mean value is from its parent's perspective, matching selection.
"""
import math

import numpy as np

BOARD = 15
N_IN_ROW = 5
CELLS = BOARD * BOARD


class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = {}
        self.current_player = 1
        self.availables = list(range(CELLS))
        self.last_move = -1
        self._winner = 0

    def copy(self):
        board = Board.__new__(Board)
        board.states = self.states.copy()
        board.current_player = self.current_player
        board.availables = self.availables.copy()
        board.last_move = self.last_move
        board._winner = self._winner
        return board

    def play(self, move):
        if isinstance(move, (bool, np.bool_)) or not isinstance(move, (int, np.integer)):
            raise ValueError('move must be an integer cell index')
        move = int(move)
        if self.game_end()[0]:
            raise ValueError('cannot play after game termination')
        if not 0 <= move < CELLS or move in self.states:
            raise ValueError(f'illegal move: {move}')
        self.states[move] = self.current_player
        if makes_five(self, move, self.current_player):
            self._winner = self.current_player
        self.availables.remove(move)
        self.current_player = 3 - self.current_player
        self.last_move = move

    def current_state(self):
        state = np.zeros((4, BOARD, BOARD), dtype=np.float32)
        flat = state.reshape(4, CELLS)
        for move, player in self.states.items():
            flat[0 if player == self.current_player else 1, move] = 1
        if self.last_move >= 0:
            flat[2, self.last_move] = 1
        if self.current_player == 1:
            flat[3].fill(1)
        return state

    def winner(self):
        return self._winner

    def game_end(self):
        winner = self.winner()
        return (True, winner) if winner else (not self.availables, -1)


def makes_five(board, move, player):
    row, col = divmod(move, BOARD)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for direction in (-1, 1):
            for distance in range(1, N_IN_ROW):
                r, c = row + direction * dr * distance, col + direction * dc * distance
                if not (0 <= r < BOARD and 0 <= c < BOARD):
                    break
                if board.states.get(r * BOARD + c) != player:
                    break
                count += 1
        if count >= N_IN_ROW:
            return True
    return False


def nearby_moves(board, radius=2, max_cand=None):
    if not board.states:
        return {CELLS // 2}
    candidates = set()
    for move in board.states:
        row, col = divmod(move, BOARD)
        for r in range(max(0, row-radius), min(BOARD, row+radius+1)):
            for c in range(max(0, col-radius), min(BOARD, col+radius+1)):
                cell = r * BOARD + c
                if cell not in board.states:
                    candidates.add(cell)
    return candidates or set(board.availables)


class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent, self.prior = parent, float(prior)
        self.children = {}
        self.visits, self.value = 0, 0.0

    def expand(self, action_probs):
        for action, probability in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(self, probability)

    def select(self, c_puct):
        # Exploration uses this node's visits, including at the root. Using
        # its parent's visits made the root exploration term constant at 1.
        scale = c_puct * math.sqrt(max(1, self.visits))
        return max(self.children.items(), key=lambda item:
                   item[1].value + scale * item[1].prior / (1 + item[1].visits))

    def update_recursive(self, value):
        node = self
        while node is not None:
            node.visits += 1
            node.value += (value - node.value) / node.visits
            value = -value
            node = node.parent


class MCTS:
    def __init__(self, policy_fn, c_puct=5.0, n_playout=200, max_candidates=60):
        if n_playout < 1 or max_candidates < 1 or not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError('playouts, candidates and c_puct must be positive')
        self.policy_fn, self.c_puct = policy_fn, c_puct
        self.n_playout, self.max_candidates = n_playout, max_candidates
        self.root = MCTSNode()

    def _playout(self, board):
        state, node = board.copy(), self.root
        while node.children:
            action, node = node.select(self.c_puct)
            state.play(action)
        ended, winner = state.game_end()
        if ended:
            value = 0.0 if winner == -1 else (1.0 if winner == state.current_player else -1.0)
        else:
            candidates = sorted(nearby_moves(state))
            wins = [m for m in candidates if makes_five(state, m, state.current_player)]
            blocks = [m for m in candidates if makes_five(state, m, 3-state.current_player)]
            if wins:
                node.expand([(m, 1.0/len(wins)) for m in wins])
                value = 1.0
            else:
                action_probs, value = self.policy_fn(state)
                if not math.isfinite(value):
                    raise ValueError('network returned a non-finite value')
                allowed = set(blocks if len(blocks) == 1 else candidates)
                filtered = [(int(a), float(p)) for a, p in action_probs
                            if a in allowed and math.isfinite(p) and p >= 0]
                filtered.sort(key=lambda item: (-item[1], item[0]))
                filtered = filtered[:self.max_candidates]
                total = sum(p for _, p in filtered)
                if total > 0:
                    node.expand([(a, p/total) for a, p in filtered])
                else:
                    moves = sorted(allowed)[:self.max_candidates]
                    node.expand([(m, 1.0/len(moves)) for m in moves])
        node.update_recursive(-value)

    def get_move(self, board, temp=0.0):
        if board.game_end()[0]:
            raise ValueError('cannot search a finished game')
        if not math.isfinite(temp) or temp < 0:
            raise ValueError('temperature must be finite and nonnegative')
        self.root = MCTSNode()
        for _ in range(self.n_playout):
            self._playout(board)
        actions = list(self.root.children)
        if temp <= 1e-3:
            return max(actions, key=lambda a: (self.root.children[a].visits,
                                              self.root.children[a].prior, -a))
        visits = np.array([self.root.children[a].visits for a in actions], dtype=np.float64)
        if visits.sum() == 0:
            visits = np.array([self.root.children[a].prior for a in actions])
        logits = np.log(np.maximum(visits, 1e-30)) / temp
        probs = np.exp(logits-logits.max())
        return int(np.random.choice(actions, p=probs/probs.sum()))


class ONNXEngine:
    def __init__(self, model_path, c_puct=5, n_playout=200, device='auto'):
        import onnxruntime as ort
        if device not in ('auto', 'cuda', 'cpu'):
            raise ValueError("device must be 'auto', 'cuda' or 'cpu'")
        available = ort.get_available_providers()
        has_cuda = 'CUDAExecutionProvider' in available
        if device == 'cuda' and not has_cuda:
            raise RuntimeError(
                'CUDAExecutionProvider is unavailable; install a CUDA-enabled onnxruntime-gpu '
                'package and run where the NVIDIA device is accessible')
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif has_cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            if device != 'auto' or providers == ['CPUExecutionProvider']:
                raise
            # Auto mode may fall back when CUDA libraries exist but the runtime
            # cannot initialize the current device. Explicit --device cuda never does.
            self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.provider = self.session.get_providers()[0]
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get('rules', 'freestyle') != 'freestyle':
            raise ValueError('ONNX model rules must match freestyle Gomoku')
        if metadata.get('board_size', '15x15') != '15x15':
            raise ValueError('ONNX engine requires a 15x15 board')
        self.c_puct, self.n_playout = c_puct, n_playout

    def policy_value(self, board):
        if board.game_end()[0]:
            raise ValueError('cannot evaluate a finished game')
        state = board.current_state()[None]
        log_policy, value = self.session.run(['log_policy', 'value'], {'state': state})
        legal = board.availables
        logits = np.asarray(log_policy[0, legal], dtype=np.float64)
        if not np.isfinite(logits).all() or not np.isfinite(value).all():
            raise ValueError('network returned non-finite predictions')
        probs = np.exp(logits-logits.max())
        probs /= probs.sum()
        return list(zip(legal, probs)), float(value.reshape(-1)[0])

    def get_move(self, board):
        return MCTS(self.policy_value, self.c_puct, self.n_playout).get_move(board)
