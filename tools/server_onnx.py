#!/usr/bin/env python3
"""Gomoku web server: human (browser) vs ONNX AI.

Runs on ONNX Runtime only (no PyTorch needed). Single-page app, human is
black, AI is white with a human-like thinking delay.

Usage: python3 server_onnx.py [--port 8000] [--model model.onnx]
                              [--playout 200] [--host 0.0.0.0]
"""
import argparse
import copy
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

BOARD = 15
N_IN_ROW = 5

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>五子棋 · AI</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1b2430; color: #e8eef5;
         display: flex; flex-direction: column; align-items: center; margin: 0; padding: 16px; }
  h1 { font-size: 20px; margin: 8px 0 4px; }
  #status { font-size: 15px; min-height: 24px; margin: 8px 0; color: #9fb3c8; }
  #status.thinking { color: #ffb84d; }
  #status.win { color: #4ade80; }
  #status.lose { color: #f87171; }
  canvas { background: #c8a56b; border-radius: 6px; box-shadow: 0 8px 24px rgba(0,0,0,.5);
           cursor: pointer; max-width: 92vw; }
  button { margin-top: 14px; padding: 8px 22px; font-size: 14px; border: none; border-radius: 6px;
           background: #3b82f6; color: #fff; cursor: pointer; }
  button:hover { background: #2563eb; }
  #tips { font-size: 12px; color: #6b7f96; margin-top: 10px; text-align: center; }
</style>
</head>
<body>
<h1>五子棋 · AI 对弈</h1>
<div id="status">你是黑棋，请落子</div>
<canvas id="board" width="600" height="600"></canvas>
<button onclick="newGame()">重新开始</button>
<div id="tips">点击棋盘交叉点落子 · AI 思考需要几秒，请耐心等待</div>
<script>
const SIZE = 15, PAD = 30, CELL = (600 - 2 * PAD) / 14;
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
let board = Array.from({length: SIZE}, () => Array(SIZE).fill(0));
let busy = false;

function draw() {
  ctx.clearRect(0, 0, 600, 600);
  ctx.strokeStyle = '#5b4328';
  ctx.lineWidth = 1;
  for (let i = 0; i < SIZE; i++) {
    ctx.beginPath();
    ctx.moveTo(PAD, PAD + i * CELL); ctx.lineTo(600 - PAD, PAD + i * CELL); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(PAD + i * CELL, PAD); ctx.lineTo(PAD + i * CELL, 600 - PAD); ctx.stroke();
  }
  ctx.fillStyle = '#3a2a14';
  for (const [r, c] of [[7,7],[3,3],[11,3],[3,11],[11,11]]) {
    ctx.beginPath(); ctx.arc(PAD + c*CELL, PAD + (14-r)*CELL, 3, 0, 2*Math.PI); ctx.fill();
  }
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      if (!board[r][c]) continue;
      const x = PAD + c * CELL, y = PAD + (14 - r) * CELL;
      ctx.beginPath(); ctx.arc(x, y, CELL * 0.42, 0, 2 * Math.PI);
      ctx.fillStyle = board[r][c] === 1 ? '#111' : '#f5f5f5';
      ctx.fill();
      ctx.strokeStyle = board[r][c] === 1 ? '#444' : '#ccc';
      ctx.lineWidth = 1; ctx.stroke();
    }
  }
}

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = cls || '';
}

async function api(path, body) {
  const res = await fetch(path, {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : '{}'});
  return res.json();
}

async function newGame() {
  busy = true;
  const data = await api('/api/new');
  board = data.board;
  draw();
  setStatus('你是黑棋，请落子');
  busy = false;
}

canvas.addEventListener('click', async (e) => {
  if (busy) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (600 / rect.width);
  const y = (e.clientY - rect.top) * (600 / rect.height);
  const c = Math.round((x - PAD) / CELL);
  const r = 14 - Math.round((y - PAD) / CELL);
  if (r < 0 || r > 14 || c < 0 || c > 14 || board[r][c] !== 0) return;
  busy = true;
  setStatus('AI 思考中...', 'thinking');
  const data = await api('/api/move', {row: r, col: c});
  board = data.board;
  draw();
  if (data.winner) {
    setStatus(data.winner === 1 ? '🎉 你赢了！' : 'AI 赢了', data.winner === 1 ? 'win' : 'lose');
  } else if (data.game_over) {
    setStatus('平局');
  } else {
    setStatus('轮到你了');
  }
  busy = false;
});

newGame();
</script>
</body>
</html>
"""


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
            for mv, pl in self.states.items():
                ch = 0 if pl == self.current_player else 1
                square[ch, mv // BOARD, mv % BOARD] = 1.0
            if self.last_move >= 0:
                square[2, self.last_move // BOARD, self.last_move % BOARD] = 1.0
        if len(self.states) % 2 == 0:
            square[3, :, :] = 1.0
        return square[:, ::-1, :]

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
            node.expand(action_probs)
        node.update_recursive(-leaf_value)

    def get_move(self, board, temp=1e-3):
        for _ in range(self.n_playout):
            self._playout(copy.deepcopy(board))
        acts = list(self.root.children.keys())
        if not acts:
            return random.choice(board.availables)
        visits = np.array([self.root.children[a].visits for a in acts], dtype=np.float64)
        logits = np.log(visits + 1e-10) / temp
        logits -= np.max(logits)
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


class GomokuServer:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.board = Board()
        self.winner = None

    def state(self):
        arr = [[0] * BOARD for _ in range(BOARD)]
        for move, player in self.board.states.items():
            r, c = divmod(move, BOARD)
            arr[r][c] = player
        return arr

    def human_move(self, row, col):
        with self.lock:
            if self.winner or self.board.current_player != 1:
                return {'board': self.state(), 'game_over': True, 'winner': self.winner}
            move = row * BOARD + col
            if move not in self.board.availables:
                return {'board': self.state(), 'game_over': False, 'winner': None}
            self.board.play(move)
            end, winner = self.board.game_end()
            if end:
                self.winner = winner
                return {'board': self.state(), 'game_over': True, 'winner': winner}
            delay = random.uniform(0.8, 2.2)
            time.sleep(delay)
            ai_move = self.engine.get_move(self.board)
            self.board.play(ai_move)
            ar, ac = divmod(int(ai_move), BOARD)
            end, winner = self.board.game_end()
            if end:
                self.winner = winner
            return {'board': self.state(), 'game_over': end,
                    'winner': self.winner,
                    'ai_move': {'row': int(ar), 'col': int(ac)},
                    'think_ms': int(delay * 1000)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype='application/json'):
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, HTML.encode(), 'text/html; charset=utf-8')
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length else b'{}'
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        server: GomokuServer = self.server.gomoku
        if self.path == '/api/move':
            self._send(200, server.human_move(int(data.get('row', -1)),
                                              int(data.get('col', -1))))
        elif self.path == '/api/new':
            server.reset()
            self._send(200, {'board': server.state(), 'game_over': False, 'winner': None})
        else:
            self._send(404, {'error': 'not found'})

    def log_message(self, fmt, *args):
        print('[%s] %s' % (time.strftime('%H:%M:%S'), fmt % args))


def main():
    parser = argparse.ArgumentParser(description='Gomoku AI web server (ONNX)')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--model', default='best_policy_cpp_5000.onnx')
    parser.add_argument('--playout', type=int, default=200)
    parser.add_argument('--c-puct', type=float, default=5.0)
    args = parser.parse_args()

    engine = ONNXEngine(args.model, args.c_puct, args.playout)
    gomoku = GomokuServer(engine)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.gomoku = gomoku
    print(f'五子棋 AI 服务器已启动: http://localhost:{args.port}')
    print(f'模型: {args.model} | playout: {args.playout}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')


if __name__ == '__main__':
    main()
