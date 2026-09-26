#!/usr/bin/env python3
"""Gomoku web server: human (browser) vs ONNX AI.

Runs on ONNX Runtime only (no PyTorch needed). Single-page app, human is
black and AI is white.

Usage: python3 server_onnx.py [--port 8000] [--model model.onnx]
                              [--playout 200] [--host 127.0.0.1]
                              [--device auto|cuda|cpu]
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gomoku import BOARD, Board, ONNXEngine

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
let requestId = 0;

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
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || '请求失败');
  return data;
}

async function newGame() {
  const id = ++requestId;
  busy = true;
  try {
    const data = await api('/api/new');
    if (id !== requestId) return;
    board = data.board;
    draw();
    setStatus('你是黑棋，请落子');
  } catch (error) {
    if (id === requestId) setStatus(error.message);
  } finally {
    if (id === requestId) busy = false;
  }
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
  const id = ++requestId;
  try {
  const data = await api('/api/move', {row: r, col: c});
  if (id !== requestId) return;
  board = data.board;
  draw();
  if (data.winner === 1 || data.winner === 2) {
    setStatus(data.winner === 1 ? '🎉 你赢了！' : 'AI 赢了', data.winner === 1 ? 'win' : 'lose');
  } else if (data.game_over) {
    setStatus('平局');
  } else {
    setStatus('轮到你了');
  }
  } catch (error) {
    if (id === requestId) setStatus(error.message);
  } finally {
    if (id === requestId) busy = false;
  }
});

newGame();
</script>
</body>
</html>
"""


class GomokuServer:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.board = Board()
            self.winner = None
            return {'board': self.state(), 'game_over': False, 'winner': None}

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
            if type(row) is not int or type(col) is not int or not (0 <= row < BOARD and 0 <= col < BOARD):
                raise ValueError('row and col must be integers between 0 and 14')
            move = row * BOARD + col
            if move not in self.board.availables:
                return {'board': self.state(), 'game_over': False, 'winner': None}
            previous = self.board.copy()
            self.board.play(move)
            end, winner = self.board.game_end()
            if end:
                self.winner = winner
                return {'board': self.state(), 'game_over': True, 'winner': winner}
            started = time.monotonic()
            try:
                ai_move = self.engine.get_move(self.board)
                self.board.play(ai_move)
            except Exception:
                # An inference failure must not strand the shared game on the
                # AI's turn or claim game_over with no winner on the next move.
                self.board = previous
                raise
            ar, ac = divmod(int(ai_move), BOARD)
            end, winner = self.board.game_end()
            if end:
                self.winner = winner
            return {'board': self.state(), 'game_over': end,
                    'winner': self.winner,
                    'ai_move': {'row': int(ar), 'col': int(ac)},
                    'think_ms': int((time.monotonic() - started) * 1000)}


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
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 <= length <= 4096:
                raise ValueError('request body too large')
            raw = self.rfile.read(length) if length else b'{}'
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('request must be a JSON object')
        except (ValueError, UnicodeDecodeError) as error:
            self._send(400, {'error': str(error)})
            return
        server: GomokuServer = self.server.gomoku
        if self.path == '/api/move':
            try:
                result = server.human_move(data.get('row'), data.get('col'))
            except ValueError as error:
                self._send(400, {'error': str(error)})
                return
            except Exception as error:
                self.log_error('AI move failed: %s', error)
                self._send(500, {'error': 'AI 暂时无法落子，请重试'})
                return
            self._send(200, result)
        elif self.path == '/api/new':
            self._send(200, server.reset())
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
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto',
                        help='ONNX Runtime provider (cuda requires CUDAExecutionProvider)')
    args = parser.parse_args()

    engine = ONNXEngine(args.model, args.c_puct, args.playout, args.device)
    gomoku = GomokuServer(engine)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.gomoku = gomoku
    print(f'五子棋 AI 服务器已启动: http://localhost:{args.port}')
    print(f'模型: {args.model} | playout: {args.playout} | provider: {engine.provider}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')


if __name__ == '__main__':
    main()
