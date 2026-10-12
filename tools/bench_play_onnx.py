#!/usr/bin/env python3
"""AI-vs-AI ONNX player, timed per move to compare with gomoku_play_benchmark.

Search comes from tools/gomoku.py unchanged (MCTS + ONNXEngine), so the C++
driver tests/play_benchmark.cpp and this script run the same tree at the same
positions; only the network backend differs (native C++ vs ONNX Runtime).

Modes
  play (default)   self-play a whole game, timing every move
  --moves a,b,c    replay that exact move list, still timing every move
  --forward-bench  time session.run alone at batch 1/8/32

Output matches the C++ driver's CSV:
  meta,<key>,<value>
  move,<ply>,<player>,<row>,<col>,<ms>,<net_ms>,<forwards>,<chosen>
  summary,<tag>,<moves>,<total_ms>,<mean_ms>,<median_ms>,<p90_ms>,<min_ms>,<max_ms>,<net_ms>,<forwards>,<mismatches>,<winner>
  moves,<tag>,<a>,<b>,...
"""
import argparse
import sys
import time

from gomoku import BOARD, CELLS, Board, ONNXEngine

PLANES = 4
BOARD_CELLS = CELLS


class TimedEngine(ONNXEngine):
    """ONNXEngine plus how much time and how many calls went into session.run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_seconds = 0.0
        self.calls = 0

    def policy_value(self, board):
        start = time.perf_counter()
        try:
            return super().policy_value(board)
        finally:
            self.net_seconds += time.perf_counter() - start
            self.calls += 1


def board_line(board):
    """Same one-line board dump as the C++ driver, rows 14 down to 0."""
    rows = []
    for row in range(BOARD - 1, -1, -1):
        cells = "".join(
            "X" if board.states.get(row * BOARD + col) == 1
            else "O" if board.states.get(row * BOARD + col) == 2
            else "."
            for col in range(BOARD)
        )
        rows.append(f";{row:2d}|{cells}")
    return "".join(rows) + f";last={board.last_move};player={board.current_player}"


def percentiles(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    ordered = sorted(values)
    def pick(fraction):
        index = int(fraction * (len(ordered) - 1) + 0.5)
        return ordered[min(index, len(ordered) - 1)]
    return (sum(ordered) / len(ordered), pick(0.5), pick(0.9), sum(ordered))


def lcg_states(count):
    """Same LCG as the C++ driver, so both benchmark identical tensors."""
    import numpy as np
    state = 20260814
    flat = np.empty(count * PLANES * CELLS, dtype=np.float32)
    for i in range(flat.size):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        flat[i] = (state >> 1) % 1000 / 1000.0
    return flat


def forward_benchmark(engine, tag):
    import numpy as np

    flat = lcg_states(32)
    for batch in (1, 8, 32):
        state = flat[: batch * PLANES * CELLS].reshape(batch, PLANES, BOARD, BOARD)
        for _ in range(3):
            engine.session.run(["log_policy", "value"], {"state": state})
        steps = 30 if batch == 1 else 20
        total = 0.0
        best = float("inf")
        for _ in range(steps):
            start = time.perf_counter()
            engine.session.run(["log_policy", "value"], {"state": state})
            elapsed = time.perf_counter() - start
            total += elapsed
            best = min(best, elapsed)
        print(
            f"forward,{tag},batch={batch},ms_per_call={1000 * total / steps:.4f},"
            f"ms_per_position={1000 * total / steps / batch:.4f},"
            f"positions_per_second={batch * steps / total:.1f},best_ms={1000 * best:.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/256x16.onnx")
    parser.add_argument("--tag", default="py")
    parser.add_argument("--moves", default="", help="replay this full move list, forcing every ply")
    parser.add_argument("--me", default="", help="my (black) moves; the engine answers as white")
    parser.add_argument("--print-board", action="store_true")
    parser.add_argument("--playout", type=int, default=200)
    parser.add_argument("--c-puct", type=float, default=5.0)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--max-moves", type=int, default=BOARD_CELLS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0,
                        help="ORT intra-op threads; 0 keeps the runtime default")
    parser.add_argument("--forward-bench", action="store_true")
    args = parser.parse_args()

    engine = TimedEngine(args.model, args.c_puct, args.playout, args.device)
    if args.threads > 0:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        engine.session = ort.InferenceSession(
            args.model, options, providers=[engine.provider])
    print(f"meta,threads,{args.threads if args.threads else 'runtime-default'}")
    print(f"meta,tag,{args.tag}")
    print(f"meta,backend,{engine.provider}")
    print(f"meta,model,{args.model}")
    print(f"meta,playout,{args.playout}")
    print(f"meta,c_puct,{args.c_puct:g}")
    print(f"meta,max_candidates,{args.max_candidates}", flush=True)

    if args.forward_bench:
        forward_benchmark(engine, args.tag)
        return 0

    forced = [int(part) for part in args.moves.split(",") if part.strip()]
    my_moves = [int(part) for part in args.me.split(",") if part.strip()]
    if forced and my_moves:
        parser.error("--moves and --me are mutually exclusive")
    mine = 0
    board = Board()
    moves, move_ms, net_ms = [], [], []
    mismatches = 0
    total_seconds = 0.0
    forwards = 0

    # Warm up lazy runtime state (graph arenas, kernel selection) so the first
    # timed move is not a cold start. Both drivers do the same.
    if args.warmup > 0:
        import numpy as np
        blank = np.zeros((1, PLANES, BOARD, BOARD), dtype=np.float32)
        for _ in range(args.warmup):
            engine.session.run(["log_policy", "value"], {"state": blank})

    while not board.game_end()[0] and len(moves) < args.max_moves:
        ply = len(moves)
        player = board.current_player
        # --moves pins every ply; --me only pins the black (my) plies and lets
        # the engine answer on white's turn.
        pinned = None
        if forced:
            if ply >= len(forced):
                break
            pinned = forced[ply]
        elif my_moves and player == 1:
            if mine >= len(my_moves):
                break
            pinned = my_moves[mine]
            mine += 1
        net_before = engine.net_seconds
        calls_before = engine.calls
        start = time.perf_counter()
        chosen = engine.get_move(board)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        # Tactically decided positions short-circuit before the network, so the
        # call count is measured, not assumed to be one per playout.
        move_forwards = engine.calls - calls_before
        forwards += move_forwards
        move_net_ms = 1000.0 * (engine.net_seconds - net_before)
        move_ms.append(1000.0 * elapsed)
        net_ms.append(move_net_ms)
        played = chosen if pinned is None else pinned
        if pinned is not None and chosen != played:
            mismatches += 1
        moves.append(played)
        print(
            f"move,{ply},{player},{played // BOARD},{played % BOARD},"
            f"{1000 * elapsed:.3f},{move_net_ms:.3f},{move_forwards},{chosen}",
            flush=True,
        )
        if args.print_board:
            print(f"board,{ply}" + board_line(board), flush=True)
        board.play(played)

    if args.print_board:
        print(f"board,{len(moves)}" + board_line(board), flush=True)
    mean, median, p90, _ = percentiles(move_ms)
    print(
        f"summary,{args.tag},{len(moves)},{1000 * total_seconds:.3f},"
        f"{mean:.3f},{median:.3f},{p90:.3f},"
        f"{min(move_ms) if move_ms else 0.0:.3f},"
        f"{max(move_ms) if move_ms else 0.0:.3f},"
        f"{sum(net_ms):.3f},{forwards},{mismatches},{board.winner()}"
    )
    print("moves," + args.tag + ("," + ",".join(str(m) for m in moves) if moves else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
