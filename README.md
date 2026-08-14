# Gomoku AlphaZero (C++)

A 15x15 Gomoku (five-in-a-row) engine trained with AlphaZero-style
reinforcement learning. The core is pure C++17 — neural network forward/backward,
Adam, lock-free MCTS and the self-play pipeline are all hand-written with no
framework dependency. An optional LibTorch/CUDA backend accelerates inference
and training when a PyTorch wheel is available.

## Highlights

- **Pure-C++ core**: forward/backward/Adam written from scratch; links only `pthread`.
- **Fast CPU inference**: AVX2 conv kernels (register tiling) + int8 quantization
  using AVX-VNNI dot-product instructions, with runtime CPU dispatch.
- **Lock-free MCTS**: arena-allocated nodes, cache-line-aligned atomics, virtual
  loss, and an evaluation cache (≈30% of playouts skip the network).
- **Batched self-play**: leaf tasks from multiple games are merged into large
  inference batches (`GOMOK_BATCH_WAIT_US` bounds the wait).
- **Optional CUDA**: auto-detected PyTorch backend for batched inference,
  autograd training and TF32 tensor cores; `--cpu` forces the CPU path.
- **Verified numerics**: C++ forward output matches the ONNX export to ~1e-5.

## Layout

```
cpp/
├── src/            # engine: game, network, gradient, selfplay, train, logger
├── tools/          # model export: pt -> .bin, pt -> onnx, npz -> bin/pt
├── tests/          # ctest units + benchmark harnesses
├── ui/             # Python play interfaces: web server + console
└── build/          # cmake output, trained weights, checkpoints
```

## Build

```sh
cd cpp
cmake -B build -DGOMOK_ENABLE_CUDA=OFF   # pure C++, no external deps
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is enabled by default when the Python PyTorch wheel is installed
(`python3 -c "import torch"` must succeed). Disable with `-DGOMOK_ENABLE_CUDA=OFF`.

## Usage

```sh
# Train (79 batches = 64 games x 400 playouts each, CUDA backend)
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init cpp_model_v2 --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8

# Same run on CPU only (--int8 for quantized inference)
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init cpp_model_v2 --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8 --int8 --cpu

# Self-play throughput benchmark
./build/gomoku_train selfplay --model cpp_model_v2 --playout 400 \
    --games 64 --batch 64 --threads 8

# Play against the engine in the terminal
./build/gomoku_train human --model cpp_model_v2 --playout 400

# Web UI (browser, 15x15, PyTorch model)
cd ui && python3 server.py --port 8000

# Console play (8x8, pure numpy, no torch needed)
cd ui && python3 human_play.py
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GOMOK_MCTS_THREADS` | min(8, cores) | MCTS tree worker threads |
| `GOMOK_BATCH_WAIT_US` | 100 | max wait to accumulate an inference batch |
| `GOMOK_LEAF_BATCH` | min(playout,128) | leaves collected per tree before submit |
| `GOMOK_CACHE_ENTRIES` | 32768 | evaluation cache size |
| `GOMOK_PROFILE` | off | print per-batch eval/call/cache stats |
| `GOMOK_SEED` | time | RNG seed |

### Model format

Weights live in a directory of 16 `.bin` files, one per layer
(`conv1_weight.bin`, `act_fc1_bias.bin`, ...). Each file is
`[u32 count][float32 values...]`. The same format is used by the C++ engine
and produced by `tools/export_model.py` from a PyTorch state dict.
`tools/export_onnx.py` exports the same net to a single ONNX file
(input `state` [B,4,15,15]; outputs `log_policy` [B,225], `value` [B,1]).

## Network

```
state [B,4,15,15]                      # stones, opponent, last move, player
  conv3x3 4->32 relu
  conv3x3 32->64 relu
  conv3x3 64->128 relu
policy: conv1x1 128->4 relu, fc 900->225, log_softmax   -> [B,225]
value:  conv1x1 128->2 relu, fc 450->64 relu, fc 64->1, tanh -> [B,1]
```

## Tests

- `gomoku_smoke` — optimized conv vs naive reference, fast/quantized vs slow
  inference consistency, game winner/undo, self-play roundtrip.
- `gomoku_game_test` — board rules: wins in all four directions, undo, state encoding.
- `gomoku_gradient_test` — hand-written backward vs finite differences.

```sh
ctest --test-dir build --output-on-failure
```

## Performance (measured on an 8-core laptop + RTX 5060 Laptop)

| Workload | CUDA | CPU fp32 | CPU int8 |
|---|---|---|---|
| Self-play 64 games x 400 playouts | ~6.6 s | ~35 s | ~27 s |
| Training step B=512 x 5 epochs | 0.08 s | 3.3 s | — |
| Single inference, batch 1 | 0.12 ms | 0.79 ms | 0.79 ms |
| Single inference, batch 512 | 0.008 ms/state | 0.17 ms/state | 0.34 ms/state |

## Data

- `cpp/mix_data*` — 200k mixed training samples (self-play + game records),
  loaded by the trainer via `--mix`.
- `data/` — renju-net style supervised game records (`.npz`).

## License

MIT. The Python play interfaces under `cpp/ui/` derive from
[Junxiao Song's AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
(MIT).
