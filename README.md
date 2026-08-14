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
├── src/            # engine: game, network, gradient, selfplay, train, logger
├── tools/          # model export utilities
├── tests/          # ctest units + benchmark harnesses
└── build/          # CMake output (ignored by Git)
```

## Build

```sh
# CUDA build (uses LibTorch from the installed Python wheel)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# Dependency-free CPU build
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
    -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j
```

CUDA is enabled by default when the Python PyTorch wheel is installed
and contains `libtorch_cuda.so`. The program falls back to CPU when no CUDA
device is available. `GOMOK_ENABLE_IPO=ON` is the default for Release builds.

## Usage

```sh
# 79 batches x 64 games = 5,056 games (CUDA by default)
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8

# Same run on CPU; INT8 inference requires AVX-VNNI
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8 --int8 --cpu

# Self-play throughput benchmark
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train selfplay \
    --model best_policy_cpp_bin --playout 400 \
    --games 64 --batch 64 --threads 8

# Play against the engine in the terminal
./build/gomoku_train human --model best_policy_cpp_bin --playout 400
```

### Training options

`train --games` counts **training batches**, not individual games. For example,
`--games 79 --batch-games 64` produces 5,056 games. Passing `--games 5000`
would produce 320,000 games.

| Option | Default | Meaning |
|---|---:|---|
| `--init DIR` | random | initial 16-file model directory |
| `--games N` | 1000 | number of self-play/training batches |
| `--playout N` | 400 | MCTS simulations per move |
| `--batch-games N` | 64 | games played together in each batch |
| `--batch-size N` | 512 | samples in one optimizer step |
| `--epochs N` | 5 | optimizer steps after each self-play batch |
| `--buffer N` | 10000 | replay-buffer capacity |
| `--check-freq N` | 50 | checkpoint interval in batches |
| `--threads N` | auto, max 8 | CPU inference workers |
| `--int8` | off | AVX-VNNI inference; CPU only |
| `--cpu` | off | disable CUDA even when available |
| `--mix PREFIX` | none | optional external training data |
| `--mix-ratio R` | 0.5 | external-data fraction per optimizer batch |
| `--tag NAME` | `cpp` | checkpoint suffix: `best_policy_<tag>_bin` |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GOMOK_MCTS_THREADS` | min(8, available cores) | MCTS tree workers |
| `GOMOK_BATCH_WAIT_US` | 100 | max wait to accumulate an inference batch |
| `GOMOK_LEAF_BATCH` | CUDA: total batch near 128; CPU: min(playout,128) per game | per-game inference queue limit |
| `GOMOK_CACHE_ENTRIES` | 32768 | evaluation cache size |
| `GOMOK_PROFILE` | off | print per-batch eval/call/cache stats |
| `GOMOK_SEED` | clock | fixed self-play RNG seed |

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

## Performance

Measurements below were taken on 2026-08-15. They are wall-clock results, not
synthetic FLOP estimates.

### Test environment

| Component | Version |
|---|---|
| OS | Arch Linux, kernel 7.1.4, x86-64 |
| CPU | Intel Core i7-14650HX, 16 cores / 24 threads |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8,151 MiB |
| Driver | NVIDIA 610.43.03 |
| Compiler | GCC 16.1.1, CMake 4.4.0 |
| PyTorch | 2.12.0+cu130, CUDA runtime 13.0 |
| Build | Release, IPO enabled, 8 MCTS/CPU threads |

### End-to-end training batch

Both backends used the same exported 5,000-game checkpoint and these settings:

```text
GOMOK_SEED=1, GOMOK_PROFILE=1
games=1, batch-games=64, playout=400
batch-size=512, epochs=5, check-freq=10, no mix data
```

Reproduce without overwriting the input checkpoint:

```sh
# CUDA
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 1 --playout 400 --batch-games 64 \
    --batch-size 512 --epochs 5 --check-freq 10 --threads 8 \
    --tag bench_cuda

# CPU
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 1 --playout 400 --batch-games 64 \
    --batch-size 512 --epochs 5 --check-freq 10 --threads 8 \
    --int8 --cpu --tag bench_cpu
```

The reported wall time includes process startup, self-play, five optimizer
steps and the final checkpoint save.

| Backend | Wall time | Network evals | Samples | Inference batches |
|---|---:|---:|---:|---:|
| RTX 5060 CUDA FP32/TF32 | **8.34 s** | 884,039 | 3,001 | 9,053 (avg 97.7, max 128) |
| CPU INT8 self-play + FP32 training | **55.17 s** | 919,553 | 3,087 | 1,237 (avg 743.4) |

CUDA is about 6.6x faster end to end in this run. A two-batch CUDA run took
14.19 s total; its second, warmed-up batch took 5.8 s. Based on those runs,
79 batches (5,056 games) are expected to take roughly **9–11 minutes on CUDA**
or **70–80 minutes on CPU**.

Game length changes the amount of work substantially, even with the same
playout count. Compare `evals`, `samples`, and `moves/game` when evaluating two
versions; total seconds alone can be misleading.

### Kernel benchmarks

| Workload | RTX 5060 CUDA | i7-14650HX CPU (8 threads) |
|---|---:|---:|
| Training, B=512 x 5 steps, warmed up | **0.069 s** | **1.84–1.91 s** |
| Pure forward, B=128 | **169.5k eval/s** | — |
| Pure forward, B=512 | **131.2k eval/s** | — |
| Pure forward, B=768 | **120.7k eval/s** | — |
| Pure forward, B=4096 | **119.2k eval/s** | — |

The small 15x15 network is fastest around CUDA batch 128. Self-play therefore
caps the default total CUDA inference batch near 128; larger batches reduced
throughput on this GPU.

## Data

- `mix_data*` — optional mixed training samples (self-play + game records),
  loaded by the trainer via `--mix`.
- `data/` — renju-net style supervised game records (`.npz`).

## License

MIT.
