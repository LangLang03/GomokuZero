# GomokuZero — 15×15 AlphaZero Gomoku in C++17 / CUDA

[![CI](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml/badge.svg)](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A from-scratch AlphaZero-style engine for **15×15 freestyle Gomoku** (five or more in a row wins, no Renju forbidden moves, no swap openings). Board, PUCT/MCTS, batched self-play, dataset pipeline, trainer, checkpointing, arena and CLI are C++17; the neural network runs through LibTorch/CUDA for training and through a native folded-BN CPU path for inference. A Python toolchain prepares the human-game dataset, exports ONNX and serves a browser/terminal opponent.

The starting point was the well-known Python project [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku). On the laptop described below, one comparable 15×15 training campaign took **about 3 days** there and **about 2.5 hours** here — see [Results at a glance](#results-at-a-glance) and [Why it is fast](#why-it-is-fast-the-optimization-list) for what is measured and what is mechanism.

| | |
|---|---|
| Game | 15×15 freestyle Gomoku, five-or-more wins, Black moves first |
| Network | residual trunk, **12 blocks × 192 channels**, 8,100,525 parameters, 51×51 trunk receptive field |
| Input / output | `[B,4,15,15]` planes → `log_policy [B,225]` + `value [B,1]` (tanh) |
| Search | batched PUCT/MCTS, 400 playouts/move default, virtual loss, subtree reuse, tactical exact values |
| Training | supervised pretraining on human games → self-play RL, Adam + cosine schedule, BF16 opt-in |
| Stack | C++17, CMake, LibTorch/CUDA training, native CPU inference, ONNX Runtime deployment |
| Dataset | RenjuNet XML → 4,218 games / 129,588 positions, whole-game split with D4 leakage purge |

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [The device this was tuned on](#the-device-this-was-tuned-on)
- [Quick start](#quick-start)
- [Why it is fast: the optimization list](#why-it-is-fast-the-optimization-list)
- [Network and search](#network-and-search)
- [Training pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Measuring strength](#measuring-strength)
- [Deployment: ONNX, browser and terminal play](#deployment-onnx-browser-and-terminal-play)
- [Repository layout](#repository-layout)
- [Tests and CI](#tests-and-ci)
- [Environment variables](#environment-variables)
- [Where to look in the code](#where-to-look-in-the-code)
- [Scope, provenance and limits](#scope-provenance-and-limits)

---

## Results at a glance

Reference campaign: `tools/train_quality_5000.sh` — 6 supervised epochs from random weights, then **250 self-play batches × 20 games = 5,000 games** with 24 optimizer updates per batch (6,000 updates), BF16, 256 playouts/move, replay buffer 20,000, 35 % external human-game samples per batch, plus five 80-game arenas.

| | Reference: junxiaosong/AlphaZero_Gomoku (PyTorch) | This repository (C++17 + LibTorch/CUDA) |
|---|---:|---:|
| Comparable 15×15 campaign, same laptop | **≈ 3 days** | **≈ 2.5 hours** |
| Speed-up | — | **≈ 29×** |

Measured pieces of that 2.5 hours (this machine, `build-training/`):

| Stage | Measured | Basis |
|---|---:|---|
| Supervised pretraining, 6 epochs (116,781 positions × 6, 5,478 updates, full 12,188-position validation) | ≈ 10 min | stage 1 of `tools/train_quality_5000.sh`, local run (18:19 → 18:29 by artifact timestamps) |
| Self-play RL, 250 batches / 5,000 games / 6,000 updates | ≈ 2 h 21 min | stage 2 of the same run (18:29:00 → 20:49:49) |
| Whole pipeline | ≈ 2 h 31 min | the two stages above; method in [Scope, provenance and limits](#scope-provenance-and-limits) |

Supporting measurements, same machine and same 192×12 network:

| Measurement | Result |
|---|---|
| Backend microbenchmark, batch 128, 8 steps after 3 warm-ups | TF32: inference 41.9 ms (3,053 samples/s), training 152.0 ms (842 samples/s) |
| Same, BF16 training | 76.6 ms/batch (**1,670 samples/s**) — **1.98×** vs TF32, **2.14×** vs commit `35f5872` |
| Same, strict FP32 (TF32 off) | inference 172.8 ms (741 samples/s), training 354.6 ms (361 samples/s) — TF32 is worth **4.1× / 2.3×** |
| Supervised stage, 3 epochs, identical seed/data/updates | FP32 460.4 s → BF16 260.5 s (**1.77×**); final validation policy CE 2.0105, top-1 44.81 % |
| Self-play throughput, production model, 256 playouts/move, tree reuse on | 8 games in 6.13 s = **0.77 s/game**, 56 ms/move, inference batch avg 15.6 / max 32, cache hit 13.4 % |
| Arena, 80 games, 400 playouts/move, 4-move paired openings | `quality5000` (batch 250) **58 W / 19 L / 3 D** vs `fromzero` (batch 150), score 0.744, pairs 21 W / 1 L / 18 D |
| Value head on 64 fresh self-play games, D4-purged | game-averaged MSE **0.1359** vs 0.2877 for the optimistic colour-only constant |
| C++ test suite, production build | `ctest` **11/11 in 35.3 s** |
| Dataset load + validation + whole-game split | 129,588 positions in **0.394 s**, peak RSS 570 MiB |

Every number above is tied to a command or artifact; nothing here is an Elo or a claim about all opponents. The older 40 W / 0 L arena against the batch-170 model, with its protocol, is in [Measuring strength](#measuring-strength).

---

## The device this was tuned on

All timings in this README were produced on this single laptop, with the exact baseline below. Treat them as one machine's evidence, not as a hardware-independent promise.

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB, compute capability 12.0 (sm_120), driver 610.57.04, CUDA UMD 13.3 |
| CPU / RAM | Intel Core i7-14650HX (24 logical cores), 15 GiB RAM |
| OS / kernel | Arch Linux, kernel 7.1.9-arch1-2, glibc 2.44 |
| Compiler / build | GCC 16.2.1, CMake 4.4.2, `-DCMAKE_BUILD_TYPE=Release`, C++17 |
| C++ trainer backend | LibTorch **2.12.0+cu130** (CUDA 13.0, cuDNN 9.24) discovered by CMake through `/usr/bin/python3.14` |
| Python tools | Python 3.14 in `.venv/`: torch 2.14.0+cu130, numpy 2.5.3, onnx 1.23.0 |
| Deployment | onnxruntime-gpu 1.28.0 (CUDA 13 / cuDNN 9), CUDA libraries from the pip `nvidia-*` wheels |

`nvcc` is **not** required: the CUDA backend is LibTorch, so CMake only needs a Python interpreter that can `import torch` and a `libtorch_cuda.so`. Pin the interpreter with `-DPython3_EXECUTABLE=...` if several environments are installed (this machine has both a system torch 2.12 and a venv torch 2.14).

---

## Quick start

### 1. Build and test

```sh
# Prerequisites: CMake >= 3.18, a C++17 compiler, Python 3 with a CUDA-enabled torch
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'   # e.g. 2.12.0+cu130 13.0

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

A CUDA-free build is supported for inference, search and the test suite; **training is refused without the LibTorch/CUDA backend**:

```sh
cmake -S . -B build-cpu -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j2 && ctest --test-dir build-cpu --output-on-failure
```

### 2. Play, watch self-play, or run a match

```sh
./build/gomoku_train human --model best_policy_quality5000_20260926_bin --playout 400   # type r,c (0-14)

GOMOK_PROFILE=1 ./build/gomoku_train selfplay \
  --model best_policy_quality5000_20260926_bin --games 8 --batch 8 --playout 256 --seed 1

./build/gomoku_train arena --model best_policy_quality5000_20260926_bin \
  --baseline best_policy_fromzero_20260926_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
```

Checkpoints are local artifacts (`best_policy_*_bin/`) and are not in version control; train one first, or export random weights with `./build/gomoku_train export --output my_model_bin`.

### 3. Prepare data and train

```sh
python3 tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python3 tools/audit_dataset.py data/gomoku_v2 --state-overlap   # audit JSON on stdout

# Supervised pretraining (absolute target: 6 epochs; --games 0 means supervised only)
./build/gomoku_train train --mix data/gomoku_v2 --pretrain 6 --games 0 \
  --batch-size 128 --precision bf16 --seed 1 --val-samples 0 --tag pretrain

# Self-play RL from the best validation checkpoint (absolute target: batch 250)
./build/gomoku_train train --init best_policy_pretrain_val_bin --mix data/gomoku_v2 \
  --pretrain 0 --games 250 --batch-games 20 --batch-size 128 --precision bf16 \
  --playout 256 --epochs 24 --buffer 20000 --mix-ratio 0.35 --seed 1 \
  --val-samples 0 --check-freq 25 --tag rl
```

The reference campaign with arenas, environment pinning and the 5,000-game budget is [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh); it refuses to overwrite existing checkpoints. `--games` and `--pretrain` are **absolute targets**, not increments — resuming batch 100 with `--games 250` generates 150 more batches.

### 4. Export and serve

```sh
./build/gomoku_train export --model best_policy_quality5000_20260926_bin --output model_export
python3 tools/export_onnx.py --bin-dir model_export model.onnx

python3 -m pip install 'onnxruntime-gpu[cuda,cudnn]'
python3 tools/server_onnx.py --model model.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000        # open http://127.0.0.1:8000
```

---

## Why it is fast: the optimization list

The reference implementation is a single-process PyTorch trainer that plays one game at a time, evaluates one leaf per forward call, runs FP32 and rebuilds the search tree every move. Everything below replaces one of those behaviours. "Baseline" always means that reference behaviour, and each row carries its evidence.

### A. Self-play parallelism and search scheduling

| # | Optimization | Baseline | Here | Evidence / impact |
|---|---|---|---|---|
| A1 | **Batched self-play over concurrent games** — one call advances `--batch-games` games (default 64) | one game per iteration | 64 games in flight, 64 independent trees | `src/selfplay.cpp` `BatchedSelfPlay::run_batch`; production profile shows inference batch avg 7–15, max 32 |
| A2 | **Lock-free MCTS worker pool** with a condition-variable round launcher | single-threaded search per game | `min(8, physical cores)` workers, clamped 1–32 | `src/selfplay.cpp` `SearchWorkerPool`; `GOMOK_MCTS_THREADS`, `--threads` |
| A3 | **CPU affinity that reserves the fastest cores for the GPU coordinator** | no pinning; searchers and inference fight for cores | physical cores sorted by max frequency; the fastest `inference_threads` cores are skipped | `src/selfplay.cpp` (topology + `cpufreq` read) |
| A4 | **Batched leaf evaluation with a collection window** — spins up to `GOMOK_BATCH_WAIT_US` (100 µs on CUDA) and submits one GPU call | one forward per leaf, GPU idle between calls | target inference batch 32; leaves from all games are merged | batch sweep 16…512 → **32 is optimal** (2,846 eval/s vs 2,588 at 128); 400-playout self-play **7.30 s → 6.26 s per game** |
| A5 | **Fixed-point virtual loss** applied to the in-flight path | none — threads pile onto the same child | integer atomic visits + Q20 value sum, default 8, disabled when the leaf batch is 1 | sweep: retries −32 %, **6 % faster per game** vs the old default 3 |
| A6 | **Collision retries do not consume the search budget** | retry costs a playout | `NODE_EXPANDING` contention returns `Retry` without decrementing issued playouts | `src/selfplay.cpp` retry loop; playout cap enforced by CAS |
| A7 | **Subtree reuse between moves** (`GOMOK_TREE_REUSE=1`, default on) | `reset_tree()` every move, search 400 playouts from scratch | the chosen child is promoted (ping-pong node arenas), retained visits count toward the next budget | CPU A/B: 8 games **9.40 s → 6.44 s**; fixed-line control: search loop 0.040 s → 0.012 s. Not yet measured on the production CUDA path |
| A8 | **Cross-move evaluation cache** — Zobrist position key, FIFO ring, 32,768 entries, shared by all 64 games in a batch | every leaf re-evaluated | cache hit rate 0.27–0.41 in production self-play; also de-duplicates identical leaves inside one batch | `GOMOK_CACHE_ENTRIES`; profile line `cache_hits`/`cache_hit_rate` |
| A9 | **Tactical and terminal short-circuits** | net consulted for every leaf | immediate five → exact +1 and no inference; forced block collapses expansion to one child; finished games and solved nodes never hit the net | `src/selfplay.cpp`; comment: exact wins must override an untrained value head |
| A10 | **Preallocated node arenas and scratch buffers** (32-byte aligned nodes, no allocation during search) | per-node Python objects | one arena per game per batch, worker scratch reused across moves | `alignas(32) GNode`, `SearchWorkerScratch`, buffers sized once per `run_batch()` |
| A11 | **Batched arena matches** | one game at a time | up to 32 games in lockstep, exactly two batched calls per round (one per model) | `play_match`; the 80-game arena above finished in ~2 min at 400 playouts |

### B. Precision and GPU kernels

| # | Optimization | Baseline | Here | Evidence / impact |
|---|---|---|---|---|
| B1 | **TF32 for cuDNN and cuBLAS**, scoped by an RAII guard, on by default | FP32 kernels | `GOMOK_TF32` / `GOMOK_INFERENCE_TF32` (both 1) | strict FP32 → TF32: inference 172.8 → 41.9 ms (**4.1×**), training 354.6 → 152.0 ms (**2.3×**) |
| B2 | **BF16 training autocast with FP32 master weights, loss and Adam moments**; the two final head nonlinearities stay FP32 | FP32 only | `--precision bf16` | training 152.0 → 76.6 ms (**1.98×** vs TF32, 2.14× vs HEAD); end-to-end supervised stage 460.4 s → 260.5 s (**1.77×**) with equal policy quality in the same-seed 3-epoch experiment |
| B3 | **Fused optimizer and fused gradient math** | per-parameter Python loop | `at::_fused_adam_`, `_foreach_norm` for the global norm, `_foreach_mul_` for clipping, five scalars read back in one sync | `GOMOK_FUSED_ADAM=1`; its own A/B measured only ~0.5–1 % (kept for the reduced launch count) |
| B4 | **Batch norm folded into the convolutions, once per weight generation** | BN kernels on every forward | inference graphs contain conv + bias + ReLU only | `src/network_cuda.cpp` `fold_bn`; thousands of self-play forwards run without a single BN kernel |
| B5 | **`torch::InferenceMode` + `eval()` + channels-last conv weights/activations** | autograd-tracked FP32 NCHW | no graph, no version counters, tensor-core-friendly layout | `src/network_cuda.cpp` |
| B6 | **cuDNN autotune deliberately disabled** | `cudnn.benchmark = True` | autotuning is a net loss because MCTS batch sizes vary (`GOMOK_DETERMINISTIC=1` gives deterministic cuDNN instead) | comment in `src/network_cuda.cpp` |
| B7 | **Redundant conv biases removed from the optimizer** | every parameter updated | conv biases are frozen (BN makes them redundant; their rounding noise was being amplified by Adam), which also shrinks Adam state | `requires_grad(false)` on conv biases |
| B8 | **The unsupervised value branch is not even in the graph** | `policy + 0 * value` still builds value gradients that Adam's weight decay destroys | when value supervision is off the value term is absent, its parameters keep their own Adam step counter | fixes a measured collapse in which the value head's dense weights had decayed to ~1e-37 under the old policy-only objective |
| B9 | **Native CPU inference path** (folded BN, 8-lane unrolled convolution, thread-local scratch, sample-parallel batches) | — | keeps MCTS, the CLI and the tests usable without CUDA; CPU training is refused, not silently degraded | `src/network.cpp`, `src/network_cuda_stub.cpp` |

### C. Data, I/O and checkpoints

| # | Optimization | Baseline | Here | Evidence / impact |
|---|---|---|---|---|
| C1 | **Memory-mapped dataset with index-only splits** | load everything into RAM as tensors | `mmap(PROT_READ, MAP_PRIVATE)` + per-record validation; train/validation are row indices, never a copy | 129,588 positions validated and split in **0.394 s**, peak RSS 570 MiB |
| C2 | **On-the-fly D4 augmentation** (8 symmetries, same transform for state and policy) | pre-expanded or no augmentation | every optimizer step re-draws samples **and** their symmetry | 8× sample efficiency at 1× storage; `--augment 1` |
| C3 | **Fresh minibatch every optimizer step** | one minibatch replayed for many updates | new replay and external draws per step, logged as `replay_draws`/`external_draws` | avoids over-fitting a handful of records (and makes `--epochs` mean real updates) |
| C4 | **Streaming, validating data tools** | in-memory numpy pipelines | per-member mmap in `npz_to_bin.py`, per-game validation and D4 de-duplication in `parse_renjunet.py`, temp-file + rename writes | 159,000 XML records → 4,218 accepted games with a rejection ledger |
| C5 | **Whole-game split with a D4 leakage purge** | random sample split | game IDs hashed with `splitmix64(id ^ seed)`, then held-out positions D4-equivalent to training are dropped; the identical algorithm exists in C++ and Python with a golden cross-language test | a random split would leak ≈ 88.9 % of games; the game split leaves 619/12,807 = 4.83 % D4 overlap, purged → 12,188 independent validation positions |
| C6 | **Atomic checkpoints and a resume contract** | save/load weights only | `.staging` → rename with `.previous` rollback; network + BN + Adam + replay + batch index + schedule horizons + seed + option/dataset fingerprint, validated before use | a completed run cannot silently restart from scratch; `--games`/`--pretrain` stay absolute across resumes |
| C7 | **Honest ETA and separated timings** | wall-clock per epoch | self-play and update seconds measured separately, smoothed over recent batches, scheduled arenas added separately | `[train]`, `[progress]` log lines |

### D. Engineering

| # | Optimization | Why |
|---|---|---|
| D1 | One shape source of truth (`src/net_config.h`) for trunk width, block count, head sizes, reachable from C++ and Python | the shape used to be duplicated in four places, so the network, the exporter and the tests could disagree |
| D2 | `gomoku_core` built once as a static library; executables and tests link it | avoids recompiling the expensive LibTorch translation unit per target |
| D3 | LTO on by default, with the LibTorch TU compiled `-O1 -fno-lto` | keeps release optimizations while bounding compile time and memory |
| D4 | CI builds the CPU engine with ASan + UBSan on a shrunken trunk, plus a production-shape compile job | concurrency and memory bugs are caught without paying for the full 12×192 network in every job |
| D5 | Strict CLI parsing (unknown/duplicate/missing-value options are fatal) and a failing model load is never silently replaced by random weights | removes the "trained the wrong experiment" failure mode |

### What did **not** pay off (kept, off by default)

| Attempt | Result |
|---|---|
| `GOMOK_PINNED_INFERENCE=1` — pinned host staging, non-blocking copies | no gain on the RTX 5060; the inference API still synchronizes per batch, so the copy cannot overlap. Default 0 |
| `GOMOK_PERSISTENT_WORKERS=1` — keep searcher threads and scratch alive across moves | no gain on the RTX 5060. Default 0 |
| Fused Adam | 0.5–1 % in its own A/B, inside noise; kept only for the reduced launch count |
| Subtree reuse on the production CUDA path | measured only on the small CPU model so far (31 % shorter games, −2.9 % evaluations per move); the CUDA A/B is not done |
| cuDNN autotune | deliberately off (see B6) |
| Native CPU training | refused by design: `train --cpu` exits with an error instead of pretending |

**Attribution of the ≈29×.** The measured, isolatable parts are: precision (TF32/BF16 ≈ 2–4× on kernels, 1.77× end-to-end for the supervised stage), batched GPU evaluation (32 leaves per call instead of one; the batch sweep shows the GPU is latency-bound, not throughput-bound, at batch 1), subtree reuse and the shared evaluation cache (fewer network calls per move), tactical short-circuits (no inference on decided positions), and removing Python/GIL overhead from the hot path (C++ single process). Only the precision factors and the batch-size sweep were measured in isolation; the rest is mechanism plus the end-to-end 2.5-hour result.

---

## Network and search

### State encoding

Four planes in the current player's perspective, matching `Board::encode_state` and the Python tools: own stones, opponent stones, last move, and "Black to move" (constant plane). This orientation means the network never sees absolute colours.

### Network

```
state [4,15,15]
  stem      conv3x3 4 -> 192, BN, ReLU
  trunk     12 x [ conv3x3 192->192, BN, ReLU, conv3x3 192->192, BN, add, ReLU ]   (51x51 receptive field)
  policy    conv1x1 192->2, BN, ReLU, flatten(450), fc 450->225, log_softmax
  value     conv1x1 192->1, BN, ReLU, flatten(225), fc 225->64, ReLU, fc 64->1, tanh
```

| Part | Parameters |
|---|---:|
| stem 3×3 4→192 | 7,104 |
| trunk, 12 blocks (98.5 % of the model) | 7,967,232 |
| policy head (1×1 conv + 450→225 dense) | 101,861 |
| value head (1×1 conv + 225→64 + 64→1) | 14,722 |
| **Total (inference export, BN folded)** | **8,090,919** |
| Training checkpoint learned parameters (adds BN γ/β) | 8,100,525 |

Rebuild other shapes with `-DGOMOK_NET_CHANNELS=N -DGOMOK_NET_RES_BLOCKS=N`; checkpoints from different shapes are incompatible. The dense policy head is 450→225 (weight sharing across cells after a 1×1 convolution) rather than the old 900→225 layer that consumed 62 % of all parameters with no cross-position sharing.

### Search

- PUCT/MCTS with `c_puct 5`, `--playout 400` (training default 256 in the reference campaign), candidate cap 60.
- Leaves are evaluated in their own side-to-move perspective; value signs alternate during backup; terminal positions are exact.
- Immediate wins return exact values without inference; a forced block survives pruning and collapses expansion to one child.
- Virtual loss applies only when several leaves of the same tree are in flight (it is disabled for the serialized arena search).
- Root Dirichlet noise (`ε 0.25`, `α 0.3`) is drawn over **all** legal moves before pruning; self-play policy targets use normalized raw visit counts, temperature controls sampling only.
- Subtree reuse keeps completed visits and values for the next move; root noise is re-drawn, previously visited children are kept and current top candidates are added.
- Arena play uses identical openings with colours swapped and reports per-game W/L/D plus pair results, because freestyle Gomoku's first-player advantage makes raw scores misleading.

---

## Training pipeline

Two stages in one `train` command:

1. **Supervised pretraining** on human games — policy cross-entropy only by default. The RenjuNet sources use different opening/overline rules, so their recorded outcomes must not become freestyle value labels: the dataset metadata sets `value_weight = 0`, and the value branch is then genuinely excluded from the graph.
2. **Self-play RL** — batched self-play generates games, the replay ring buffer mixes them with external human-game samples (`--mix-ratio`), and each batch runs `--epochs` fresh optimizer updates.

| Aspect | Implementation |
|---|---|
| Optimizer | Adam (β 0.9/0.999, ε 1e-8), L2 via `--weight-decay 1e-4`, global-norm clipping at 2.0 |
| LR schedule | warmup `base·(0.1+0.9·(step+1)/warmup)` then cosine to a 10 % floor; base 2e-3, 2 warm-up batches (RL) / 20 warm-up steps (supervised) |
| Loss | policy CE + `value_loss_weight` × weighted value MSE (per-sample confidence normalized by its batch sum) |
| Batch | `--batch-size` (default 512; the reference campaign uses 128) with a `--mix-ratio` share of external samples, each drawn with a fresh D4 transform |
| Replay | FIFO ring (`--buffer 20000` in the reference campaign), streamed to `trainer.replay` for resume |
| Precision | `--precision bf16` (autocast) or `fp32`; TF32 kernels are on by default in both |
| Checkpoints | `best_policy_<tag>_bin` (latest) and `best_policy_<tag>_val_bin` (best supervised validation objective); the `best` prefix is historical and does not mean strongest |
| Metrics | `runs/<tag>/metrics.csv` — fixed train/validation probe losses, value error, top-1, entropy, value weight |
| Resume | validates batch index, seed, options/env contract, dataset path/size/mtime, replay contents and Adam state; the saved LR horizon is preserved so extending `--games` stays at the floor |
| Early stopping | supervised `--early-stop-patience`; patience survives resume; the validation-best checkpoint is never restored automatically |

Deadlines are logged honestly: `[train]` prints `selfplay_seconds` and `update_seconds` separately, `[progress]` adds an ETA that accounts for scheduled arenas.

**Checkpoint formats and known-bad models.** Trainer state is format **v2** (network + BN + Adam + replay + schedule horizons + option/dataset contract). Early v1 experiment states are rejected on load because their policy-only optimization could decay the value branch to ~1e-37. The verified final checkpoint is `best_policy_gomoku_fast_bin`, produced by a zero-update resume-and-save of the verified RL model: its `.bin` tensors, replay buffer and network metadata are byte-identical after the round trip. Do **not** start further training from `best_policy_verify_learning_20260926_*`; those checkpoints come from the pre-fix code and are kept only as evidence of the value-branch collapse.

---

## Dataset

`data/renjunet.xml` (RenjuNet, 159,000 records) → `data/gomoku_v2`:

| | |
|---|---|
| Accepted games / positions | 4,218 games / 129,588 positions (from rules 7/11/28 = Gomoku Pro / Swap2 / Swap; Classic Renju excluded) |
| Rejected with reasons | other rule 138,416; swap opening 6,020; shorter than 10 moves 10,317; terminal/result conflict 18; moves after terminal 6; duplicate game 5 |
| Policy targets | first five placements excluded; label smoothing 0.1 for external samples |
| Value supervision | disabled (`value_weight 0`) because the source rules differ from this engine |
| Split (seed 1, 10 %) | 116,781 train / 12,807 held out / 619 purged as D4-equivalent to training / **12,188 independent validation** |
| Storage | float32 `[N,4,15,15]` states, `[N,225]` legal policies, `[N]` outcomes in [−1,1], uint64 game IDs, JSON metadata; 557.6 MiB total, mmap-read |

Binary arrays start with a little-endian `uint64` element count. The loader re-validates every record (binary/finite planes, no overlap, last-move ownership, normalized legal policy with no mass on occupied cells, consistent stone counts, |z| ≤ 1, no already-terminated position). Legacy data without game IDs can only be grouped with `--infer-game-groups 1`, which requires complete ordered games — a random row split cannot repair mixed or truncated data and is refused.

---

## Measuring strength

Freestyle Gomoku is strongly first-player biased, so this project only reports paired-opening results, and the arena prints per-game W/L/D, the score, the first-player win rate and the pair record.

Fresh measurement for this README (`build-training/`, seed 1, local checkpoints):

```sh
./build-training/gomoku_train arena --model best_policy_quality5000_20260926_bin \
  --baseline best_policy_fromzero_20260926_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
# arena 58W 19L 3D score=0.74375 first_player_win_rate=0.688312 games=80 pairs=21W/1L/18D
```

Recorded earlier with the verified RL model against a frozen inference export of the older batch-170 model, same protocol family:

```sh
./build-training/gomoku_train arena --model best_policy_verify_rl_20260926_bin \
  --baseline <export-of-the-old-batch-170-model> --games 40 --playout 400 \
  --opening-moves 4 --seed 1
# 40W 0L 0D, pairs 20W/0L/0D, first-player win rate 0.500, 42.1 s
```

Value quality was checked on **new** self-play games rather than training data: 64 games from an unused seed, 167 of 855 positions purged as D4-overlapping with the replay buffer, and the remaining 688 evaluated game-by-game. Game-averaged MSE: model **0.1359**, best colour-only constant (an optimistic reference) 0.2877, best global constant 0.9939, always-zero 1.0000.

Interpreting these numbers:

- one seed, one protocol, one opponent — not Elo, not a statement about all openings;
- a rising parameter count, a falling training loss, a passing test suite and a kernel speed-up are each insufficient evidence of stronger play;
- a first-player win rate far from 50 % means the opening protocol, not the model, dominates the score.

---

## Deployment: ONNX, browser and terminal play

Training checkpoints keep raw convolution + BN tensors so training can resume faithfully. The exporter reads those authoritative tensors and folds BN for inference:

```sh
./build/gomoku_train export --model best_policy_quality5000_20260926_bin --output model_export
python3 tools/export_onnx.py --bin-dir model_export model.onnx     # 30.9 MiB, log_policy [B,225] + value [B,1]

python3 tools/play_onnx.py   --model model.onnx --playout 400 --device cuda   # terminal
python3 tools/server_onnx.py --model model.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000                                                # browser
```

ONNX Runtime does not need PyTorch, but its CUDA provider links against CUDA/cuDNN libraries that pip installs under `site-packages/nvidia` — off the dynamic loader path. The engine therefore calls `onnxruntime.preload_dlls()` before creating the session; without it, ONNX Runtime silently falls back to CPU with only a warning. `--device cuda` now fails loudly if the provider cannot load, and `--device auto` is the only mode that falls back. Measured through the web server on this machine: 400 playouts per AI move in ≈ 0.5–0.8 s on the GPU.

---

## Repository layout

```
src/            C++17 engine
  game.{h,cpp}        board, rules, Zobrist position keys, state encoding
  net_config.h        single source of truth for the network shape
  network.{h,cpp}     PureNet: native folded-BN CPU inference, parameter/checkpoint I/O
  network_cuda.cpp    LibTorch/CUDA backend: training step, batched inference, TF32/BF16, fused Adam
  selfplay.{h,cpp}    batched lock-free MCTS, batched self-play, arena, tree reuse
  dataset.{h,cpp}     mmap dataset, D4 augmentation, whole-game split with D4 purge
  train.{h,cpp}       Trainer: pretraining, self-play RL, replay, metrics, checkpoints, resume
  gradient.{h,cpp}    reference CPU kernels (used by the gradient tests)
  main.cpp            CLI: train | selfplay | arena | human | export
tools/          Python toolchain: dataset conversion/audit, ONNX export, learning benchmark, ONNX play/web server
tests/          C++ test/benchmark executables + Python unittest suite
data/           dataset prefix (large binaries are generated locally, not committed)
runs/           per-tag metrics.csv (generated locally, not committed)
```

## Tests and CI

```sh
ctest --test-dir build --output-on-failure          # 11/11, 35 s on this laptop
GOMOK_RUN_TRAINING_TESTS=1 GOMOK_TRAIN_BINARY=./build/gomoku_train \
  python3 -m unittest discover -s tests -p 'test_*.py'
```

Coverage: board rules and termination, tactical search, value signs, real concurrent self-play leaf queues, legal policy targets, dataset grouping/augmentation/D4 invariants, CPU↔CUDA backend parity, optimizer/checkpoint recovery and I/O failure injection, CLI error handling, ONNX export parity, and the Python tools. CI builds the CPU engine with ASan+UBSan on a shrunken trunk and compiles the production shape separately.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GOMOK_SEED` | 1 (`--seed` wins) | RNG seed |
| `GOMOK_MCTS_THREADS` | `min(8, physical cores)` | search worker count, clamped 1–32 |
| `GOMOK_LEAF_BATCH` | derived from the inference target | in-flight leaves per game, clamped 1…playout |
| `GOMOK_TREE_REUSE` | **1** | subtree reuse between moves; `0` restores rebuild-every-move |
| `GOMOK_BATCH_WAIT_US` | CUDA 100, CPU 0 | inference collection window |
| `GOMOK_CACHE_ENTRIES` | 32768 | per-batch evaluation cache capacity |
| `GOMOK_PROFILE` | off | print self-play evaluation/cache/batch statistics |
| `GOMOK_TF32` / `GOMOK_INFERENCE_TF32` | 1 / 1 | TF32 training / inference kernels |
| `GOMOK_FUSED_ADAM` | 1 | fused Adam update |
| `GOMOK_DETERMINISTIC` | 0 | deterministic cuDNN kernels (slower) |
| `GOMOK_PINNED_INFERENCE` | 0 | pinned staging experiment (no measured gain) |
| `GOMOK_PERSISTENT_WORKERS` | 0 | persistent searcher threads (no measured gain) |
| `GOMOK_LOG_FILE` | off | duplicate logs to a file |
| `GOMOK_RUN_TRAINING_TESTS` | off | enable the GPU training lifecycle test |

## Where to look in the code

| Path | What it is |
|---|---|
| [`src/net_config.h`](src/net_config.h) | the single definition of the network shape |
| [`src/selfplay.cpp`](src/selfplay.cpp) | batched lock-free MCTS, self-play loop, arena, subtree reuse |
| [`src/network_cuda.cpp`](src/network_cuda.cpp) | TF32/BF16, folded BN, fused optimizer, checkpoint I/O |
| [`src/train.cpp`](src/train.cpp) | pretraining, self-play RL, replay, metrics, resume contract |
| [`src/main.cpp`](src/main.cpp) | the `gomoku_train` CLI |
| [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh) | the reference 5,000-game campaign |
| [`tools/benchmark_learning.py`](tools/benchmark_learning.py) | isolated FP32-vs-BF16 supervised comparison driver |
| [`tests/`](tests) | C++ test/benchmark targets and the Python unittest suite |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CPU ASan/UBSan jobs and the production-shape compile job |

## Scope, provenance and limits

- **Baseline provenance.** The ≈3-day reference figure is a measurement of `junxiaosong/AlphaZero_Gomoku` (PyTorch, 15×15, comparable self-play + training budget) on this same laptop, recorded outside this repository, so it cannot be re-derived from the artifacts here.
- **2.5-hour provenance.** There is no retained stdout log for the 5,000-game campaign. The duration is derived from artifact timestamps and checkpoint state: pretraining `18:19 → 18:28:59`, self-play `18:29:00 → 20:49:49`, with the checkpoint's `trainer.meta` confirming 6 supervised epochs (5,478 updates) and batch 250 of 250 (6,000 RL updates, replay 9,975 of 20,000). Re-running the script reproduces the protocol; the wall-clock is machine- and thermal-dependent.
- Numbers in the tables come from reproducible commands in this repository or from measurements taken for this README on the machine above; they are single-run, single-seed results unless stated otherwise.
- Microbenchmarks exclude self-play and full validation; end-to-end supervised numbers include them.
- Measurements taken on an earlier network, earlier data, or before TF32 was correctly disabled are not carried over into this README; where local notes disagree with a reproducible command, the command wins.
- Training is CUDA-only by construction; CPU builds provide inference, search, the arena and the tests.
- RenjuNet's opening and overline rules differ from this engine's freestyle rules, so the human-game partition supervises **policy only**; value learning comes from self-play.

## References and acknowledgements

- The reference implementation this project was rewritten from: [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku).
- AlphaZero method: Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* ([arXiv:1712.01815](https://arxiv.org/abs/1712.01815)).
- Training data: RenjuNet game records (`data/renjunet.xml`).
- Numerical-accuracy guidance: [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

Licensed under the [MIT License](LICENSE) © 2026 LangLang03.
