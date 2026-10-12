# GomokuZero — 15×15 AlphaZero Gomoku in C++17 / CUDA

[![CI](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml/badge.svg)](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **This project is vibe-coded.** The engine and this document were produced with AI coding agents under human direction, and reviewed against the commands, logs and artifacts named below. Every measurement in this README is tied to something reproducible precisely so it can be checked rather than taken on trust — treat the code the same way.

A from-scratch AlphaZero-style engine for **15×15 freestyle Gomoku** (five or more in a row wins, no Renju forbidden moves, no swap openings). Board, PUCT/MCTS, batched self-play, dataset pipeline, trainer, checkpointing, arena and CLI are C++17; the neural network runs through LibTorch/CUDA for training and through a native folded-BN CPU path for inference. A Python toolchain prepares the human-game dataset, exports ONNX, quantizes it and serves a browser/terminal opponent.

The starting point was the well-known Python project [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku). On the laptop described below, one comparable 15×15 campaign on the current **16-block × 256-channel** network — 250 self-play batches of 64 games at 400 playouts per move — took **about 7 days** there and **about 14 hours** here — see [Results at a glance](#results-at-a-glance) and [Why it is fast](#why-it-is-fast-the-optimization-list) for what is measured and what is mechanism.

| | |
|---|---|
| Game | 15×15 freestyle Gomoku, five-or-more wins, Black moves first |
| Network | residual trunk, **16 blocks × 256 channels**, 19,025,709 parameters, 67×67 trunk receptive field |
| Input / output | `[B,4,15,15]` planes → `log_policy [B,225]` + `value [B,1]` (tanh) |
| Search | batched PUCT/MCTS, 400 playouts/move default, virtual loss, subtree reuse, tactical exact values |
| Training | supervised pretraining on human games → search distillation → self-play RL, Adam + cosine schedule, BF16 opt-in, value-weighted checkpoint selection |
| Stack | C++17, CMake, LibTorch/CUDA training, native CPU inference, ONNX Runtime deployment |
| Dataset | RenjuNet XML → 4,218 games / 129,588 positions, whole-game split with D4 leakage purge, plus 3,000 search-distilled self-play games / 51,755 positions |

The reference network is built with `-DGOMOK_NET_CHANNELS=256 -DGOMOK_NET_RES_BLOCKS=16`; the compile-time default is still the older 12-block × 192-channel shape, which is a different checkpoint format and cannot be mixed with it.

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
- [Deployment: ONNX, quantization, browser and terminal play](#deployment-onnx-quantization-browser-and-terminal-play)
- [Repository layout](#repository-layout)
- [Tests and CI](#tests-and-ci)
- [Environment variables](#environment-variables)
- [Where to look in the code](#where-to-look-in-the-code)
- [Scope, provenance and limits](#scope-provenance-and-limits)

---

## Results at a glance

Reference campaign: [`tools/train_256x16_quality.sh`](tools/train_256x16_quality.sh) — the current 256×16 pipeline, in three resumable stages:

1. **Search-distilled self-play data** — 3,000 games at 400 playouts from the strongest 192×12 checkpoint → 51,755 positions with real freestyle outcomes (value supervision on), 17.3 positions per game.
2. **Supervised pretraining** — 6 epochs of policy imitation on the 116,781 human positions (batch 256, BF16, best-by-validation), then 2 more epochs on the distilled set (absolute target 8).
3. **Self-play RL** — **250 batches × 64 games = 16,000 games** at 400 playouts/move with 16 fresh optimizer minibatches per batch (4,000 optimizer updates), replay buffer 50,000, 35 % human-game samples per batch, value-weighted checkpoint selection on a held-out self-play probe, and 80-game arenas every 50 batches.

| | Reference: junxiaosong/AlphaZero_Gomoku (PyTorch) | This repository (C++17 + LibTorch/CUDA) |
|---|---:|---:|
| Comparable 15×15 campaign, same laptop | **≈ 7 days** | **≈ 14 hours** |
| Speed-up | — | **≈ 12×** |

How the ≈14 hours breaks down (this machine, `build-256x16/`):

| Stage | Measured | Basis |
|---|---:|---|
| Self-play RL, batches 50 → 250 resumed from the batch-50 checkpoint (200 batches, 4 × 80-game arenas), 64 games per batch at 400 playouts | 11.1 h wall clock | pipeline lock `2026-09-29 01:29` → final checkpoint `12:35` |
| That leg extrapolated to the full 250 batches | ≈ 13.9 h | 11.1 h × 250/200. The same log's steady-state rate is 211 s per batch and its `[progress]` lines project 16.3 h for the whole run (`elapsed` + *remaining* `ETA`: batch 49 `2:53:16 + 13:23:38`, batch 56 `3:35:53 + 12:40:07`) |
| Supervised pretraining, 6 human + 2 distilled epochs, batch 256, BF16 | ≈ 17 min | 3,148 updates (457 × 6 + 203 × 2) at the run's measured 807 samples/s, plus 8 validations |
| Distillation data, 3,000 games at 400 playouts (192×12 generator) | ≈ 2 h | the generator script's own 2.2 s/game estimate |

The comparison row uses the RL stage, which is where the two implementations differ. A clean run sits between the two estimates above — 13.9 h if the resumed leg's rate holds, 16.3 h if the first (interrupted) run's own projection holds — which is why the headline says ≈14 hours. The ≈2 h distillation batch is a one-off data generation step that runs once per network family.

What that 250-batch run actually reached (logged in `runs/q256_rl/metrics.csv` and `runs/train_256x16.log`):

| Measurement | Result |
|---|---|
| RL validation, batch 0 → batch 250, 2,048 sampled positions | policy CE 2.047 → **1.807**, top-1 46.3 % → **48.9 %**, value MSE 1.091 → **0.751** |
| Arena, batch 50 vs the distilled supervised checkpoint, 80 games, 400 playouts, 4-move paired openings | `q256_rl` (batch 50) 41 W / 37 L / 2 D, score 0.5250, pairs 6 W / 4 L / 30 D, first-player win rate 0.795, 1,058 s |

Supporting measurements from the same campaign and the same 256×16 network:

| Measurement | Result |
|---|---|
| ONNX Runtime vs native C++ engine, 30-ply replay at 200 playouts/move | 343.8 ms/move vs 348.2 ms/move, identical move at all 30 positions, 4,798 network calls each |
| Forward pass, batch 1 (CUDA) | C++ 2.138 ms vs ONNX Runtime 1.752 ms |
| Forward pass, batch 32 (CUDA) | C++ 22.896 ms (1,398 pos/s) vs ONNX Runtime 23.358 ms (1,370 pos/s) |
| FP16 / INT8 ONNX variants of the 72.5 MiB export | 38.0 MB / 19.2 MB, top-1 agreement with FP32 100.00 % / 93.23 %, value MAE 0.0003 / 0.0674 |
| C++ test suite, 12×192 production build | `ctest` **11/11 in 71.0 s** |
| Dataset load + validation + whole-game split | 129,588 positions in 0.394 s, peak RSS 570 MiB |

Measurements taken on the older 12×192 network are kept below and labelled as such; they are not silently transferred onto the new shape. Every number here is tied to a command, a log line or an artifact; nothing is an Elo or a claim about all opponents.

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
| Python tools | Python 3.14 in `.venv/`: torch 2.14.0+cu130, numpy 2.5.3, onnx 1.23.0, onnxruntime-gpu 1.28.0 |
| Deployment | onnxruntime-gpu 1.28.0 (CUDA 13 / cuDNN 9), CUDA libraries from the pip `nvidia-*` wheels |

`nvcc` is **not** required: the CUDA backend is LibTorch, so CMake only needs a Python interpreter that can `import torch` and a `libtorch_cuda.so`. Pin the interpreter with `-DPython3_EXECUTABLE=...` if several environments are installed (this machine has both a system torch 2.12 and a venv torch 2.14).

The 256×16 network raises the 12×192 parameter count by 2.35× (19.0 M vs 8.1 M) and roughly doubles the arithmetic of every forward pass, on the same 8 GiB laptop GPU; the campaign numbers above are what that costs here.

---

## Quick start

### 1. Build and test

```sh
# Prerequisites: CMake >= 3.18, a C++17 compiler, Python 3 with a CUDA-enabled torch
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'   # e.g. 2.12.0+cu130 13.0

# Reference 256x16 shape (this is what tools/train_256x16_quality.sh expects):
cmake -S . -B build-256x16 -DCMAKE_BUILD_TYPE=Release \
  -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DGOMOK_NET_CHANNELS=256 -DGOMOK_NET_RES_BLOCKS=16 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-256x16 -j2
ctest --test-dir build-256x16 --output-on-failure
```

A CUDA-free build is supported for inference, search and the test suite; **training is refused without the LibTorch/CUDA backend**:

```sh
cmake -S . -B build-cpu -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j2 && ctest --test-dir build-cpu --output-on-failure
```

Omitting both `-DGOMOK_NET_*` flags builds the older 12×192 shape. Shapes are a compile-time constant, so checkpoints, exports and benchmarks belong to exactly one shape.

### 2. Play, watch self-play, or run a match

```sh
./build-256x16/gomoku_train human --model best_policy_q256_rl_val_bin --playout 400   # type r,c (0-14)

GOMOK_PROFILE=1 ./build-256x16/gomoku_train selfplay \
  --model best_policy_q256_rl_val_bin --games 8 --batch 8 --playout 400 --seed 1

./build-256x16/gomoku_train arena --model best_policy_q256_rl_val_bin \
  --baseline best_policy_q256_distill_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
```

Checkpoints are local artifacts (`best_policy_*_bin/`) and are not in version control; train one first, or export random weights with `./build-256x16/gomoku_train export --output my_model_bin`.

### 3. Prepare data and train

```sh
python3 tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python3 tools/audit_dataset.py data/gomoku_v2 --state-overlap   # audit JSON on stdout

# Supervised pretraining (absolute target: 6 epochs; --games 0 means supervised only)
./build-256x16/gomoku_train train --mix data/gomoku_v2 --pretrain 6 --games 0 \
  --batch-size 256 --epochs 1 --precision bf16 --seed 1 \
  --val-samples 2048 --val-freq 1 --kl-freq 0 --early-stop-patience 2 --tag q256_pre

# Self-play RL from the best validation checkpoint (absolute target: batch 250)
./build-256x16/gomoku_train train --init best_policy_q256_pre_val_bin --mix data/gomoku_v2 \
  --pretrain 6 --games 250 --batch-games 64 --batch-size 256 --epochs 16 --precision bf16 \
  --playout 400 --buffer 50000 --mix-ratio 0.35 --seed 1 \
  --val-samples 2048 --val-freq 5 --kl-freq 5 --selfplay-val-every 10 \
  --select-value-weight 1 --check-freq 25 \
  --eval-games 80 --eval-freq 50 --eval-playout 400 --tag q256_rl
```

The reference campaign with its distillation stage, resumable stages, GPU lock and arenas is [`tools/train_256x16_quality.sh`](tools/train_256x16_quality.sh); the earlier 12×192 5,000-game campaign is still [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh). `--games` and `--pretrain` are **absolute targets**, not increments — resuming batch 100 with `--games 250` generates 150 more batches.

### 4. Export, quantize and serve

```sh
./build-256x16/gomoku_train export --model best_policy_q256_rl_val_bin --output model_export_256x16
python3 tools/export_onnx.py --bin-dir model_export_256x16 256x16.onnx     # 76.0 MB
python3 tools/quantize_onnx.py --src 256x16.onnx --out-dir runs/quantized  # fp16 + int8 + drift report

python3 -m pip install 'onnxruntime-gpu[cuda,cudnn]'
python3 tools/server_onnx.py --model 256x16.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000        # open http://127.0.0.1:8000
```

---

## Why it is fast: the optimization list

The reference implementation is a single-process PyTorch trainer that plays one game at a time, evaluates one leaf per forward call, runs FP32 and rebuilds the search tree every move. Everything below replaces one of those behaviours. "Baseline" always means that reference behaviour, and each row carries its evidence.

Kernel and micro-benchmark rows in this section were measured on the **12×192** network (the shape the repository default builds); the end-to-end campaign rows are 256×16. The optimizations are shape-independent; the constants are not.

### A. Self-play parallelism and search scheduling

| # | Optimization | Baseline | Here | Evidence / impact |
|---|---|---|---|---|
| A1 | **Batched self-play over concurrent games** — one call advances `--batch-games` games (default 64) | one game per iteration | 64 games in flight, 64 independent trees | `src/selfplay.cpp` `BatchedSelfPlay::run_batch`; the 256×16 RL run generates 64 games per batch |
| A2 | **Lock-free MCTS worker pool** with a condition-variable round launcher | single-threaded search per game | `min(8, physical cores)` workers, clamped 1–32 | `src/selfplay.cpp` `SearchWorkerPool`; `GOMOK_MCTS_THREADS`, `--threads` |
| A3 | **CPU affinity that reserves the fastest cores for the GPU coordinator** | no pinning; searchers and inference fight for cores | physical cores sorted by max frequency; the fastest `inference_threads` cores are skipped | `src/selfplay.cpp` (topology + `cpufreq` read) |
| A4 | **Batched leaf evaluation with a collection window** — spins up to `GOMOK_BATCH_WAIT_US` (100 µs on CUDA) and submits one GPU call | one forward per leaf, GPU idle between calls | target inference batch 32; leaves from all games are merged | on 12×192 the batch sweep 16…512 made **32 optimal** (2,846 eval/s vs 2,588 at 128) and moving the target 128 → 32 also cut collision retries 32 %; on 256×16 throughput still climbs up to batch 32 (1,398 pos/s vs 468 at batch 1), so the target stays 32 |
| A5 | **Fixed-point virtual loss** applied to the in-flight path | none — threads pile onto the same child | integer atomic visits + Q20 value sum, default 8, disabled when the leaf batch is 1 | sweep on 12×192: virtual loss 3 → 8 cuts collision retries **27 %** (720 → 523 per evaluation) and is **6 % faster per game** (1.66 → 1.56 s) |
| A6 | **Collision retries do not consume the search budget** | retry costs a playout | `NODE_EXPANDING` contention returns `Retry` without decrementing issued playouts | `src/selfplay.cpp` retry loop; playout cap enforced by CAS |
| A7 | **Subtree reuse between moves** (`GOMOK_TREE_REUSE=1`, default on) | `reset_tree()` every move, search 400 playouts from scratch | the chosen child is promoted (ping-pong node arenas), retained visits count toward the next budget | CPU A/B on the small model: 8 games **9.40 s → 6.44 s**; fixed-line control: search loop 0.040 s → 0.012 s. Enabled in the 256×16 campaign; not isolated on that shape |
| A8 | **Cross-move evaluation cache** — Zobrist position key, FIFO ring, 32,768 entries, shared by all 64 games in a batch | every leaf re-evaluated | cache hit rate 0.27–0.41 in 12×192 production self-play; also de-duplicates identical leaves inside one batch | `GOMOK_CACHE_ENTRIES`; profile line `cache_hits`/`cache_hit_rate` |
| A9 | **Tactical and terminal short-circuits** | net consulted for every leaf | immediate five → exact +1 and no inference; forced block collapses expansion to one child; finished games and solved nodes never hit the net | `src/selfplay.cpp`; in the 256×16 replay the per-ply network calls tail off exactly as the game resolves: 200, 169, 147, 146, 99, 98, 93, 121, 120, then **4, 1, 0** over the final three plies |
| A10 | **Preallocated node arenas and scratch buffers** (32-byte aligned nodes, no allocation during search) | per-node Python objects | one arena per game per batch, worker scratch reused across moves | `alignas(32) GNode`, `SearchWorkerScratch`, buffers sized once per `run_batch()` |
| A11 | **Batched arena matches** | one game at a time | up to 32 games in lockstep, exactly two batched calls per round (one per model) | `play_match`; the 80-game 256×16 arena above finished in 1,058 s at 400 playouts |

### B. Precision and GPU kernels

| # | Optimization | Baseline | Here | Evidence / impact |
|---|---|---|---|---|
| B1 | **TF32 for cuDNN and cuBLAS**, scoped by an RAII guard, on by default | FP32 kernels | `GOMOK_TF32` / `GOMOK_INFERENCE_TF32` (both 1) | 12×192: strict FP32 → TF32 inference 172.8 → 41.9 ms (**4.1×**), training 354.6 → 152.0 ms (**2.3×**) |
| B2 | **BF16 training autocast with FP32 master weights, loss and Adam moments**; the two final head nonlinearities stay FP32 | FP32 only | `--precision bf16` | 12×192: training 152.0 → 76.6 ms (**1.98×** vs TF32); end-to-end supervised stage 460.4 s → 260.5 s (**1.77×**). Both 256×16 supervised stages and the RL stage run BF16 |
| B3 | **Fused optimizer and fused gradient math** | per-parameter Python loop | `at::_fused_adam_`, `_foreach_norm` for the global norm, `_foreach_mul_` for clipping, five scalars read back in one sync | `GOMOK_FUSED_ADAM=1`; its own A/B measured only ~0.5–1 % (kept for the reduced launch count) |
| B4 | **Batch norm folded into the convolutions, once per weight generation** | BN kernels on every forward | inference graphs contain conv + bias + ReLU only | `src/network_cuda.cpp` `fold_bn`; the 256×16 export is 19,008,807 folded parameters out of 19,025,709 trained ones |
| B5 | **`torch::InferenceMode` + `eval()` + channels-last conv weights/activations** | autograd-tracked FP32 NCHW | no graph, no version counters, tensor-core-friendly layout | `src/network_cuda.cpp` |
| B6 | **cuDNN autotune deliberately disabled** | `cudnn.benchmark = True` | autotuning is a net loss because MCTS batch sizes vary (`GOMOK_DETERMINISTIC=1` gives deterministic cuDNN instead) | comment in `src/network_cuda.cpp` |
| B7 | **Redundant conv biases removed from the optimizer** | every parameter updated | conv biases are frozen (BN makes them redundant; their rounding noise was being amplified by Adam), which also shrinks Adam state | `requires_grad(false)` on conv biases |
| B8 | **The unsupervised value branch is not even in the graph** | `policy + 0 * value` still builds value gradients that Adam's weight decay destroys | when value supervision is off the value term is absent, its parameters keep their own Adam step counter | fixes a measured collapse in which the value head's dense weights had decayed to ~1e-37 under the old policy-only objective |
| B9 | **Native CPU inference path** (folded BN, 8-lane unrolled convolution, thread-local scratch, sample-parallel batches) | — | keeps MCTS, the CLI and the tests usable without CUDA; CPU training is refused, not silently degraded | measured on this machine with 256×16: **3,374.8 ms per forward** (single thread) versus 12 ms for the FP16 ONNX export on CPU — fine for tests and the small default shape, not for playing the 256×16 model without a GPU |

### C. Data, I/O and checkpoints

| # | Optimization | Baseline | Here | Evidence / impact |
|---|---|---|---|---|
| C1 | **Memory-mapped dataset with index-only splits** | load everything into RAM as tensors | `mmap(PROT_READ, MAP_PRIVATE)` + per-record validation; train/validation are row indices, never a copy | 129,588 positions validated and split in **0.394 s**, peak RSS 570 MiB |
| C2 | **On-the-fly D4 augmentation** (8 symmetries, same transform for state and policy) | pre-expanded or no augmentation | every optimizer step re-draws samples **and** their symmetry | 8× sample efficiency at 1× storage; `--augment 1` |
| C3 | **Fresh minibatch every optimizer step** | one minibatch replayed for many updates | new replay and external draws per step, logged as `replay_draws`/`external_draws` | the 256×16 RL run draws 2,672 replay + 1,424 external samples for each of its 16 updates per batch |
| C4 | **Streaming, validating data tools** | in-memory numpy pipelines | per-member mmap in `npz_to_bin.py`, per-game validation and D4 de-duplication in `parse_renjunet.py`, temp-file + rename writes | 159,000 XML records → 4,218 accepted games with a rejection ledger |
| C5 | **Whole-game split with a D4 leakage purge** | random sample split | game IDs hashed with `splitmix64(id ^ seed)`, then held-out positions D4-equivalent to training are dropped; the identical algorithm exists in C++ and Python with a golden cross-language test | a random split would leak ≈ 88.9 % of games; the game split leaves 619/12,807 = 4.83 % D4 overlap, purged → 12,188 independent validation positions. The 3,000-game distilled set is audited the same way (80.4 % expected leakage under a row split) |
| C6 | **Atomic checkpoints and a resume contract** | save/load weights only | `.staging` → rename with `.previous` rollback; network + BN + Adam + replay + batch index + schedule horizons + seed + option/dataset fingerprint, validated before use | a completed run cannot silently restart from scratch; `--games`/`--pretrain` stay absolute across resumes. The 256×16 RL run was interrupted once and resumed from its batch-50 checkpoint |
| C7 | **Honest ETA and separated timings** | wall-clock per epoch | self-play and update seconds measured separately, smoothed over recent batches, scheduled arenas added separately | `[train]`, `[progress]` log lines; the ≈14 h figure above is read from them |
| C8 | **Value-weighted checkpoint selection** | the last batch wins, or policy loss alone decides | `--select-value-weight W` scores a snapshot as `policy + W × value` on a held-out self-play probe (`--selfplay-val-every N`) instead of on the human data, whose outcomes are not value labels for this engine | `src/train.cpp`; without it the 256×16 pipeline kept a snapshot that had never been scored on the value head at all |

### D. Engineering

| # | Optimization | Why |
|---|---|---|
| D1 | One shape source of truth (`src/net_config.h`) for trunk width, block count, head sizes, reachable from C++ and Python | the shape used to be duplicated in four places, so the network, the exporter and the tests could disagree. Every shape-dependent number in this README (19,008,807 folded / 19,025,709 trained, 67×67) derives from it |
| D2 | `gomoku_core` built once as a static library; executables and tests link it | avoids recompiling the expensive LibTorch translation unit per target |
| D3 | LTO on by default, with the LibTorch TU compiled `-O1 -fno-lto` | keeps release optimizations while bounding compile time and memory |
| D4 | CI builds the CPU engine with ASan + UBSan on a shrunken trunk, plus a production-shape compile job | concurrency and memory bugs are caught without paying for the full 16×256 network in every job |
| D5 | Strict CLI parsing (unknown/duplicate/missing-value options are fatal) and a failing model load is never silently replaced by random weights | removes the "trained the wrong experiment" failure mode |
| D6 | A C++ mirror of the Python search (`tests/play_benchmark.cpp` vs `tools/gomoku.py`) | makes "C++ engine vs ONNX Runtime" a controlled measurement of one backend instead of two implementations; both drivers are checked to choose identical moves |

### What did **not** pay off (kept, off by default)

| Attempt | Result |
|---|---|
| `GOMOK_PINNED_INFERENCE=1` — pinned host staging, non-blocking copies | no gain on the RTX 5060; the inference API still synchronizes per batch, so the copy cannot overlap. Default 0 |
| `GOMOK_PERSISTENT_WORKERS=1` — keep searcher threads and scratch alive across moves | no gain on the RTX 5060. Default 0 |
| Fused Adam | 0.5–1 % in its own A/B, inside noise; kept only for the reduced launch count |
| Subtree reuse on the production CUDA path | isolated only on the small CPU model so far (31 % shorter games, −2.9 % evaluations per move); enabled but not isolated on CUDA |
| cuDNN autotune | deliberately off (see B6) |
| Native CPU training | refused by design: `train --cpu` exits with an error instead of pretending |
| Native CPU inference for the 256×16 shape | measured 3,374.8 ms per forward; usable for tests, not for play (see B9) |

**Attribution of the ≈12×.** The measured, isolatable parts are: precision (TF32/BF16 ≈ 2–4× on kernels, 1.77× end-to-end for a supervised stage), batched GPU evaluation (32 leaves per call instead of one; the batch sweep shows the GPU is latency-bound, not throughput-bound, at batch 1), subtree reuse and the shared evaluation cache (fewer network calls per move), tactical short-circuits (no inference on decided positions), and removing Python/GIL overhead from the hot path (C++ single process). Only the precision factors and the batch-size sweep were measured in isolation; the rest is mechanism plus the end-to-end ≈14-hour result. The ratio is smaller than a like-for-like kernel comparison suggests because the reference implementation is one game at a time in FP32 while this campaign runs 2.35× the parameters and 400 playouts per move.

---

## Network and search

### State encoding

Four planes in the current player's perspective, matching `Board::encode_state` and the Python tools: own stones, opponent stones, last move, and "Black to move" (constant plane). This orientation means the network never sees absolute colours.

### Network

```
state [4,15,15]
  stem      conv3x3 4 -> 256, BN, ReLU
  trunk     16 x [ conv3x3 256->256, BN, ReLU, conv3x3 256->256, BN, add, ReLU ]   (67x67 receptive field)
  policy    conv1x1 256->2, BN, ReLU, flatten(450), fc 450->225, log_softmax
  value     conv1x1 256->1, BN, ReLU, flatten(225), fc 225->64, ReLU, fc 64->1, tanh
```

| Part | 16×256 (reference) | 12×192 (default build) |
|---|---:|---:|
| stem 3×3 4→C | 9,472 | 7,104 |
| trunk, 16 / 12 blocks (99.3 % / 98.5 % of the model) | 18,882,560 | 7,967,232 |
| policy head (1×1 conv + 450→225 dense) | 101,989 | 101,861 |
| value head (1×1 conv + 225→64 + 64→1) | 14,786 | 14,722 |
| **Total (inference export, BN folded)** | **19,008,807** | **8,090,919** |
| Training checkpoint learned parameters (adds BN γ/β) | 19,025,709 | 8,100,525 |

The 19,025,709 figure is what the trainer prints at startup (`[net] CUDA backend enabled (19025709 parameters)`) and what `runs/model_export_256x16/` reconstructs file by file; the folded export loses exactly the 16,902 batch-norm γ/β values.

Rebuild other shapes with `-DGOMOK_NET_CHANNELS=N -DGOMOK_NET_RES_BLOCKS=N`; checkpoints from different shapes are incompatible. The dense policy head is 450→225 (weight sharing across cells after a 1×1 convolution) rather than the old 900→225 layer that consumed 62 % of all parameters with no cross-position sharing.

### Search

- PUCT/MCTS with `c_puct 5`, `--playout 400` (the reference campaign uses 400 for both data generation and RL), candidate cap 60.
- Leaves are evaluated in their own side-to-move perspective; value signs alternate during backup; terminal positions are exact.
- Immediate wins return exact values without inference; a forced block survives pruning and collapses expansion to one child.
- Virtual loss applies only when several leaves of the same tree are in flight (it is disabled for the serialized arena search).
- Root Dirichlet noise (`ε 0.25`, `α 0.3`) is drawn over **all** legal moves before pruning; self-play policy targets use normalized raw visit counts, temperature controls sampling only.
- Subtree reuse keeps completed visits and values for the next move; root noise is re-drawn, previously visited children are kept and current top candidates are added.
- Arena play uses identical openings with colours swapped and reports per-game W/L/D plus pair results, because freestyle Gomoku's first-player advantage makes raw scores misleading.

---

## Training pipeline

Three stages, driven from one `train` command plus the self-play generator:

1. **Supervised pretraining** on human games — policy cross-entropy only by default. The RenjuNet sources use different opening/overline rules, so their recorded outcomes must not become freestyle value labels: the dataset metadata sets `value_weight = 0`, and the value branch is then genuinely excluded from the graph.
2. **Search distillation** — self-play positions generated by an already-trained engine carry real freestyle outcomes, so their metadata sets `value_weight = 1` and they *do* supervise the value head. The 256×16 pipeline trains 6 human epochs and then 2 distilled epochs (`--pretrain 8` with the distilled dataset).
3. **Self-play RL** — batched self-play generates games, the replay ring buffer mixes them with external human-game samples (`--mix-ratio`), and each batch runs `--epochs` fresh optimizer updates.

| Aspect | Implementation |
|---|---|
| Optimizer | Adam (β 0.9/0.999, ε 1e-8), L2 via `--weight-decay 1e-4`, global-norm clipping at 2.0 |
| LR schedule | warmup `base·(0.1+0.9·(step+1)/warmup)` then cosine to a 10 % floor; base 2e-3, 2 warm-up batches (RL) / 20 warm-up steps (supervised) |
| Loss | policy CE + `value_loss_weight` × weighted value MSE (per-sample confidence normalized by its batch sum) |
| Batch | `--batch-size` (default 512; the 256×16 campaign uses 256) with a `--mix-ratio` share of external samples, each drawn with a fresh D4 transform |
| Replay | FIFO ring (`--buffer 50000` in the 256×16 campaign, 20000 in the old one), streamed to `trainer.replay` for resume |
| Precision | `--precision bf16` (autocast) or `fp32`; TF32 kernels are on by default in both |
| Checkpoints | `best_policy_<tag>_bin` (latest) and `best_policy_<tag>_val_bin` (best selection score); the `best` prefix is historical and does not mean strongest |
| Selection | `--select-value-weight W --selfplay-val-every N` scores every kept snapshot as `policy + W · value` on a self-play probe held out of the replay buffer; `0 / 0` restores the old policy-only behaviour |
| Metrics | `runs/<tag>/metrics.csv` — fixed train/validation losses, value error, top-1, entropy and the selection weight; the held-out self-play probe and the resulting selection score are printed on the `[val]` log line, not stored in the CSV |
| Resume | validates batch index, seed, options/env contract, dataset path/size/mtime, replay contents and Adam state; the saved LR horizon is preserved so extending `--games` stays at the floor |
| Early stopping | supervised `--early-stop-patience`; patience survives resume; the validation-best checkpoint is never restored automatically |
| Arenas during training | `--eval-games/--eval-freq/--eval-playout/--baseline`; the 256×16 campaign plays 80 games every 50 batches against its own supervised checkpoint |

Deadlines are logged honestly: `[train]` prints `selfplay_seconds` and `update_seconds` separately, `[progress]` adds an ETA that accounts for scheduled arenas.

**Checkpoint formats and known-bad models.** Trainer state is format **v2** (network + BN + Adam + replay + schedule horizons + option/dataset contract). Early v1 experiment states are rejected on load because their policy-only optimization could decay the value branch to ~1e-37. The 256×16 lineage is `best_policy_q256_pre_bin` → `best_policy_q256_distill_bin` → `best_policy_q256_rl_bin`, with the selected snapshot in `best_policy_q256_rl_val_bin`. The older verified 12×192 final checkpoint is `best_policy_gomoku_fast_bin`; do **not** start further training from `best_policy_verify_learning_20260926_*`, which come from the pre-fix code and are kept only as evidence of the value-branch collapse.

---

## Dataset

Two partitions feed the 256×16 pipeline. Human records supervise **policy only**; distilled self-play records supervise both heads.

| | Human: RenjuNet (`data/gomoku_v2`) | Distilled self-play (`data/selfplay_256x16_p400`) |
|---|---|---|
| Source | `data/renjunet.xml`, 159,000 records | 3,000 games at 400 playouts from the 192×12 `best_policy_quality5000_20260926_bin` |
| Accepted games / positions | 4,218 games / 129,588 positions (rules 7/11/28 = Gomoku Pro / Swap2 / Swap; Classic Renju excluded) | 3,000 games / 51,755 positions, 17.3 per game |
| Rejected with reasons | other rule 138,416; swap opening 6,020; shorter than 10 moves 10,317; terminal/result conflict 18; moves after terminal 6; duplicate game 5 | audited: tensors validated, freestyle rules, 0 terminal-policy samples |
| Policy targets | first five placements excluded; label smoothing 0.1 for external samples | raw MCTS visit counts |
| Value supervision | disabled (`value_weight 0`) because the source rules differ from this engine | enabled (`value_weight 1`), outcomes split 27,226 / 24,529 |
| Split (seed 1, 10 %) | 116,781 train / 12,807 held out / 619 purged as D4-equivalent → **12,188 independent validation** | grouped by game ID; a row split would leak 80.4 % of games |
| Storage | float32 `[N,4,15,15]` states, `[N,225]` legal policies, `[N]` outcomes, uint64 game IDs, JSON metadata; 557.6 MiB, mmap-read | same format |

Binary arrays start with a little-endian `uint64` element count. The loader re-validates every record (binary/finite planes, no overlap, last-move ownership, normalized legal policy with no mass on occupied cells, consistent stone counts, |z| ≤ 1, no already-terminated position). Legacy data without game IDs can only be grouped with `--infer-game-groups 1`, which requires complete ordered games — a random row split cannot repair mixed or truncated data and is refused.

---

## Measuring strength

Freestyle Gomoku is strongly first-player biased, so this project only reports paired-opening results, and the arena prints per-game W/L/D, the score, the first-player win rate and the pair record.

Fresh measurement for this README on the 256×16 lineage (`build-256x16/`, seed 1, local checkpoints). The number below is the training run's own arena **at batch 50**, quoted from its log; that checkpoint has since been overwritten by batch 250, so it is a historical record and the command only shows the protocol:

```sh
# What the training run actually did (runs/train_256x16.log, [arena] batch 50):
#   --eval-games 80 --eval-freq 50 --eval-playout 400 --baseline best_policy_q256_distill_bin
# The batch-50 snapshot was later overwritten by the batch-250 checkpoints, so the
# quoted line can no longer be re-run; this is the equivalent standalone form:
./build-256x16/gomoku_train arena --model <the batch-50 snapshot> \
  --baseline best_policy_q256_distill_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
# [arena] batch 50 41W 37L 2D score:0.5250 first_player_win_rate:0.795
#         games:80 pairs:6W/4L/30D seconds:1058.0
```

The same shape cannot be loaded by the old 12×192 binary (the trunk is a compile-time constant), so the 256×16 baseline is this run's own supervised checkpoint: that arena measures **what RL added**, not absolute strength, and 0.525 over 80 games is within noise of "nothing yet". Cross-shape comparisons have to go through the ONNX export.

Recorded earlier on the 12×192 network, same protocol family:

```sh
./build-training/gomoku_train arena --model best_policy_quality5000_20260926_bin \
  --baseline best_policy_fromzero_20260926_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
# arena 58W 19L 3D score=0.74375 first_player_win_rate=0.688312 games=80 pairs=21W/1L/18D

./build-training/gomoku_train arena --model best_policy_verify_rl_20260926_bin \
  --baseline <export-of-the-old-batch-170-model> --games 40 --playout 400 \
  --opening-moves 4 --seed 1
# 40W 0L 0D, pairs 20W/0L/0D, first-player win rate 0.500, 42.1 s
```

Value quality was checked on **new** self-play games rather than training data: 64 games from an unused seed, 167 of 855 positions purged as D4-overlapping with the replay buffer, and the remaining 688 evaluated game-by-game. Game-averaged MSE on that 12×192 model: 0.1359, best colour-only constant (an optimistic reference) 0.2877, best global constant 0.9939, always-zero 1.0000. For the 256×16 model, `runs/q256_rl/metrics.csv` reports the **mix-validation** value MSE falling from 1.091 to 0.751 over the 250 batches — that column is the human/distilled validation split, not a self-play probe. The probe that checkpoint selection actually scores on is only logged: `probe_value` reads 0.806 at batch 10, 0.602 at 40, 0.705 at 50 and 0.522 at 55.

Interpreting these numbers:

- one seed, one protocol, one opponent — not Elo, not a statement about all openings;
- a rising parameter count, a falling training loss, a passing test suite and a kernel speed-up are each insufficient evidence of stronger play;
- a first-player win rate far from 50 % means the opening protocol, not the model, dominates the score;
- the 256×16 arena at batch 50 did **not** yet show RL beating its own supervised checkpoint, and the table above says so rather than dressing 0.525 up.

---

## Deployment: ONNX, quantization, browser and terminal play

Training checkpoints keep raw convolution + BN tensors so training can resume faithfully. The exporter reads those authoritative tensors and folds BN for inference:

```sh
./build-256x16/gomoku_train export --model best_policy_q256_rl_val_bin --output model_export_256x16
python3 tools/export_onnx.py --bin-dir model_export_256x16 256x16.onnx
# 256x16.onnx: 76,050,598 bytes (72.5 MiB), log_policy [B,225] + value [B,1]

python3 tools/play_onnx.py   --model 256x16.onnx --playout 400 --device cuda   # terminal
python3 tools/server_onnx.py --model 256x16.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000                                                # browser
```

ONNX Runtime does not need PyTorch, but its CUDA provider links against CUDA/cuDNN libraries that pip installs under `site-packages/nvidia` — off the dynamic loader path. The engine therefore calls `onnxruntime.preload_dlls()` before creating the session; without it, ONNX Runtime silently falls back to CPU with only a warning. `--device cuda` now fails loudly if the provider cannot load, and `--device auto` is the only mode that falls back.

### Native C++ engine vs ONNX Runtime (256×16, this machine, 2026-10-09)

`tests/play_benchmark.cpp` is a line-by-line mirror of `tools/gomoku.py`'s MCTS — same candidate generation, same tactical shortcuts, same selection formula and tie-breaking — so the two drivers search an identical tree and only the backend differs. Both replay the same 30-ply game at 200 playouts per move:

| Measurement | C++ CUDA | ONNX Runtime CUDA |
|---|---:|---:|
| Forward pass, batch 1 | 2.138 ms (468 pos/s) | **1.752 ms (571 pos/s)** |
| Forward pass, batch 8 | **6.535 ms (1,224 pos/s)** | 9.923 ms (806 pos/s) |
| Forward pass, batch 32 | 22.896 ms (1,398 pos/s) | 23.358 ms (1,370 pos/s) |
| 30-ply replay, 200 playouts/move | 10.31 s total, 343.8 ms/move | 10.45 s total, 348.2 ms/move |
| Network calls over those 30 moves | 4,798 | 4,798 |
| Points where the two engines chose a different move | 0 | 0 |

The two backends are within ~1.3 % of each other end-to-end at MCTS batch sizes, `batch 1` favours ONNX Runtime and larger batches favour C++: at 200 playouts per move the search is a sequence of small, latency-bound calls, which is exactly why A4's collection window matters more than raw throughput. The C++ native CPU path (`--cpu`) is a different story on this shape: 3,374.8 ms per forward, i.e. ~2.5 GFLOPS of scalar work versus 147 GFLOPS available from single-threaded BLAS — usable for tests, not for play.

### Quantization

`tools/quantize_onnx.py` builds deployment variants of the FP32 export and reports the numeric drift next to the size (calibration positions come from the front of the dataset, the quality positions from the end, so the two never overlap):

| Variant | Size | Top-1 agreement | Mean KL | Max KL | Value MAE | ORT CPU, batch 1 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 76.1 MB | — | — | — | — | ≈300–320 ms |
| FP16 | 38.0 MB | 100.00 % | 0.00000 | 0.0000 | 0.0003 | ≈12–15 ms |
| INT8 (QDQ, per-channel) | 19.2 MB | 93.23 % | 0.00585 | 0.0228 | 0.0674 | 6 ms |

FP16 is a free win — same moves, half the size, ~20–25× faster on CPU (≈300 → ≈12 ms in the table, 321 → 15 ms on a repeat run). INT8 changes the top move on ~7 % of positions, so treat it as an option for a CPU-only deployment rather than a drop-in replacement. Measured on CPU with 8 intra-op threads (`runs/quantized/` also holds FP8 and INT4 files that the committed tool does not reproduce); the CUDA provider is not affected by these numbers.

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
  train.{h,cpp}       Trainer: pretraining, distillation, self-play RL, replay, selection, checkpoints
  gradient.{h,cpp}    reference CPU kernels (used by the gradient tests)
  main.cpp            CLI: train | selfplay | arena | human | export
tools/          Python toolchain: dataset conversion/audit, ONNX export/quantization,
                C++-vs-ONNX play benchmark, Python MCTS mirror, learning benchmark, web server
tests/          C++ test/benchmark executables + Python unittest suite
data/           dataset prefixes (large binaries are generated locally, not committed)
runs/           per-tag metrics.csv, campaign logs, ONNX exports (generated locally)
analysis/       review notes, old benchmark JSON/logs, analysis/play_bench (C++ vs ONNX report)
```

## Tests and CI

```sh
ctest --test-dir build-training --output-on-failure     # 11/11, 71.0 s on this laptop (12x192 build)
GOMOK_RUN_TRAINING_TESTS=1 GOMOK_TRAIN_BINARY=./build/gomoku_train \
  python3 -m unittest discover -s tests -p 'test_*.py'
```

Coverage: board rules and termination, tactical search, value signs, real concurrent self-play leaf queues, legal policy targets, dataset grouping/augmentation/D4 invariants, CPU↔CUDA backend parity, optimizer/checkpoint recovery and I/O failure injection, CLI error handling, ONNX export parity, and the Python tools. CI builds the CPU engine with ASan+UBSan on a shrunken trunk and compiles the production shape separately. `gomoku_play_benchmark` is a benchmark target, not a `ctest` test.

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
| [`src/train.cpp`](src/train.cpp) | pretraining, distillation, self-play RL, replay, selection, resume contract |
| [`src/main.cpp`](src/main.cpp) | the `gomoku_train` CLI |
| [`tools/train_256x16_quality.sh`](tools/train_256x16_quality.sh) | the reference 256×16 campaign |
| [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh) | the earlier 12×192 5,000-game campaign |
| [`tests/play_benchmark.cpp`](tests/play_benchmark.cpp) | C++ mirror of the Python search, for backend timing |
| [`tools/gomoku.py`](tools/gomoku.py) | the Python board/state encoding/MCTS the C++ driver mirrors |
| [`tools/bench_play_onnx.py`](tools/bench_play_onnx.py) | the ONNX Runtime side of that comparison |
| [`tools/quantize_onnx.py`](tools/quantize_onnx.py) | FP16/INT8 variants plus the drift report |
| [`tools/benchmark_learning.py`](tools/benchmark_learning.py) | isolated FP32-vs-BF16 supervised comparison driver |
| `analysis/play_bench/report.md` | the C++-vs-ONNX write-up and its raw logs (kept locally: `analysis/` is git-ignored) |
| [`tests/`](tests) | C++ test/benchmark targets and the Python unittest suite |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CPU ASan/UBSan jobs and the production-shape compile job |

## Scope, provenance and limits

- **Baseline provenance.** The ≈7-day reference figure is a measurement of `junxiaosong/AlphaZero_Gomoku` (PyTorch, 15×15, comparable self-play + training budget) on this same laptop, recorded outside this repository, so it cannot be re-derived from the artifacts here.
- **≈14-hour provenance.** No single clean stdout log of the 256×16 campaign is retained, so the figure is a reconstruction with two bounds. **Measured:** the resumed leg (lock `2026-09-29 01:29` → final checkpoint `12:35`) ran 200 batches and 4 arenas in 11.1 h, which extrapolates to ≈13.9 h for 250 batches. **Projected by the run itself:** the `[progress]` lines' `elapsed` + *remaining* `ETA` add up to ≈16.3 h for the whole run (batch 49: `2:53:16 + 13:23:38`; batch 56: `3:35:53 + 12:40:07`), at a steady-state 211 s per batch. The headline uses the lower, measurement-based end. The supervised stages add ≈17 min at their measured throughput, and the 3,000-game distillation batch is a separate one-off data-generation step. The pipeline was interrupted once and resumed from its batch-50 checkpoint; the 25-batch checkpoint interval bounds lost work. Re-running the script reproduces the protocol; the wall clock is machine- and thermal-dependent.
- **7 days vs 14 hours compares the same budget, not the same code.** The original project plays one game at a time in FP32 and evaluates one leaf per forward pass; the 7-day figure is an external measurement of it, on this same laptop, at the budget the 256×16 campaign uses. It is a single data point, not a benchmark curve.
- Numbers in the tables come from reproducible commands in this repository or from measurements taken for this README on the machine above; they are single-run, single-seed results unless stated otherwise.
- Microbenchmarks exclude self-play and full validation; end-to-end numbers include them.
- Measurements taken on an earlier network, earlier data, or before TF32 was correctly disabled are not carried over into this README; where local notes disagree with a reproducible command, the command wins. Rows measured on the 12×192 shape are labelled as such.
- Training is CUDA-only by construction; CPU builds provide inference, search, the arena and the tests.
- RenjuNet's opening and overline rules differ from this engine's freestyle rules, so the human-game partition supervises **policy only**; value learning comes from self-play, either distilled from a trained engine or generated during RL.

## References and acknowledgements

- The reference implementation this project was rewritten from: [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku).
- AlphaZero method: Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* ([arXiv:1712.01815](https://arxiv.org/abs/1712.01815)).
- Training data: RenjuNet game records (`data/renjunet.xml`).
- Numerical-accuracy guidance: [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

Licensed under the [MIT License](LICENSE) © 2026 LangLang03.
