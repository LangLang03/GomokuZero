# Gomoku AlphaZero (C++)

A 15×15 **freestyle** Gomoku engine: players alternate moves, and five or more consecutive stones win. There are no Renju forbidden moves or swap openings. The C++17 engine implements the board, PUCT/MCTS, batched self-play and native CPU inference. Training uses the optional LibTorch/CUDA backend.

The current work addresses data quality, search correctness, training, checkpoint recovery and reproducible measurement. **Passing tests, reducing training loss or increasing parameter count does not establish stronger play.** Generalization and strength require independent validation and paired-opening matches. Historical throughput and strength claims from older networks are not presented as current results.

[中文说明](README_CN.md)

## Build and test

Requires CMake, a C++17 compiler and a threading library. CUDA training also needs an available NVIDIA GPU and a CUDA-enabled PyTorch installation in the selected Python environment. CMake reads LibTorch's path and C++ ABI from that installation.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure

# Native CPU inference, search and tests; no native CPU training
cmake -S . -B build-cpu -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j2
```

The four input planes encode own stones, opponent stones, the last move, and whether Black is to move. A 3×3 stem feeds the residual trunk and separate policy/value heads. Dense heads mix information across the board; a single convolutional feature's receptive field does not establish that the network cannot learn a tactic.

The default network has 12 residual blocks and 192 channels. Build other shapes with `-DGOMOK_NET_CHANNELS=N -DGOMOK_NET_RES_BLOCKS=N`; checkpoints from different shapes are incompatible. Smaller networks may run faster, but their generalization and playing strength must be measured independently.

A small network makes concurrency and memory checks practical without replacing full-network numerical or performance checks:

```sh
cmake -S . -B build-tsan -DGOMOK_ENABLE_CUDA=OFF \
  -DGOMOK_NET_CHANNELS=8 -DGOMOK_NET_RES_BLOCKS=1 \
  -DGOMOK_SANITIZER=thread -DBUILD_TESTING=ON
cmake --build build-tsan -j2
ctest --test-dir build-tsan --output-on-failure
# In another build directory, use -DGOMOK_SANITIZER=address,undefined for memory checks.
```

Tests cover board rules, tactical search, value signs, actual concurrent self-play, legal policy targets, dataset grouping and augmentation, backend numerics, optimizer recovery and Python tools. CUDA-specific checks require an active CUDA backend; successful CPU tests do not validate CUDA execution.

## Prepare and audit data

Regenerate records with game IDs and rule metadata:

```sh
python3 tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python3 tools/audit_dataset.py data/gomoku_v2 --state-overlap \
  --output analysis/gomoku_v2_dataset_audit.json
# Convert an existing compatible NPZ:
# python3 tools/npz_to_bin.py dataset.npz data/custom
```

Data tools require NumPy. The trainer reads:

| File | Contents |
|---|---|
| `<prefix>_states.bin` | float32 `[N,4,15,15]`, from the player-to-move perspective |
| `<prefix>_probs.bin` | float32 `[N,225]` legal policy distributions |
| `<prefix>_winners.bin` | float32 `[N]` outcomes in `[-1,1]`, from the player-to-move perspective |
| `<prefix>_game_ids.bin` | uint64 `[N]` game IDs |
| `<prefix>_meta.json` | Source/target rules and value supervision metadata |

Binary arrays start with a uint64 element count and use little-endian storage. The mmap loader checks dimensions, state encoding, legal policies, normalization, terminal positions and outcome consistency within each game.

The default converter selects source rule IDs `7/11/28`, excludes Classic Renju, explicit swap openings and invalid records, removes D4-equivalent complete games, and excludes the first five placements from policy targets. The remaining source opening/overline rules still differ from this engine. **These records therefore supervise policy only by default, with value weight zero.** Their recorded outcomes should not automatically become freestyle value labels. Legacy data without rule metadata also defaults to disabled value supervision.

Validation splits whole games, then removes held-out positions that are D4-equivalent to training positions. Regenerate missing game IDs where possible. `--infer-game-groups 1` is only for legacy records retaining complete game order; shuffled, truncated or mixed-rule data cannot be repaired by a random row split. `--val-fraction 0` disables independent validation and cannot establish generalization.

## Train and resume

Run `./build/gomoku_train --help` for all options. Unknown, duplicate and missing-value options fail explicitly, as do requested model loads that cannot be completed.

Supervised pretraining:

```sh
./build/gomoku_train train --mix data/gomoku_v2 \
  --pretrain 5 --games 0 --early-stop-patience 3 --seed 1 --tag pretrain
```

Every supervised epoch shuffles and traverses the complete training partition, including its partial final minibatch, with online D4 augmentation. Fixed training/validation probes and checkpoints run each epoch. An exhausted early-stopping patience remains effective on resume.

Start self-play from the checkpoint with the best observed supervised validation objective:

```sh
./build/gomoku_train train --init best_policy_pretrain_val_bin \
  --mix data/gomoku_v2 --pretrain 0 --games 100 \
  --early-stop-patience 3 --seed 1 --tag rl
```

Continue that run to batch 150:

```sh
./build/gomoku_train train --init best_policy_rl_bin \
  --mix data/gomoku_v2 --pretrain 0 --games 150 \
  --early-stop-patience 3 --seed 1 --tag rl
```

`train --games` is the **absolute final self-play batch index**, not an additional batch count or a number of individual games. A fresh `--games 100 --batch-games 64` run generates 6,400 games; resuming batch 100 to 150 generates only the next 50 batches. `--pretrain` is likewise an absolute supervised epoch target. `--epochs` means fresh optimizer minibatches after each self-play batch.

Recovery restores network/BN parameters, Adam state, replay samples, batch number, supervised progress and learning-rate horizons. It checks training options and dataset path, size and modification time. Keep the original options, seed and execution environment; concurrent scheduling is not guaranteed to reproduce bit for bit. Extending an existing self-play run preserves its original schedule and stays at the learning-rate floor afterwards. A supervised-only checkpoint initializes a new self-play schedule when that phase first begins. Legacy checkpoints without trainer state explicitly start with empty replay and cannot provide a complete continuation.

| Output | Meaning |
|---|---|
| `best_policy_<tag>_bin/` | **Latest** complete checkpoint; the historical `best` prefix does not mean strongest |
| `best_policy_<tag>_val_bin/` | Saved when supervised validation improves; not an arena strength champion |
| `runs/<tag>/metrics.csv` | Fixed train/validation probe losses, value error, top-1 accuracy and entropy |

Early stopping does not automatically restore the validation-best network. Select its `_val_bin` explicitly when continuing from it. A self-play-only run or a run without validation need not create that directory.

Common training options:

| Option | Default | Meaning |
|---|---:|---|
| `--batch-size` / `--epochs` / `--buffer` | 512 / 5 / 10000 | Optimizer batch size, updates per self-play batch, replay capacity |
| `--lr` / `--lr-final-fraction` | 0.002 / 0.1 | Base learning rate and cosine-decay floor fraction |
| `--lr-warmup` / `--pretrain-warmup` | 2 / 20 | Self-play warmup batches / supervised warmup steps |
| `--weight-decay` / `--grad-clip` | 0.0001 / 2 | Adam L2 regularization / gradient norm limit |
| `--mix-ratio` / `--mix-value-weight` | 0.5 / metadata | External sample fraction / relative external value confidence; zero disables it |
| `--value-loss-weight` / `--label-smooth` | 1 / 0.1 | Overall value loss coefficient / legal-action smoothing of external policies |
| `--augment` | 1 | Online rotations/reflections |
| `--precision` | `fp32` | `bf16` is explicit opt-in; validate its numerics and throughput separately |
| `--val-fraction` / `--val-samples` | 0.1 / 2048 | Game holdout fraction / validation probe limit; zero limit means all held-out samples |
| `--val-freq` / `--kl-freq` | 5 / 5 | Self-play validation / observational KL intervals; zero disables that probe |
| `--check-freq` | 50 | Self-play checkpoint interval; final state is also saved |

After each self-play batch, the trainer reports elapsed time and a remaining-time estimate. It smooths recent batch durations and adds upcoming arena matches separately, so the estimate settles as the run progresses. Supervised-pretraining progress logs also include an ETA.

Per-sample value weights are normalized by their batch sum. They express relative confidence, not the overall scale controlled by `--value-loss-weight`.

`--precision fp32` selects FP32 training tensors; CUDA convolutions/matrix multiplications still permit TF32 by default. Set `GOMOK_TF32=0` to disable TF32. `--precision bf16` explicitly enables training mixed precision; neither a speedup nor unchanged learning quality is assumed.

## Search, matches and evaluation

```sh
# Generates games and reports throughput; does not update the model
GOMOK_PROFILE=1 ./build/gomoku_train selfplay \
  --model best_policy_rl_bin --games 8 --batch 8 --playout 400 --seed 1

# Identical openings with model colours swapped
./build/gomoku_train arena --model best_policy_rl_bin \
  --baseline best_policy_pretrain_val_bin --games 40 --playout 400 \
  --opening-moves 4 --seed 1

./build/gomoku_train human --model best_policy_rl_bin --playout 400
```

`selfplay --games` and `arena --games` count **individual games**. Arena requires a positive even count. Human input is `r,c`, both in `0..14`. These commands support `--cpu`; native inference with the full default network may be slow.

MCTS evaluates the selected leaf in its current-player perspective, alternates value signs during backup and handles terminal positions exactly. Immediate wins use exact values without neural inference; forced blocks survive candidate pruning. Virtual loss applies only to concurrent paths within one tree, and collision retries do not consume completed simulation budget. Root noise is applied before pruning. Self-play policy targets use normalized raw visits; temperature controls move sampling only. A single simulation that only evaluates the root falls back to its prior. Finite budgets and candidate caps remain approximations and can miss longer tactics.

| Search option | Default |
|---|---:|
| `--playout` / training `--batch-games` | 400 / 64 |
| `--c-puct` / `--max-candidates` | 5 / 60 |
| `--temp-high` / `--temp-low` / `--temp-switch` | 1 / 0.5 / 30 |
| `--dirichlet-eps` / `--dirichlet-alpha` | 0.25 / 0.3 |
| `--virtual-loss` / `--unseen-smooth` | 8 / 0 |
| `--inference-batch` | 0, deriving a target of 32 |
| `--opening-moves` | 4; allowed 0..8, with zero selecting an empty board |
| `--seed` | `GOMOK_SEED`, otherwise 1 |

`--unseen-smooth` optionally smooths over legal moves. Arena and human play disable root noise and select the most visited action. Periodic training matches use `--eval-games N --eval-freq N --eval-playout N --baseline DIR`; they are disabled by default. The baseline must stay fixed and cannot be the checkpoint directory currently being written.

Self-play compacts the chosen subtree after each move and retains its completed visits and values. The next move searches only the remaining budget. Fresh root noise is drawn over all legal actions; the new root keeps previously visited children and adds its current top candidates. Reuse is enabled by default; `GOMOK_TREE_REUSE=0` disables it for a controlled comparison. New training checkpoints record this setting, while old checkpoints log the setting chosen on resume.

Set `GOMOK_PERSISTENT_WORKERS=1` to keep MCTS search workers and their scratch buffers alive across moves in one self-play batch. It showed no gain on an RTX 5060 and is off by default. For CUDA inference, `GOMOK_PINNED_INFERENCE=1` opts into reusable pinned host buffers and nonblocking copies. The inference API still synchronizes before returning each batch; this option also showed no gain on the RTX 5060 and remains off by default.

Arena reports individual W/L/D and `pair_wins/pair_losses/pair_draws`. The candidate wins an opening pair when its combined score over both colours exceeds 1, loses below 1, and ties at 1. Statistical comparisons should use opening pairs as their unit; small samples still cannot establish stable strength differences.

Freestyle Gomoku has first-player bias. Paired openings, colour-specific results and sufficient matches help detect it. Neither 50% nor 100% in a small match alone proves equal or superior strength. Falling training loss with worsening validation loss and a growing gap suggests overfitting. Failure of both losses to improve may reflect data, optimization or insufficient training, and cannot be diagnosed from model size alone.

## Export and measurement

```sh
./build/gomoku_train export --model best_policy_rl_bin --output model_export
python3 tools/export_onnx.py --bin-dir model_export model.onnx
# ONNX export requires PyTorch, onnx and onnxruntime.
```

To start the ONNX web server with GPU inference on a CUDA-enabled host:

```sh
python3 tools/export_onnx.py --bin-dir best_policy_fromzero_20260926_bin build-training/fromzero_20260926.onnx
python3 -c 'import onnxruntime as ort; print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
python3 tools/server_onnx.py --model build-training/fromzero_20260926.onnx --playout 400 --device cuda --host 127.0.0.1 --port 8000
```

The server prints the selected ONNX Runtime provider. Open `http://127.0.0.1:8000` in a browser. Explicit `--device cuda` fails if the CUDA provider/device is unavailable; only `--device auto` falls back to CPU.

Training checkpoints contain raw network/BN parameters, optimizer and trainer state. Inference exports contain convolutions with BN folded into their weights. Such exports support play or a new weight initialization, but cannot faithfully resume the original optimizer/training process. ONNX uses input `[B,4,15,15]` and outputs log-policy `[B,225]` and value `[B,1]`.

`./build/gomoku_diag_capacity <weights-dir> [states]` reports the loaded network's shape and outputs on random legal positions. These probes can reveal output anomalies, but cannot replace validation on real records or establish overfitting/underfitting.

Measure performance with the same model, shape, hardware, precision, batch and search budget. `gomoku_backend_benchmark [batch] [steps] [weights-dir]` measures warmed-up inference/training throughput. Self-play measurements must also report actual network evaluations, cache hits, moves and games. Kernel throughput improvements alone do not prove faster end-to-end training or learning.

One real-data comparison is complete: the same 192×12 network, seed, batch 128 and three full supervised epochs, evaluating all 12,188 held-out game positions after removing D4 overlap with training. Commands and evidence are in the [training and acceptance report](analysis/07_训练重构与验收.md).

| Metric | FP32, permitting TF32 | BF16 |
|---|---:|---:|
| Total supervised-stage time, including validation and saving | 460.418 s | 260.498 s |
| Final validation policy cross-entropy | 2.010746 | 2.010494 |
| Final validation top-1 | 44.7407% | 44.8146% |

BF16 completed this supervised stage about **1.77×** faster, with similar short-term policy quality in this single-seed experiment. This does not establish long-term convergence, stronger play or the same speedup for complete self-play training.

**The experimental version also exposed L2-driven collapse of the unsupervised value branch. Do not use `best_policy_verify_learning_20260926_*` checkpoints as final models or starting points for further self-play training.** The corrected code has now passed production recovery, value learning and save-failure checks. Without value supervision, the entire value branch skips gradients, decay and momentum; new output layers start at zero, and value updates use their own Adam counter. A fresh corrected model completed pretraining and 160 self-play games, then won 40/40 games against the previous batch-170 model at 400 playouts per move with 20 colour-swapped opening pairs. On 64 new-seed self-play games, after purging replay-equivalent D4 states, its game-averaged value MSE was 0.1359 versus the optimistic colour-only constant reference's 0.2877. These results apply to the reported protocols, not universal Elo or all opponents.

| Environment variable | Purpose |
|---|---|
| `GOMOK_PROFILE` | Self-play evaluation, cache, retry and inference-batch statistics |
| `GOMOK_MCTS_THREADS` | Search worker override, clamped to 1..32 |
| `GOMOK_LEAF_BATCH` | Per-game in-flight leaf override, clamped to 1..playout |
| `GOMOK_TREE_REUSE` | Self-play subtree reuse, enabled by default; set 0 to disable |
| `GOMOK_PERSISTENT_WORKERS` | Reuse self-play search threads and scratch buffers across moves; default 0 |
| `GOMOK_BATCH_WAIT_US` | Inference collection wait; default CUDA 100µs, CPU 0µs |
| `GOMOK_CACHE_ENTRIES` | Per-self-play-batch inference cache capacity, default 32768 |
| `GOMOK_SEED` | RNG seed; CLI `--seed` takes precedence |
| `GOMOK_TF32` | Training TF32 convolution/matrix multiplication switch, default 1 |
| `GOMOK_INFERENCE_TF32` | Inference TF32 switch, default 1; set to 0 to disable |
| `GOMOK_PINNED_INFERENCE` | Experimental pinned host staging and nonblocking CUDA copies; default 0 |
| `GOMOK_FUSED_ADAM` | Fused Adam update switch, default 1 |
| `GOMOK_BENCH_BF16` | Benchmark-only BF16 opt-in; set to 1 to enable |

Sources are in `src/`, tests in `tests/`, conversion/export tools in `tools/` and audit reports in `analysis/`. Generate datasets and training artifacts locally; large binary data, checkpoints and `runs/` are excluded from source control.

Trainer state uses format v2. Early v1 experiment states are rejected because their policy-only optimization could collapse the value branch. On the first transition from supervised-only training to self-play, search/replay settings can be chosen for the new phase; they become part of the resume contract after self-play starts.

The verified full checkpoint is `best_policy_gomoku_fast_bin`; its parameter/replay bytes match the evaluated model after a real no-update resume and save. See `analysis/final_model_20260926.json` for fingerprints.

```sh
./build-training/gomoku_train human --model best_policy_gomoku_fast_bin --playout 400
./build-training/gomoku_value_evaluation best_policy_gomoku_fast_bin 64 64 20260927
# Continue from self-play batch 10 to absolute batch 20:
./build-training/gomoku_train train --init best_policy_gomoku_fast_bin \
  --mix data/gomoku_v2 --pretrain 0 --games 20 --tag gomoku_fast \
  --batch-size 128 --precision bf16 --seed 1 --val-samples 0 \
  --playout 64 --batch-games 16 --epochs 20
```

The checked ONNX export is `build-experiment-artifact/gomoku_verified_20260926.onnx`. Re-evaluate and re-export after further training.
