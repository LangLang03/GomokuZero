#!/usr/bin/env bash
# 256x16 pipeline: human pretrain -> search-distilled pretrain -> value-selected RL.
#
# Unlike train_quality_5000.sh this runs on a wider trunk, so it also has to fix
# two things the old pipeline got wrong:
#   1. Checkpoint selection ignored the value head entirely (the mix dataset
#      declares value_weight 0, so `validate()` scored policy loss only and the
#      RL loop threw its score away). Here every RL validation scores
#      policy + select_value_weight * value on a held-out self-play probe.
#   2. The RL loop only ever saved "the last batch". Here the best scoring
#      snapshot is kept in best_policy_<tag>_val_bin.
#
# The 256x16 binary cannot load the 192x12 baseline (the trunk shape is a
# compile-time constant), so the arena baseline is this run's own supervised
# checkpoint: it measures what RL added, not absolute strength. Cross-shape
# comparisons go through the ONNX export instead.
#
# Stages are resumable and a lock file serializes runs: an earlier launch ran
# two pipelines at once, shared the GPU and the log, and the loser died on the
# dataset overwrite guard after 3000 games. A stage counts as done only when its
# trainer.meta reached the absolute epoch target or stopped on patience.
#
# Estimated cost on this machine (RTX 5060, ~1500 samples/s at 192x12):
#   stage A  2.2 s/game at playout 400 -> 3000 games ~110 min (51.7k samples)
#   B1 ~20 min   B2 ~5 min   B3 ~3 min/batch at 400 playouts on this GPU
#   BATCHES=250 -> ~12 h, BATCHES=500 -> ~24 h (includes periodic arena matches)
set -euo pipefail

cd "$(dirname "$0")/.."

BATCHES="${BATCHES:-250}"          # absolute RL batch target (self-play batches)
GAMES="${GAMES:-3000}"             # stage A: distillation games to generate
DISTILL_PLAYOUT="${DISTILL_PLAYOUT:-400}"
RL_PLAYOUT="${RL_PLAYOUT:-400}"
BATCH_GAMES="${BATCH_GAMES:-64}"
PRETRAIN_EPOCHS=6                  # human-only supervised epochs
DISTILL_EPOCHS=8                   # absolute target: 6 human + 2 distilled
PATIENCE=2
NET_TAG=256x16

seed_model="best_policy_quality5000_20260926_bin"
distill_prefix="data/selfplay_${NET_TAG}_p${DISTILL_PLAYOUT}"
pre_tag="q256_pre"
distill_tag="q256_distill"
rl_tag="q256_rl"

pre_out="best_policy_${pre_tag}_bin"
pre_best="best_policy_${pre_tag}_val_bin"
distill_out="best_policy_${distill_tag}_bin"
distill_best="best_policy_${distill_tag}_val_bin"
rl_out="best_policy_${rl_tag}_bin"
rl_best="best_policy_${rl_tag}_val_bin"

# A supervised stage is done only at its absolute epoch target, or when
# validation stopped improving and the saved patience is exhausted. The
# periodic checkpoint is written every epoch, the "best" one only on
# improvement, so either directory can prove completion.
stage_complete() {  # <checkpoint dir> <target epochs>
    local meta="$1/trainer.meta"
    [[ -f "$meta" ]] || return 1
    local completed stale
    completed=$(sed -n '2p' "$meta" | awk '{print $2}')
    stale=$(sed -n '3p' "$meta" | awk '{print $2}')
    [[ "$completed" =~ ^[0-9]+$ && "$stale" =~ ^[0-9]+$ ]] || return 1
    (( completed >= $2 || stale >= PATIENCE ))
}
preferred() {  # prefer the validation-selected checkpoint when it exists
    if [[ -f "$1/trainer.meta" ]]; then echo "$1"; else echo "$2"; fi
}
checkpoint_batch() {  # <checkpoint dir>; prints the saved batch index, or nothing
    [[ -f "$1/trainer.meta" ]] || return 0
    sed -n '2p' "$1/trainer.meta" | awk '{print $1}'
}

if [[ ! -d "$seed_model" ]]; then
    echo "missing $seed_model (the strongest same-family checkpoint to distil from)" >&2
    exit 1
fi
if [[ ! -f "data/gomoku_v2_states.bin" ]]; then
    echo "missing data/gomoku_v2 dataset" >&2
    exit 1
fi
# Every stage is resumable, including RL: the periodic checkpoint carries the
# optimizer, the replay buffer and the LR schedule, so an interrupted run
# continues from its last saved batch instead of restarting. Reuse a different
# rl_tag for a fresh RL experiment rather than deleting a recorded one.
# Two concurrent pipelines once shared the GPU and the log, and the loser died
# on the dataset overwrite guard after 3000 games.

mkdir -p runs
if command -v flock >/dev/null 2>&1; then
    exec 9>"runs/.train_${NET_TAG}.lock"
    if ! flock -n 9; then
        echo "another train_${NET_TAG}_quality.sh is already running; refusing to share the GPU" >&2
        exit 1
    fi
fi

python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable")'

export GOMOK_TF32=1
export GOMOK_INFERENCE_TF32=1
export GOMOK_FUSED_ADAM=1
export GOMOK_TREE_REUSE=1
export GOMOK_PERSISTENT_WORKERS=0
export GOMOK_PINNED_INFERENCE=0
export GOMOK_MCTS_THREADS=4

# ---------------------------------------------------------------------------
# A. Search-distilled self-play data from the strongest 192x12 checkpoint.
#    z is a real freestyle outcome, so the metadata sets value_weight 1 and the
#    trainer will use it for value supervision (human records stay policy-only).
# ---------------------------------------------------------------------------
if [[ -f "${distill_prefix}_states.bin" ]]; then
    echo "[resume] reusing distillation dataset $distill_prefix"
else
    ./build-training/gomoku_train selfplay \
        --model "$seed_model" --games "$GAMES" --batch 64 \
        --playout "$DISTILL_PLAYOUT" --output "$distill_prefix"
fi
python3 tools/audit_dataset.py "$distill_prefix" > "runs/${NET_TAG}_distill_audit.json"
head -12 "runs/${NET_TAG}_distill_audit.json"

# ---------------------------------------------------------------------------
# B1. Supervised pretrain from random weights: policy imitation of human games.
# ---------------------------------------------------------------------------
if stage_complete "$pre_out" "$PRETRAIN_EPOCHS" || stage_complete "$pre_best" "$PRETRAIN_EPOCHS"; then
    echo "[resume] supervised stage already complete"
else
    resume=()
    if [[ -f "$pre_out/trainer.meta" ]]; then
        resume=(--init "$pre_out")
        echo "[resume] continuing supervised pretrain from $pre_out"
    fi
    ./build-256x16/gomoku_train train \
        --mix data/gomoku_v2 --pretrain "$PRETRAIN_EPOCHS" --games 0 \
        --batch-size 256 --epochs 1 --precision bf16 --seed 1 \
        --val-samples 2048 --val-freq 1 --kl-freq 0 \
        --early-stop-patience "$PATIENCE" --tag "$pre_tag" ${resume[@]+"${resume[@]}"}
fi
pre_source=$(preferred "$pre_best" "$pre_out")
test -f "$pre_source/trainer.meta"

# ---------------------------------------------------------------------------
# B2. Distillation pretrain. --pretrain is absolute, so 8 = 6 human + 2 on the
#     distilled data. Same numeric options as B1: the resume contract is exact.
# ---------------------------------------------------------------------------
if stage_complete "$distill_out" "$DISTILL_EPOCHS" || stage_complete "$distill_best" "$DISTILL_EPOCHS"; then
    echo "[resume] distillation stage already complete"
else
    source_dir="$pre_source"
    if [[ -f "$distill_out/trainer.meta" ]]; then
        source_dir="$distill_out"
        echo "[resume] continuing distilled pretrain from $distill_out"
    fi
    ./build-256x16/gomoku_train train \
        --init "$source_dir" --mix "$distill_prefix" --allow-data-change --pretrain "$DISTILL_EPOCHS" --games 0 \
        --batch-size 256 --epochs 1 --precision bf16 --seed 1 \
        --val-samples 2048 --val-freq 1 --kl-freq 0 \
        --early-stop-patience "$PATIENCE" --tag "$distill_tag"
fi
distill_source=$(preferred "$distill_best" "$distill_out")
test -f "$distill_source/trainer.meta"

# ---------------------------------------------------------------------------
# B3. RL: self-play supplies value targets, human games stay the policy anchor.
#     --selfplay-val-every holds out every 10th batch (never entering the replay
#     buffer) so the value term in the selection score is honest, and
#     --select-value-weight 1 makes the score policy + value. Checkpoints are
#     written every 25 batches, so an interruption costs at most 25 batches.
# ---------------------------------------------------------------------------
rl_init="$distill_source"
rl_done=$(checkpoint_batch "$rl_out")
if [[ "$rl_done" =~ ^[0-9]+$ ]] && (( rl_done >= BATCHES )); then
    echo "[resume] RL already reached batch $rl_done/$BATCHES; nothing to do"
    exit 0
fi
if [[ "$rl_done" =~ ^[0-9]+$ ]]; then
    rl_init="$rl_out"
    echo "[resume] continuing RL from $rl_out (batch $rl_done/$BATCHES)"
fi
./build-256x16/gomoku_train train \
    --init "$rl_init" --mix data/gomoku_v2 --allow-data-change \
    --pretrain "$DISTILL_EPOCHS" --games "$BATCHES" --batch-games "$BATCH_GAMES" \
    --batch-size 256 --epochs 16 --precision bf16 --seed 1 \
    --playout "$RL_PLAYOUT" --buffer 50000 --mix-ratio 0.35 \
    --val-samples 2048 --val-freq 5 --kl-freq 5 \
    --early-stop-patience "$PATIENCE" \
    --selfplay-val-every 10 --select-value-weight 1 \
    --check-freq 25 \
    --eval-games 80 --eval-freq 50 --eval-playout "$RL_PLAYOUT" \
    --baseline "$distill_source" \
    --tag "$rl_tag"

# ---------------------------------------------------------------------------
# C. Deploy and verify. ONNX export works for any compiled shape; the Python
#    engine and server are shape agnostic.
# ---------------------------------------------------------------------------
# ./build-256x16/gomoku_train export --model "$rl_best" --output model_export_${NET_TAG}
# python3 tools/export_onnx.py --bin-dir model_export_${NET_TAG} ${NET_TAG}.onnx
# python3 tools/server_onnx.py --model ${NET_TAG}.onnx --playout 400 --device cuda --port 8001
