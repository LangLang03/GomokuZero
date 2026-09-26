#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pretrain_tag=quality5000_pre_20260926
train_tag=quality5000_20260926
pretrain_best="best_policy_${pretrain_tag}_val_bin"
train_output="best_policy_${train_tag}_bin"

if [[ -e "best_policy_${pretrain_tag}_bin" || -e "$pretrain_best" ||
      -e "$train_output" ]]; then
    echo "A quality5000 checkpoint already exists; refusing to overwrite a run started from scratch." >&2
    exit 1
fi

python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable in this session")'

export GOMOK_TF32=1
export GOMOK_INFERENCE_TF32=1
export GOMOK_FUSED_ADAM=1
export GOMOK_TREE_REUSE=1
export GOMOK_PERSISTENT_WORKERS=0
export GOMOK_PINNED_INFERENCE=0
export GOMOK_MCTS_THREADS=4

# Start with random weights. Keep the best independent-validation checkpoint
# rather than assuming the last supervised epoch has the best policy.
./build-training/gomoku_train train \
  --mix data/gomoku_v2 --pretrain 6 --games 0 \
  --batch-size 128 --precision bf16 --seed 1 --val-samples 0 \
  --early-stop-patience 2 --tag "$pretrain_tag"

test -f "$pretrain_best/trainer.meta"

# 250 self-play batches x 20 games = exactly 5000 games. The human-game
# dataset supplies policy supervision only; self-play supplies value targets.
./build-training/gomoku_train train \
  --init "$pretrain_best" --mix data/gomoku_v2 \
  --pretrain 0 --games 250 --batch-games 20 \
  --batch-size 128 --precision bf16 --seed 1 --val-samples 0 \
  --early-stop-patience 2 --playout 256 --epochs 24 \
  --buffer 20000 --mix-ratio 0.35 --check-freq 25 \
  --tag "$train_tag" \
  --eval-games 80 --eval-freq 50 --eval-playout 400 \
  --baseline best_policy_fromzero_20260926_bin
