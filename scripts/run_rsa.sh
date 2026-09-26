#!/usr/bin/env bash
# Sec. 3.3 - Representational Similarity Analysis for both tasks and both probed tokens.
#   bash scripts/run_rsa.sh [GPU] [JSON_DIR] [OUT_DIR]
# Produces OUT_DIR/{aavr,vaar}/{anchor,last}/rsa_{semantic,position,temporal}_all.npy
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
JSON_DIR="${2:-json_files/rsa}"
OUT_DIR="${3:-exp/rsa}"
CKPT_PATH="${CKPT_PATH:-checkpoints/video-SALMONN-2_plus_7B-merged}"
MODEL_BASE="${MODEL_BASE:-checkpoints/Qwen2.5-VL-7B-Instruct-Audio}"
EXTRA="${EXTRA:-}"   # e.g. EXTRA="--max_samples 100"

for TASK in aavr vaar; do
  for TOKEN in anchor last; do
    echo "=== RSA task=${TASK} token=${TOKEN} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" python rsa/rsa_${TOKEN}_token.py \
      --json_path "${JSON_DIR}/${TASK}.json" \
      --save_path "${OUT_DIR}/${TASK}/${TOKEN}" \
      --ckpt_path "${CKPT_PATH}" --model_base "${MODEL_BASE}" ${EXTRA}
  done
done
