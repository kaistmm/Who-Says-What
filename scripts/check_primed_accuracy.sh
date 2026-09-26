#!/usr/bin/env bash
# Primed vs. unprimed accuracy of the binding task (Sec. 3.1 / App. A.1, Table 3).
#   bash scripts/check_primed_accuracy.sh [GPU] [JSON_DIR] [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
JSON_DIR="${2:-json_files/rsa}"
OUT_DIR="${3:-exp/infer}"
CKPT_PATH="${CKPT_PATH:-checkpoints/video-SALMONN-2_plus_7B-merged}"
MODEL_BASE="${MODEL_BASE:-checkpoints/Qwen2.5-VL-7B-Instruct-Audio}"

for NAME in aavr vaar aavr_unprimed vaar_unprimed; do
  CUDA_VISIBLE_DEVICES="${GPU}" python tools/infer.py \
    --json_path "${JSON_DIR}/${NAME}.json" --save_path "${OUT_DIR}/${NAME}" \
    --ckpt_path "${CKPT_PATH}" --model_base "${MODEL_BASE}" --max_new_tokens 8
  python tools/accuracy.py --jsonl "${OUT_DIR}/${NAME}/${NAME}_results.jsonl" --prefix-match \
    --animals canada,egypt,germany,japan
done
