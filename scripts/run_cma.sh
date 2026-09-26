#!/usr/bin/env bash
# Sec. 3.4 - Causal Mediation Analysis: 2 tasks x 3 manipulations x 2 probed tokens.
#   bash scripts/run_cma.sh [GPU] [JSON_DIR] [OUT_DIR]
# Produces OUT_DIR/{aavr,vaar}/{temporal,position,semantic}/{anchor,last}/<name>_cma_results.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
JSON_DIR="${2:-json_files/cma}"
OUT_DIR="${3:-exp/cma}"
CKPT_PATH="${CKPT_PATH:-checkpoints/video-SALMONN-2_plus_7B-merged}"
MODEL_BASE="${MODEL_BASE:-checkpoints/Qwen2.5-VL-7B-Instruct-Audio}"
WINDOW_SIZE="${WINDOW_SIZE:-5}"
PATCH_BS="${PATCH_BS:-1}"   # 1 = paper setting (see cma/cma_window.py --help)
EXTRA="${EXTRA:-}"   # e.g. EXTRA="--max_samples 100"

for TASK in aavr vaar; do
  for MANIP in temporal position semantic; do
    for TOKEN in anchor last; do
      if [[ "${TOKEN}" == "anchor" ]]; then MODE=target_animal; else MODE=last_token; fi
      echo "=== CMA task=${TASK} manipulation=${MANIP} token=${TOKEN} ==="
      CUDA_VISIBLE_DEVICES="${GPU}" python cma/cma_window.py \
        --json_path "${JSON_DIR}/${TASK}_${MANIP}.json" \
        --save_path "${OUT_DIR}/${TASK}/${MANIP}/${TOKEN}" \
        --replace_mode "${MODE}" --window_size "${WINDOW_SIZE}" --patch_batch_size "${PATCH_BS}" \
        --ckpt_path "${CKPT_PATH}" --model_base "${MODEL_BASE}" ${EXTRA}
    done
  done
done
