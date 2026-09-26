#!/usr/bin/env bash
# Smoke test on the bundled 10-sample subset (samples/ + json_files/samples/).
#   bash scripts/run_sample_demo.sh [GPU]
# Runs RSA (2 tasks x 2 tokens, 10 samples), CMA (2 tasks x 3 manipulations x 2 tokens,
# CMA_SAMPLES rows each, default 3) and the plots.  About 1.5 h on one A6000 with the
# defaults (CMA costs ~95 s per sample); results land in exp_samples/ and figures_samples/.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
export CKPT_PATH="${CKPT_PATH:-checkpoints/video-SALMONN-2_plus_7B-merged}"
export MODEL_BASE="${MODEL_BASE:-checkpoints/Qwen2.5-VL-7B-Instruct-Audio}"
bash scripts/run_rsa.sh "${GPU}" json_files/samples exp_samples/rsa
EXTRA="--max_samples ${CMA_SAMPLES:-3}" bash scripts/run_cma.sh "${GPU}" json_files/samples exp_samples/cma
bash scripts/plot_figures.sh exp_samples/rsa exp_samples/cma figures_samples
