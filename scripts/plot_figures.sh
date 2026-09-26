#!/usr/bin/env bash
# Render the paper figures (Fig. 2: rsa1-4.pdf, Fig. 4: cma1-4.pdf) from finished runs.
#   bash scripts/plot_figures.sh [RSA_DIR] [CMA_DIR] [FIG_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
RSA_DIR="${1:-exp/rsa}"
CMA_DIR="${2:-exp/cma}"
FIG_DIR="${3:-figures}"
mkdir -p "${FIG_DIR}"

# Fig. 2 (a) AAVR anchor  (b) VAAR anchor  (c) AAVR last  (d) VAAR last
python rsa/plot_rsa.py --result_dir "${RSA_DIR}/aavr/anchor" --output "${FIG_DIR}/rsa1.pdf"
python rsa/plot_rsa.py --result_dir "${RSA_DIR}/vaar/anchor" --output "${FIG_DIR}/rsa2.pdf"
python rsa/plot_rsa.py --result_dir "${RSA_DIR}/aavr/last"   --output "${FIG_DIR}/rsa3.pdf"
python rsa/plot_rsa.py --result_dir "${RSA_DIR}/vaar/last"   --output "${FIG_DIR}/rsa4.pdf"

# Fig. 4 (a) AAVR anchor  (b) VAAR anchor  (c) AAVR last  (d) VAAR last
i=1
for TOKEN in anchor last; do
  for TASK in aavr vaar; do
    python cma/plot_cma.py \
      "${CMA_DIR}/${TASK}/temporal/${TOKEN}/${TASK}_temporal_cma_results.jsonl" \
      "${CMA_DIR}/${TASK}/position/${TOKEN}/${TASK}_position_cma_results.jsonl" \
      "${CMA_DIR}/${TASK}/semantic/${TOKEN}/${TASK}_semantic_cma_results.jsonl" \
      --output-path "${FIG_DIR}/cma${i}.pdf"
    i=$((i + 1))
  done
done
echo "figures written to ${FIG_DIR}/"
