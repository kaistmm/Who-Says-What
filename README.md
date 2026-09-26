<div align="center">

# Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual LLMs

[![Paper](https://img.shields.io/badge/OpenReview-M0fBuZdipl-b31b1b.svg)](https://openreview.net/forum?id=M0fBuZdipl)

*Official implementation of* **"Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual LLMs"** *(NeurIPS 2026)*

</div>

---

## ✨ Overview

We show that Audio-Visual LLMs resolve *"who says what"* through **symbolic IDs**: utterances are tracked by their **Temporal ID** (speaking order) and speakers by their **Position ID** (screen location), and trimodal binding is a three-stage process — *anchor ID retrieval → target ID selection → feature retrieval*.
This repository contains the two mechanistic analyses of Sec. 3 for **video-SALMONN 2+ (7B)**:

- **Representational Similarity Analysis (Sec. 3.3)** — which symbolic ID each layer encodes at the anchor-attribute token and at the last prompt token.
- **Causal Mediation Analysis (Sec. 3.4)** — layer-window activation patching between an original and a manipulated context (temporal ID / position ID / semantic content).

Both use two tasks on a synthetic four-animal dataset: **AAVR** (audio anchor → visual target, `aavr`) and **VAAR** (visual anchor → audio target, `vaar`).
Hidden states are read from every layer's attention output before `o_proj` (`self_attn.attn_out_pre_proj`, eager attention).

---

## 🛠️ 1. Environment

```bash
conda create -n bind python=3.10 -y && conda activate bind
pip install -r requirements.txt      # torch 2.7.1 / transformers 4.51.3 / flash-attn / liger-kernel
```

**Checkpoint.** video-SALMONN 2+ is a LoRA adapter on an audio-augmented Qwen2.5-VL-7B-Instruct; the scripts expect the merged model.

```bash
# base   : Qwen2.5-VL-7B-Instruct + Whisper, built with gen_audio_model.py of https://github.com/bytedance/video-SALMONN-2
# adapter: https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B
python tools/merge_lora.py \
    --model_base checkpoints/Qwen2.5-VL-7B-Instruct-Audio \
    --lora_ckpt  checkpoints/video-SALMONN-2_plus_7B \
    --out        checkpoints/video-SALMONN-2_plus_7B-merged
```

All scripts default to these two `checkpoints/` paths (`--ckpt_path`, `--model_base`; env `CKPT_PATH` / `MODEL_BASE` for the shell scripts) and are run from the repository root.

---

## 📦 2. Dataset

Synthetic 2×2 four-animal videos: each animal speaks one country name in turn (1 s pause), so every event has a layout slot (Position ID) and a speaking order (Temporal ID).

```
synthetic_animal_dataset/
├── four_high_entropy/{meta.json, video/, audio/}   800 videos  (RSA)       -> json_files/rsa/{aavr,vaar}.json
└── cma_four/{meta.json, videos/, audios/}          320 base videos × 12 manipulated versions (CMA)
                                                   -> json_files/cma/{aavr,vaar}_{temporal,position,semantic}.json
```

- `json_files/` holds the exact json files used in the paper (`*_unprimed.json` = same rows with the unprimed prompt).
- `samples/` + `json_files/samples/` bundle a **10-row subset** of every json with its videos/wavs (~390 MB) for a quick run.
- To rebuild from the source clips: `data_gen/make_four_high_entropy_meta.py` → `render_four_from_meta.py` / `render_cma_four_from_meta.py` → `extract_mp4_to_wav.py` → `build_rsa_json.py` / `build_cma_json.py` (see each `--help`).

---

## 🔍 3. Representational Similarity Analysis (Sec. 3.3)

Per layer, the cosine-similarity RSM of the recorded hidden states is correlated (Pearson r) with hypothesis RSMs built from `temporal_id`, `position_id` and `semantic_id` (the target attribute).

```bash
# anchor-attribute token (Fig. 2a/2b) and last prompt token (Fig. 2c/2d), for aavr and vaar
python rsa/rsa_anchor_token.py --json_path json_files/rsa/aavr.json --save_path exp/rsa/aavr/anchor
python rsa/rsa_last_token.py   --json_path json_files/rsa/aavr.json --save_path exp/rsa/aavr/last
#   or all four runs:  bash scripts/run_rsa.sh <GPU>

python rsa/plot_rsa.py --result_dir exp/rsa/aavr/anchor --output figures/rsa1.pdf
```

Outputs: `rsa_{semantic,position,temporal}_all.npy` (28 layer scores), `rsa_summary.json`.

---

## 🧪 4. Causal Mediation Analysis (Sec. 3.4)

Each row pairs an original context `video` with a manipulated context `c2_video` under the same prompt; `base_ans` is the original answer y1 and `causal_ans` the expected answer y1\* after patching.
The attention output of `c2` at the probed token is patched into the `c1` run for every sliding window of `--window_size` layers, and the Causal Mediation Score
`s = (M(c1*)[y1*] − M(c1*)[y1]) − (M(c1)[y1*] − M(c1)[y1])` is stored per window as `logits_diff_change`.

```bash
python cma/cma_window.py \
    --json_path json_files/cma/aavr_temporal.json \
    --save_path exp/cma/aavr/temporal/anchor \
    --replace_mode target_animal
#   or all 2 tasks × 3 manipulations × 2 tokens:  bash scripts/run_cma.sh <GPU>

python cma/plot_cma.py \
    exp/cma/aavr/temporal/anchor/aavr_temporal_cma_results.jsonl \
    exp/cma/aavr/position/anchor/aavr_position_cma_results.jsonl \
    exp/cma/aavr/semantic/anchor/aavr_semantic_cma_results.jsonl \
    --output-path figures/cma1.pdf
```

**Arguments**

- `--replace_mode` — `target_animal` (anchor-attribute token, Fig. 4a/4b) or `last_token` (last prompt token, Fig. 4c/4d).
- `--window_size` — layers patched together; the paper uses `5` (24 windows over 28 layers, `layer` = window centre).
- `--patch_batch_size` — keep `1` (paper setting); larger values batch the patched runs but exceed 48 GB.

`bash scripts/plot_figures.sh` renders `rsa1-4.pdf` and `cma1-4.pdf` from finished runs.

---

## 🚀 Quick start on the bundled samples

```bash
bash scripts/run_sample_demo.sh <GPU>        # RSA (10 samples) + CMA (3 samples/config) + figures, ~1.5 h on one A6000
```

Results go to `exp_samples/` and `figures_samples/`. `scripts/check_primed_accuracy.sh` reproduces the primed vs. unprimed accuracy check (App. A.1).

---

## 📝 Notes

- Runtime on one A6000: RSA ≈ 6 s/sample, CMA ≈ 95 s/sample (26 forward passes); the 12 CMA configurations are best spread over several GPUs.
- CM scores use single-token surface forms of the answer words; rows whose word has none (e.g. `giraffe` in the AAVR semantic set) get `null` and are excluded from the plots, as in the paper.
- Rows with a missing video/wav are skipped with a warning instead of being run without audio.
- `qwenvl/` is adapted from video-SALMONN 2 (Apache-2.0); `logic/nethook.py` from ROME (MIT).

---

## 📝 Citation
```bibtex
@inproceedings{
jung2026who,
title={Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual {LLM}s},
author={Anonymous},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=M0fBuZdipl}
}
```
