<div align="center">

# Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2609.31193-b31b1b.svg)](https://arxiv.org/abs/2609.31193)

*Official implementation of* **"Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual LLMs"** *(NeurIPS 2026)*

</div>

---

## ✨ Overview

Resolving *"who says what"* in a multi-speaker video requires binding text, audio and vision. We show that Audio-Visual LLMs (AVLLMs) do this through **content-independent symbolic IDs** — an utterance is tracked by its **Temporal ID** (speaking order), a speaker by its **Position ID** (location on screen) — in three stages (Fig. 1a):

1. **Anchor ID retrieval** — the anchor attribute in the prompt is mapped to its symbolic ID (*"Japan"* → Temporal ID 3);
2. **Target ID selection** — that ID is converted into the symbolic ID of the target modality (Temporal ID 3 → Position ID 1);
3. **Feature retrieval** — the target ID is used to fetch the semantic content it points to (Position ID 1 → *"tiger"*).

This repository contains the two mechanistic analyses of Sec. 3 for **video-SALMONN 2+ (7B)**, the model reported in the main text:

- **Representational Similarity Analysis (Sec. 3.3)** — per layer, how strongly the hidden state at the *anchor-attribute token* and at the *last prompt token* correlates with each hypothesis space (Temporal ID / Position ID / semantic content).
- **Causal Mediation Analysis (Sec. 3.4)** — activation patching over sliding layer windows, from a manipulated context (temporal / position / semantic manipulation) into the original run, at the same two tokens.

Both run on two tasks over a synthetic four-animal dataset (Sec. 3.1): **AAVR** (`aavr`; acoustic anchor → visual target, e.g. *"Japan is said by the ___"* → *tiger*) and **VAAR** (`vaar`; visual anchor → acoustic target, e.g. *"… and the tiger says ___"* → *Japan*). Hidden states are the per-layer attention outputs before `o_proj` (`self_attn.attn_out_pre_proj`, eager attention).

## 🛠️ 1. Environment

```bash
conda create -n bind python=3.10 -y && conda activate bind
pip install -r requirements.txt      # torch 2.7.1 / transformers 4.51.3 / flash-attn / liger-kernel
```

**Checkpoint.** video-SALMONN 2+ is released as a LoRA adapter on an audio-augmented Qwen2.5-VL-7B-Instruct; the scripts expect the merged model.

```bash
# base   : Qwen2.5-VL-7B-Instruct + Whisper, built with gen_audio_model.py of https://github.com/bytedance/video-SALMONN-2
# adapter: https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B
python tools/merge_lora.py \
    --model_base checkpoints/Qwen2.5-VL-7B-Instruct-Audio \
    --lora_ckpt  checkpoints/video-SALMONN-2_plus_7B \
    --out        checkpoints/video-SALMONN-2_plus_7B-merged
```

All scripts are run from the repository root and default to the merged model (`--ckpt_path`) and the base model (`--model_base`) above; the shell scripts read them from `CKPT_PATH` / `MODEL_BASE`.

## 📦 2. Dataset

Synthetic single-shot videos: four animals in a 2×2 grid, each uttering one country name in turn (1 s pause), with placement and speaking order randomised per video. Every speech event is thus a tuple *(animal, Position ID, country, Temporal ID)* — the (v, p, a, t) of Sec. 3.1.

```
synthetic_animal_dataset/
├── four_high_entropy/{meta.json, video/, audio/}   800 videos  (RSA)       -> json_files/rsa/{aavr,vaar}.json
└── cma_four/{meta.json, videos/, audios/}          320 base videos × 12 manipulated versions (CMA)
                                                   -> json_files/cma/{aavr,vaar}_{temporal,position,semantic}.json
```

- `json_files/` holds the exact json files used in the paper. Prompts are contextually primed (Sec. 3.1 / App. A.1) so that the model binds correctly and the analyses condition on correct predictions; `*_unprimed.json` are the same rows with the unprimed prompt.
- `samples/` + `json_files/samples/` bundle a **10-row subset** of every json with its videos/wavs (~390 MB) for a quick run.
- To rebuild from the source clips: `data_gen/make_four_high_entropy_meta.py` → `render_four_from_meta.py` / `render_cma_four_from_meta.py` → `extract_mp4_to_wav.py` → `build_rsa_json.py` / `build_cma_json.py` (see each `--help`).

## 🔍 3. Representational Similarity Analysis (Sec. 3.3)

RSA tests whether a layer's representation is organised by a symbolic ID. For every layer, the cosine-similarity RSM of the recorded hidden states (pairwise over samples) is correlated (Pearson *r*) with three hypothesis RSMs built from the queried event's `temporal_id`, `position_id` and `semantic_id` (the target attribute). Expected pattern (Fig. 2): the anchor ID dominates at the anchor-attribute token in mid-to-late layers; at the last prompt token the target ID dominates in late layers and semantic content in the deepest layers.

```bash
# anchor-attribute token (Fig. 2a/2b) and last prompt token (Fig. 2c/2d), for aavr and vaar
python rsa/rsa_anchor_token.py --json_path json_files/rsa/aavr.json --save_path exp/rsa/aavr/anchor
python rsa/rsa_last_token.py   --json_path json_files/rsa/aavr.json --save_path exp/rsa/aavr/last
#   or all four runs:  bash scripts/run_rsa.sh <GPU>

python rsa/plot_rsa.py --result_dir exp/rsa/aavr/anchor --output figures/rsa1.pdf
```

Outputs: `rsa_{semantic,position,temporal}_all.npy` (one score per layer, 28 layers) and `rsa_summary.json`.

## 🧪 4. Causal Mediation Analysis (Sec. 3.4)

CMA tests whether the symbolic IDs are *causally used*. It involves three forward passes: the original context c₁ (answer y₁), a manipulated context c₂, and a patched run c₁\* in which the attention output of c₂ at the probed token is copied into the c₁ run, with expected answer y₁\* ≠ y₁. The manipulated context is derived from c₁ in one of three ways: swapping the speaking order of the queried event *k* with another event *j* (**temporal**), swapping their screen positions (**position**), or replacing the target attribute of *k* with one unseen in the video, e.g. a giraffe (**semantic**). Each json row pairs `video` (c₁) with `c2_video` (c₂) under the same prompt; `base_ans` is y₁ and `causal_ans` is y₁\*.

Patching is repeated for every sliding window of `--window_size` layers, and the Causal Mediation Score

s = (M(c₁\*)[y₁\*] − M(c₁\*)[y₁]) − (M(c₁)[y₁\*] − M(c₁)[y₁])

— how much patching shifts the logit gap toward y₁\* — is stored per window as `logits_diff_change`. Expected pattern (Fig. 4): the anchor-ID manipulation (temporal for AAVR, position for VAAR) peaks at the anchor-attribute token in mid-to-late layers; at the last prompt token the target-ID manipulation peaks in late layers and the semantic manipulation in the deepest layers.

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

- `--replace_mode` — `target_animal` patches at the anchor-attribute token (Fig. 4a/4b), `last_token` at the last prompt token (Fig. 4c/4d).
- `--window_size` — number of layers patched together; the paper uses `5` (24 windows over the 28 layers; `layer` in the results is the window centre).
- `--patch_batch_size` — keep `1` (paper setting); larger values batch the patched runs but exceed 48 GB.

`bash scripts/plot_figures.sh` renders `rsa1-4.pdf` and `cma1-4.pdf` from finished runs.

## 🚀 Quick start on the bundled samples

```bash
bash scripts/run_sample_demo.sh <GPU>        # RSA (10 samples) + CMA (3 samples/config) + figures, ~1.5 h on one A6000
```

Results go to `exp_samples/` and `figures_samples/`. `scripts/check_primed_accuracy.sh` reproduces the primed vs. unprimed accuracy check (App. A.1).

## 📝 Notes

- Runtime on one A6000: RSA ≈ 6 s/sample, CMA ≈ 95 s/sample (26 forward passes: original, manipulated and 24 patched windows); the 12 CMA configurations are best spread over several GPUs.
- CM scores use single-token surface forms of the answer words; rows whose word has none (e.g. `giraffe` in the AAVR semantic set) get `null` and are excluded from the plots, as in the paper.
- Rows with a missing video/wav are skipped with a warning instead of being run without audio.
- `qwenvl/` is adapted from [video-SALMONN 2](https://github.com/bytedance/video-SALMONN-2) (Apache-2.0); `logic/nethook.py` from [ROME](https://github.com/kmeng01/rome) (MIT).

## 📖 Citation

```bibtex
@inproceedings{
jung2026who,
title={Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual {LLM}s},
author={Jihoo Jung and Youngjoon Jang and Joon Son Chung},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=M0fBuZdipl}
}
```
