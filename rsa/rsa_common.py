"""
Shared utilities for Representational Similarity Analysis (RSA), Sec. 3.3.

Pipeline
--------
1. Build video-SALMONN 2+ inputs with the same data pipeline that is used for
   inference (video frames + Whisper features + chat-template prompt).
2. Run one forward pass per sample and record, for every decoder layer, the
   attention output at the probed token(s).  The hook point is
   ``model.layers.{l}.self_attn.attn_out_pre_proj`` (an ``nn.Identity`` placed
   on the concatenated head outputs, shape ``[B, T, H, D]``, right before
   ``o_proj``).
3. For every layer, build the model RSM (cosine similarity between the recorded
   vectors of all samples) and correlate its upper triangle (Pearson r) with
   three hypothesis RSMs that encode, respectively, the Temporal ID, the
   Position ID and the semantic content of the *target* attribute.

Two probe positions are provided by the entry scripts:
    rsa_anchor_token.py  - the token(s) of the anchor attribute in the prompt
    rsa_last_token.py    - the last prompt token
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import AutoTokenizer, WhisperFeatureExtractor

from logic import nethook
from qwenvl.data.dataset import LazySupervisedDataset
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.data.rope2d import get_rope_index_25
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.train.argument import DataArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)
torch.set_grad_enabled(False)

NUM_LAYERS = 28
HOOK_LAYER_FMT = "model.layers.{}.self_attn.attn_out_pre_proj"
DEFAULT_CKPT = "checkpoints/video-SALMONN-2_plus_7B-merged"
DEFAULT_MODEL_BASE = "checkpoints/Qwen2.5-VL-7B-Instruct-Audio"
HYPOTHESES = ("semantic", "position", "temporal")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--json_path", type=str, required=True,
                        help="RSA json (e.g. json_files/rsa/aavr.json)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output directory for rsa_*_all.npy and quick-look plots")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT,
                        help="Merged video-SALMONN 2+ checkpoint (see tools/merge_lora.py)")
    parser.add_argument("--model_base", type=str, default=DEFAULT_MODEL_BASE,
                        help="Directory holding the tokenizer / image-processor config")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Randomly subsample this many rows (-1 = use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    # video / audio preprocessing (video-SALMONN 2+ evaluation defaults)
    parser.add_argument("--model_max_length", type=int, default=131072)
    parser.add_argument("--video_max_frames", type=int, default=768)
    parser.add_argument("--video_min_frames", type=int, default=32)
    parser.add_argument("--base_interval", type=float, default=0.5)
    parser.add_argument("--max_pixels", type=int, default=61250)
    parser.add_argument("--min_pixels", type=int, default=784)
    parser.add_argument("--video_max_frame_pixels", type=int, default=61250)
    parser.add_argument("--video_min_frame_pixels", type=int, default=784)
    return parser


# ------------------------------------------------------------------ #
#  Model / data pipeline
# ------------------------------------------------------------------ #

def load_model(args):
    print(f"Loading video-SALMONN 2+ from {args.ckpt_path} ...")
    # eager attention is required so that the hook point sees [B, T, H, D]
    model = video_SALMONN2_plus.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    return model


def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(
        args.model_base,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )


def build_dataset(args, tokenizer):
    """A LazySupervisedDataset instance used purely for its preprocessing methods."""
    data_args = DataArguments(
        dataset_use="__dummy__",
        video_max_frames=args.video_max_frames,
        video_min_frames=args.video_min_frames,
        base_interval=args.base_interval,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        video_max_frame_pixels=args.video_max_frame_pixels,
        video_min_frame_pixels=args.video_min_frame_pixels,
        run_test=True,
    )
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(args.model_base)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    dataset.tokenizer = tokenizer
    dataset.data_args = data_args
    dataset.list_data_dict = []
    dataset.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
    dataset.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
    dataset.model_type = data_args.model_type
    dataset.get_rope_index = get_rope_index_25
    dataset.data_args.image_processor.max_pixels = data_args.max_pixels
    dataset.data_args.image_processor.min_pixels = data_args.min_pixels
    dataset.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
    dataset.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
    dataset.modality = "av"
    dataset.mode = None
    return dataset


def build_inputs(dataset, sample_dict, device="cuda"):
    """Run one json row through the dataset pipeline and return model-ready tensors."""
    for key in ("video", "audio"):
        path = sample_dict.get(key)
        if path and not os.path.exists(path):
            # the data pipeline would silently drop a missing audio track; fail loudly instead
            raise FileNotFoundError(f"{key} file not found: {path}")
    orig = dataset.list_data_dict
    dataset.list_data_dict = [sample_dict]
    try:
        data = dataset._get_item(0)
    finally:
        dataset.list_data_dict = orig
    return {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}


# ------------------------------------------------------------------ #
#  JSON loading
# ------------------------------------------------------------------ #

def load_json_samples(json_path, max_samples=-1, seed=42):
    """Load RSA rows.  Each row is a flat dict (see README, "JSON formats")."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if 0 <= max_samples < len(data):
        random.seed(seed)
        data = random.sample(data, max_samples)

    samples = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict) or row.get("video") is None:
            continue
        samples.append({
            "id": str(idx),
            "salmon_dict": row,
            "target_animal": row["target_animal"],      # anchor attribute word in the prompt
            "base_video": row["video"],
            "semantic": int(row.get("semantic_id", -1)),
            "position": int(row.get("position_id", -1)),
            "temporal": int(row.get("temporal_id", -1)),
        })
    if not samples:
        raise ValueError(f"No valid samples in {json_path}")
    return samples


# ------------------------------------------------------------------ #
#  Token selection
# ------------------------------------------------------------------ #

def _find_subsequence(seq, sub):
    n = len(sub)
    if n == 0 or n > len(seq):
        return []
    for start in range(len(seq) - n + 1):
        if seq[start:start + n] == sub:
            return list(range(start, start + n))
    return []


def find_anchor_token_ids(prompt_ids, tokenizer, anchor_word):
    """Token indices of the anchor attribute word inside the prompt."""
    prompt_token_ids = prompt_ids.detach().cpu().tolist()
    cap = anchor_word[:1].upper() + anchor_word[1:]
    for cand in [anchor_word, f" {anchor_word}", cap, f" {cap}"]:
        sub = tokenizer.encode(cand, add_special_tokens=False)
        ids = _find_subsequence(prompt_token_ids, sub)
        if ids:
            return ids
    raise ValueError(f"anchor word not found in prompt: '{anchor_word}'")


def last_token_ids(prompt_ids, tokenizer=None, anchor_word=None):
    return [int(prompt_ids.shape[0] - 1)]


# ------------------------------------------------------------------ #
#  Activation collection
# ------------------------------------------------------------------ #

def collect_activations(data, model, tokenizer, dataset, device, token_selector):
    """
    For every sample, record the attention output at the selected token(s) of
    every layer.  If several tokens are selected (multi-token anchor word) their
    vectors are averaged.
    Returns {layer: [vec, ...]}, {layer: [meta, ...]}.
    """
    layer_vecs = defaultdict(list)
    layer_meta = defaultdict(list)
    layer_names = [HOOK_LAYER_FMT.format(l) for l in range(NUM_LAYERS)]
    layer_name_to_idx = {n: i for i, n in enumerate(layer_names)}

    for idx_, sample in enumerate(tqdm(data, desc="Collecting activations")):
        meta = {
            "target_animal": sample["target_animal"],
            "base_video": sample["base_video"],
            "semantic": sample["semantic"],
            "position": sample["position"],
            "temporal": sample["temporal"],
        }
        try:
            inputs = build_inputs(dataset, sample["salmon_dict"], device=device)
            token_ids = token_selector(inputs["input_ids"][0], tokenizer, sample["target_animal"])
        except Exception as e:  # noqa: BLE001
            print(f"  Skip {idx_}: {e}")
            continue

        captured = {}

        def save_hook(output, layer):
            h = output[0] if isinstance(output, tuple) else output
            if h.ndim == 4 and h.shape[1] > 1:          # prompt forward only
                li = layer_name_to_idx[layer]
                vecs = [h[0, t, :, :].reshape(-1) for t in token_ids if 0 <= t < h.shape[1]]
                if vecs:
                    captured[li] = torch.stack(vecs).mean(dim=0).detach().clone()
            return output

        with torch.no_grad(), nethook.TraceDict(model, layer_names, edit_output=save_hook):
            model.generate(
                **inputs,
                max_new_tokens=1,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

        for li, vec in captured.items():
            layer_vecs[li].append(vec.float().cpu())
            layer_meta[li].append(meta)

    return layer_vecs, layer_meta


# ------------------------------------------------------------------ #
#  RSA
# ------------------------------------------------------------------ #

def _pairwise_from_groups(groups):
    """Hypothesis RSM: 1 if two samples share the ID, 0 otherwise (0.5 if unknown)."""
    n = len(groups)
    rsm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] is None or groups[j] is None:
                rsm[i, j] = rsm[j, i] = 0.5
            else:
                rsm[i, j] = rsm[j, i] = 1.0 if groups[i] == groups[j] else 0.0
    np.fill_diagonal(rsm, 1.0)
    return rsm


def build_target_rsm(meta, key):
    ids = [m.get(key, -1) for m in meta]
    if key == "semantic" and not any(v >= 0 for v in ids):
        # fall back to the label string when no semantic_id was given
        groups = [m["target_animal"] for m in meta]
    else:
        groups = [int(v) if v >= 0 else None for v in ids]
    return _pairwise_from_groups(groups)


def build_model_rsm(vecs):
    mat = torch.stack(vecs).numpy()
    norms = np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
    mat_normed = mat / norms
    return mat_normed @ mat_normed.T


def upper_tri(rsm):
    idx = np.triu_indices(rsm.shape[0], k=1)
    return rsm[idx]


def rsa_score(model_rsm, target_rsm):
    v1 = upper_tri(model_rsm)
    v2 = upper_tri(target_rsm)
    if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
        return 0.0
    corr, _ = pearsonr(v1, v2)
    return corr


def compute_rsa_layer_vector(layer_vecs, layer_meta, key, n_layers=NUM_LAYERS):
    scores = np.full((n_layers,), np.nan)
    for layer in range(n_layers):
        vecs = layer_vecs.get(layer, [])
        meta = layer_meta.get(layer, [])
        if len(vecs) < 4:
            continue
        scores[layer] = rsa_score(build_model_rsm(vecs), build_target_rsm(meta, key))
    return scores


# ------------------------------------------------------------------ #
#  Quick-look plots (paper-style figure: plot_rsa.py)
# ------------------------------------------------------------------ #

def _auto_ylim(values, pad=0.05):
    finite_vals = np.asarray(values)[np.isfinite(values)]
    if finite_vals.size == 0:
        return -0.3, 0.6
    lo = max(-1.0, min(float(np.min(finite_vals)) - pad, -pad))
    hi = min(1.0, max(float(np.max(finite_vals)) + pad, pad))
    if hi <= lo:
        lo, hi = -0.3, 0.6
    return lo, hi


def plot_rsa_combined_lines(score_dict, output_path, title):
    palette = {"semantic": "#7F77DD", "position": "#EF9F27", "temporal": "#1D9E75"}
    fig, ax = plt.subplots(figsize=(12, 5.2))
    n_layers = None
    for name, scores in score_dict.items():
        n_layers = len(scores)
        ax.plot(np.arange(n_layers), scores, label=name.capitalize(),
                color=palette.get(name, "#666666"), marker="o", linewidth=1.8, markersize=4.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation (r)")
    ax.set_title(title)
    if n_layers is not None:
        ax.set_xticks(np.arange(n_layers))
    vmin, vmax = _auto_ylim(np.concatenate([np.asarray(v) for v in score_dict.values()]))
    ax.set_ylim(vmin, vmax)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"  Saved: {output_path}")


# ------------------------------------------------------------------ #
#  Driver
# ------------------------------------------------------------------ #

def run(args, token_selector, tag):
    model = load_model(args)
    tokenizer = load_tokenizer(args)
    dataset = build_dataset(args, tokenizer)

    data = load_json_samples(args.json_path, args.max_samples, args.seed)
    print(f"Loaded {len(data)} samples from {args.json_path}")

    layer_vecs, layer_meta = collect_activations(data, model, tokenizer, dataset, args.device, token_selector)
    if not layer_vecs:
        print("No activations collected, exiting.")
        return

    os.makedirs(args.save_path, exist_ok=True)
    score_dict = {}
    for name in HYPOTHESES:
        print(f"Computing RSA for '{name}' ...")
        scores = compute_rsa_layer_vector(layer_vecs, layer_meta, name)
        score_dict[name] = scores
        np.save(os.path.join(args.save_path, f"rsa_{name}_all.npy"), scores)
        print("  " + " ".join(f"{s:.3f}" for s in scores))

    n_used = len(layer_meta[0]) if 0 in layer_meta else 0
    with open(os.path.join(args.save_path, "rsa_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"json_path": args.json_path, "probe": tag, "n_samples": n_used,
                   **{k: [None if np.isnan(x) else float(x) for x in v] for k, v in score_dict.items()}},
                  f, indent=2)
    plot_rsa_combined_lines(score_dict, os.path.join(args.save_path, "rsa_combined_all.png"),
                            title=f"RSA @ {tag} ({os.path.basename(args.json_path)}, n={n_used})")
    print("\nALL DONE!")
