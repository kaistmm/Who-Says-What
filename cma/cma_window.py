"""
Causal Mediation Analysis (CMA) with layer-window activation patching, Sec. 3.4.

For every sample the json row gives an original context c1 (``video``), a
manipulated context c2 (``c2_video``) that shares the *same* text prompt, the
original answer y1 (``base_ans``) and the expected answer after patching y1*
(``causal_ans``).  Three forward passes are run:

    1. c2 run   : cache the attention output (hook point
                  ``model.layers.{l}.self_attn.attn_out_pre_proj``, [B, T, H, D])
                  at the probed token position for every layer.
    2. c1 run   : original logits.
    3. patched  : re-run c1 while overwriting the cached c2 activations at the
                  probed token for every layer inside a sliding layer window
                  (``--window_size`` consecutive layers, one window per row).

The Causal Mediation Score (Wang et al., 2022) is
    s = (M(c1*)[y1*] - M(c1*)[y1]) - (M(c1)[y1*] - M(c1)[y1])
and is stored per window in ``logits_diff_change`` of the output jsonl
(``layer`` = centre layer of the window).

Probed token (``--replace_mode``):
    target_animal : the anchor-attribute token(s) in the prompt  (Fig. 4a / 4b)
    last_token    : the last prompt token                        (Fig. 4c / 4d)

Example
    python cma/cma_window.py --json_path json_files/cma/aavr_temporal.json \
        --save_path exp/cma/aavr/temporal/anchor --replace_mode target_animal
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)
torch.set_grad_enabled(False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, WhisperFeatureExtractor  # noqa: E402

from logic import nethook  # noqa: E402
from qwenvl.data.dataset import LazySupervisedDataset  # noqa: E402
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast  # noqa: E402
from qwenvl.data.rope2d import get_rope_index_25  # noqa: E402
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus  # noqa: E402
from qwenvl.train.argument import DataArguments  # noqa: E402

NUM_LAYERS = 28
HOOK_LAYER_FMT = "model.layers.{}.self_attn.attn_out_pre_proj"
DEFAULT_CKPT = "checkpoints/video-SALMONN-2_plus_7B-merged"
DEFAULT_MODEL_BASE = "checkpoints/Qwen2.5-VL-7B-Instruct-Audio"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json_path", type=str, required=True, help="CMA json (json_files/cma/*.json)")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--model_base", type=str, default=DEFAULT_MODEL_BASE)

    parser.add_argument("--replace_mode", type=str, choices=["target_animal", "last_token"], default="target_animal",
                        help="Probed token: anchor-attribute token(s) or the last prompt token")
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=NUM_LAYERS - 1)
    parser.add_argument("--window_size", type=int, default=5,
                        help="Number of consecutive layers patched together (paper: 5)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Use only the first N rows (-1 = all)")
    parser.add_argument("--patch_batch_size", type=int, default=1,
                        help="Layer windows patched per forward pass. 1 = paper setting; larger values batch the "
                             "patched runs and need far more GPU memory than 48 GB (not used for the paper)")
    parser.add_argument("--seed", type=int, default=2025)
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
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Token-finding helpers
# ------------------------------------------------------------------ #

def _find_subsequence(seq, sub):
    n = len(sub)
    if n == 0 or n > len(seq):
        return []
    for start in range(len(seq) - n + 1):
        if seq[start:start + n] == sub:
            return list(range(start, start + n))
    return []


def get_replace_ids_target_animal(prompt_ids, tokenizer, target_animal):
    prompt_token_ids = prompt_ids.detach().cpu().tolist()
    for cand in [target_animal, f" {target_animal}"]:
        animal_token_ids = tokenizer.encode(cand, add_special_tokens=False)
        replace_ids = _find_subsequence(prompt_token_ids, animal_token_ids)
        if replace_ids:
            return replace_ids
    raise ValueError(f"Could not find target_animal token span: {target_animal}")


def get_replace_ids_last_token(prompt_ids):
    last_idx = int(prompt_ids.shape[0] - 1)
    if last_idx < 0:
        raise ValueError("Empty prompt_ids")
    return [last_idx]


# ------------------------------------------------------------------ #
#  Layer-window helpers
# ------------------------------------------------------------------ #

def build_layer_windows(layer_list, window_size):
    if not layer_list:
        raise ValueError("layer_list must not be empty")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    sorted_layers = sorted(int(x) for x in layer_list)
    if window_size >= len(sorted_layers):
        return [sorted_layers]
    return [sorted_layers[s:s + window_size] for s in range(len(sorted_layers) - window_size + 1)]


def resolve_layer_list(args):
    ls, le = args.layer_start, args.layer_end
    if not (0 <= ls <= le < NUM_LAYERS):
        raise ValueError(f"Invalid layer range [{ls}, {le}] for NUM_LAYERS={NUM_LAYERS}")
    return list(range(ls, le + 1))


# ------------------------------------------------------------------ #
#  Metric helpers
# ------------------------------------------------------------------ #

def _to_single_token_ids(tokenizer, token_str):
    ids_out = []
    for cand in [token_str, " " + token_str]:
        tok_ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(tok_ids) == 1 and tok_ids[0] not in ids_out:
            ids_out.append(tok_ids[0])
    return ids_out


def _metric_surface_forms(ans):
    """Surface forms whose single-token logits are max-pooled for an answer word."""
    if not isinstance(ans, str):
        return []
    lower_ans = ans.strip().lower()
    if not lower_ans:
        return []
    if lower_ans == "germany":
        return ["germany", "german", "Germany", "German"]
    if lower_ans == "italy":
        return ["italy", "Italy", "Italia", "italia"]
    return [lower_ans, lower_ans[:1].upper() + lower_ans[1:]]


def _metric_token_ids_with_case_variants(tokenizer, ans):
    ids = []
    for cand in _metric_surface_forms(ans):
        for tok_id in _to_single_token_ids(tokenizer, cand):
            if tok_id not in ids:
                ids.append(tok_id)
    return ids


def _last_step_logits(logits):
    if isinstance(logits, (tuple, list)):
        if len(logits) == 0:
            return None
        step_logits = logits[0]
        return step_logits[0] if step_logits.ndim == 2 else step_logits[0, 0]
    if torch.is_tensor(logits):
        if logits.ndim == 3:
            return logits[0, -1]
        if logits.ndim == 2:
            return logits[0]
    return None


def _max_logit_for_ids(step_logits, token_ids):
    vals = [step_logits[tid] for tid in token_ids if tid < step_logits.shape[0]]
    if not vals:
        return None
    return torch.stack(vals).max()


def cal_logit_prob_diff(logits, tokenizer, causal_ans, original_ans):
    causal_ids = _metric_token_ids_with_case_variants(tokenizer, causal_ans)
    original_ids = _metric_token_ids_with_case_variants(tokenizer, original_ans)
    if not causal_ids or not original_ids:
        return None
    step_logits = _last_step_logits(logits)
    if step_logits is None:
        return None
    causal_logit = _max_logit_for_ids(step_logits, causal_ids)
    original_logit = _max_logit_for_ids(step_logits, original_ids)
    if causal_logit is None or original_logit is None:
        return None
    return {
        "causal_logit": float(causal_logit.item()),
        "original_logit": float(original_logit.item()),
        "logits_diff": float((causal_logit - original_logit).item()),
    }


# ------------------------------------------------------------------ #
#  JSON loading
# ------------------------------------------------------------------ #

def _prompt_text_after_video_tag(human_value):
    if not isinstance(human_value, str):
        return ""
    if human_value.startswith("<video>\n"):
        return human_value[len("<video>\n"):]
    if human_value.startswith("<video>"):
        return human_value[len("<video>"):].lstrip("\n")
    return human_value


def _prompt_from_row(row):
    for turn in row.get("conversations", []):
        if isinstance(turn, dict) and turn.get("from") == "human":
            return _prompt_text_after_video_tag(turn.get("value", ""))
    return ""


def load_json_samples(json_path, max_samples=-1):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples >= 0:
        data = data[:max_samples]

    samples = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        required = ("video", "c2_video", "base_ans", "exp_ans", "causal_ans", "target_animal")
        if any(row.get(k) is None for k in required):
            continue
        samples.append({
            "id": idx,
            "base_ans": row["base_ans"],
            "exp_ans": row["exp_ans"],
            "causal_ans": row["causal_ans"],
            "target_animal": row["target_animal"],
            "base_video": row["video"],
            "base_audio": row.get("audio"),
            "exp_video": row["c2_video"],
            "exp_audio": row.get("c2_audio"),
            "prompt_text": _prompt_from_row(row),
            "label": row.get("label", row["base_ans"]),
        })
    if not samples:
        raise ValueError(f"No valid samples parsed from {json_path}")
    return samples


# ------------------------------------------------------------------ #
#  Input builder
# ------------------------------------------------------------------ #

def _video_to_audio_path(video_path):
    # synthetic_animal_dataset/<set>/video(s)/x.mp4 -> synthetic_animal_dataset/<set>/audio(s)/x.wav
    return video_path.replace("video", "audio").replace(".mp4", ".wav")


def _make_salmon_dict(video_path, prompt_text, label, audio_path=None):
    return {
        "video": video_path,
        "audio": audio_path or _video_to_audio_path(video_path),
        "use_audio": True,
        "conversations": [
            {"from": "human", "value": f"<video>\n{prompt_text}"},
            {"from": "gpt", "value": label},
        ],
    }


def build_inputs_from_salmon_dict(dataset, sample_dict, device="cuda"):
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


def build_dataset(args, tokenizer):
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


# ------------------------------------------------------------------ #
#  Windowed patching (all heads)
# ------------------------------------------------------------------ #

def trace_with_patch_windows(model, layer_windows, replace_id, base_inputs, exp_inputs, patch_batch_size=1):
    """
    Returns
        base_output           : generate() output of the un-patched c1 run
        combos                : [(layer_start, layer_end), ...] one per window
        patched_step_logits   : [n_windows, vocab] next-token logits of the patched runs
        patched_pred_token_ids: [n_windows] argmax token of the patched runs
    """
    layer_windows = [[int(l) for l in w] for w in layer_windows]
    combos = [(int(w[0]), int(w[-1])) for w in layer_windows if w]
    if not combos:
        raise ValueError("layer_windows must contain at least one target window.")

    unique_layers = sorted({l for w in layer_windows for l in w})
    layer_name_by_idx = {layer: HOOK_LAYER_FMT.format(layer) for layer in unique_layers}
    layer_idx_by_name = {v: k for k, v in layer_name_by_idx.items()}
    layers_to_track = list(layer_idx_by_name.keys())

    exp_cached = {}

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # --- 1. c2 run: cache activations at the probed token ---------------
    def save_hook(x, layer_name):
        h = untuple(x)
        if h.shape[1] == 1:
            return x
        layer = layer_idx_by_name.get(layer_name)
        if layer is None:
            return x
        for t in replace_id:
            if 0 <= t < h.shape[1]:
                exp_cached[(layer, t)] = h[0, t, :, :].detach().clone()
        return x

    gen_kwargs = dict(max_new_tokens=1, output_attentions=False, output_hidden_states=False,
                      return_dict_in_generate=True, output_logits=True)

    with torch.no_grad(), nethook.TraceDict(model, layers_to_track, edit_output=save_hook):
        _ = model.generate(**exp_inputs, **gen_kwargs)

    # --- 2. c1 run --------------------------------------------------------
    with torch.no_grad():
        base_output = model.generate(**base_inputs, **gen_kwargs)

    # --- 3. patched runs, one batch element per layer window -------------
    def repeat_batch_inputs(inputs, repeat_count):
        """Tile a batch-1 input dict `repeat_count` times along its batch axis."""
        if repeat_count == 1:
            return inputs
        repeated = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value) or value.ndim == 0:
                repeated[key] = value
            elif key == "position_ids":                      # [3, B, T]  (M-RoPE)
                repeated[key] = value.repeat_interleave(repeat_count, dim=1)
            elif key in ("pixel_values_videos", "pixel_values"):  # [n_patches, D], no batch axis
                repeated[key] = value.repeat(repeat_count, 1)
            elif value.shape[0] == 1:                        # input_ids, attention_mask, *_grid_thw, audio_feature
                repeated[key] = value.repeat_interleave(repeat_count, dim=0)
            else:
                raise ValueError(f"Do not know how to batch input '{key}' with shape {tuple(value.shape)}")
        return repeated

    patched_step_logits = []
    patched_pred_token_ids = []
    for chunk_start in range(0, len(combos), patch_batch_size):
        chunk_combos = combos[chunk_start:chunk_start + patch_batch_size]
        combo_indices_by_layer = {}
        for batch_idx, (layer_start, layer_end) in enumerate(chunk_combos):
            for layer in range(layer_start, layer_end + 1):
                combo_indices_by_layer.setdefault(layer_name_by_idx[layer], []).append((batch_idx, layer))

        chunk_inputs = repeat_batch_inputs(base_inputs, len(chunk_combos))

        def patch_hook(x, layer_name):
            h = untuple(x)
            if h.shape[1] == 1:
                return x
            for batch_idx, layer in combo_indices_by_layer.get(layer_name, []):
                for t in replace_id:
                    cached = exp_cached.get((layer, t))
                    if cached is None or not (0 <= t < h.shape[1]):
                        continue
                    h[batch_idx, t, :, :] = cached
            return x

        with torch.no_grad(), nethook.TraceDict(model, layers_to_track, edit_output=patch_hook):
            patched_output = model.generate(**chunk_inputs, **gen_kwargs)

        seq = patched_output["sequences"]
        prompt_len = chunk_inputs["input_ids"].shape[1]
        if seq.ndim == 2 and seq.shape[1] > prompt_len:
            next_ids = seq[:, prompt_len].detach().cpu()
        else:
            next_ids = torch.full((len(chunk_combos),), -1, dtype=torch.long)
        patched_pred_token_ids.append(next_ids)

        if isinstance(patched_output.logits, (tuple, list)):
            if len(patched_output.logits) == 0:
                continue
            step_logits = patched_output.logits[0]
            if step_logits.ndim == 3:
                step_logits = step_logits[:, 0, :]
        else:
            step_logits = patched_output.logits
            if step_logits.ndim == 3:
                step_logits = step_logits[:, -1, :]
        patched_step_logits.append(step_logits.detach().cpu())

    patched_step_logits = torch.cat(patched_step_logits, dim=0) if patched_step_logits \
        else torch.empty((0, 0), dtype=torch.float32)
    patched_pred_token_ids = torch.cat(patched_pred_token_ids, dim=0) if patched_pred_token_ids \
        else torch.empty((0,), dtype=torch.long)
    return base_output, combos, patched_step_logits, patched_pred_token_ids


# ------------------------------------------------------------------ #
#  Per-sample CMA
# ------------------------------------------------------------------ #

def run_one_cma_sample(idx, sample, model, tokenizer, dataset, layer_windows, args):
    base_ans = sample["base_ans"]
    exp_ans = sample["exp_ans"]
    causal_ans = sample["causal_ans"]
    target_animal = sample["target_animal"]

    base_salmon = _make_salmon_dict(sample["base_video"], sample["prompt_text"], sample["label"], sample["base_audio"])
    exp_salmon = _make_salmon_dict(sample["exp_video"], sample["prompt_text"], sample["label"], sample["exp_audio"])

    base_inputs = build_inputs_from_salmon_dict(dataset, base_salmon, device=args.device)
    exp_inputs = build_inputs_from_salmon_dict(dataset, exp_salmon, device=args.device)

    if args.replace_mode == "target_animal":
        replace_id = get_replace_ids_target_animal(exp_inputs["input_ids"][0], tokenizer, target_animal)
    else:
        replace_id = get_replace_ids_last_token(exp_inputs["input_ids"][0])

    base_output, combos, patched_step_logits, patched_pred_token_ids = trace_with_patch_windows(
        model=model,
        layer_windows=layer_windows,
        replace_id=replace_id,
        base_inputs=base_inputs,
        exp_inputs=exp_inputs,
        patch_batch_size=args.patch_batch_size,
    )

    base_answer = tokenizer.batch_decode(
        base_output["sequences"][0][base_inputs["input_ids"].shape[1]:].unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    base_logit_info = cal_logit_prob_diff(base_output.logits, tokenizer, causal_ans=causal_ans, original_ans=base_ans)
    causal_ans_ids = _metric_token_ids_with_case_variants(tokenizer, causal_ans)
    original_ans_ids = _metric_token_ids_with_case_variants(tokenizer, base_ans)

    base_causal_logit = base_logit_info["causal_logit"] if base_logit_info else None
    base_original_logit = base_logit_info["original_logit"] if base_logit_info else None
    base_logits_diff = base_logit_info["logits_diff"] if base_logit_info else None

    rows = []
    for combo_idx, (layer_start, layer_end) in enumerate(combos):
        logits_diff_change = None
        patched_base_answer = None
        patched_causal_logit = None
        patched_original_logit = None
        patched_logits_diff = None

        if combo_idx < patched_pred_token_ids.shape[0]:
            tok_id = int(patched_pred_token_ids[combo_idx].item())
            if tok_id >= 0:
                patched_base_answer = tokenizer.decode([tok_id], skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)

        if base_logits_diff is not None and causal_ans_ids and original_ans_ids \
                and combo_idx < patched_step_logits.shape[0]:
            patched_logits = patched_step_logits[combo_idx]
            causal_logit = _max_logit_for_ids(patched_logits, causal_ans_ids)
            original_logit = _max_logit_for_ids(patched_logits, original_ans_ids)
            if causal_logit is not None and original_logit is not None:
                patched_causal_logit = float(causal_logit.item())
                patched_original_logit = float(original_logit.item())
                patched_logits_diff = patched_causal_logit - patched_original_logit
                logits_diff_change = patched_logits_diff - base_logits_diff   # CM score

        rows.append({
            "id": sample.get("id", idx),
            "layer": int((layer_start + layer_end) // 2),
            "layer_window_index": int(combo_idx),
            "layer_window_start": int(layer_start),
            "layer_window_end": int(layer_end),
            "base_video": sample.get("base_video"),
            "exp_video": sample.get("exp_video"),
            "base_ans": base_ans,
            "exp_ans": exp_ans,
            "causal_ans": causal_ans,
            "target_animal": target_animal,
            "base_answer": base_answer,
            "patched_base_answer": patched_base_answer,
            "base_causal_logit": base_causal_logit,
            "base_original_logit": base_original_logit,
            "base_logits_diff": base_logits_diff,
            "patched_causal_logit": patched_causal_logit,
            "patched_original_logit": patched_original_logit,
            "patched_logits_diff": patched_logits_diff,
            "logits_diff_change": logits_diff_change,
            "replace_mode": args.replace_mode,
            "window_size": int(args.window_size),
            "replace_id": replace_id,
        })
    return rows


# ------------------------------------------------------------------ #
#  I/O helpers
# ------------------------------------------------------------------ #

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_window_summary(rows_by_window, output_dir):
    summary = {}
    for (layer_start, layer_end), rows in sorted(rows_by_window.items()):
        vals = [r["logits_diff_change"] for r in rows if r.get("logits_diff_change") is not None]
        if not vals:
            continue
        scores = torch.tensor(vals, dtype=torch.float32)
        torch.save(scores, os.path.join(output_dir, f"causal_scores_layer{layer_start}-{layer_end}.pt"))
        summary[f"{layer_start}-{layer_end}"] = {
            "n": int(scores.numel()),
            "mean": float(scores.mean().item()),
            "std": float(scores.std(unbiased=False).item()),
            "min": float(scores.min().item()),
            "max": float(scores.max().item()),
        }
    with open(os.path.join(output_dir, "cma_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Mean CM score per window:")
    for k, v in summary.items():
        print(f"  layers {k:>5}: {v['mean']:+.4f} (n={v['n']})")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError(f"--window_size must be > 0, got {args.window_size}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading video-SALMONN 2+ from {args.ckpt_path} ...")
    model = video_SALMONN2_plus.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",     # required: hook point expects [B, T, H, D]
    ).to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base, model_max_length=args.model_max_length, padding_side="right", use_fast=False,
    )
    dataset = build_dataset(args, tokenizer)

    data = load_json_samples(args.json_path, args.max_samples)
    layer_list = resolve_layer_list(args)
    layer_windows = build_layer_windows(layer_list, args.window_size)
    print(f"Loaded samples: {len(data)}")
    print(f"replace_mode={args.replace_mode} window_size={args.window_size}")
    print(f"layer_windows={[f'{w[0]}-{w[-1]}' for w in layer_windows]}")

    os.makedirs(args.save_path, exist_ok=True)
    save_result_filename = os.path.join(
        args.save_path, os.path.basename(args.json_path).replace(".json", "_cma_results.jsonl"))
    if os.path.exists(save_result_filename):
        os.remove(save_result_filename)

    rows_by_window = {}
    error_count = 0
    for idx_, sample in enumerate(tqdm(data, desc="CMA samples")):
        try:
            sample_rows = run_one_cma_sample(idx_, sample, model, tokenizer, dataset, layer_windows, args)
        except Exception as e:  # noqa: BLE001
            error_count += 1
            print(f"[WARN] Skip sample idx={idx_}, id={sample.get('id', idx_)} due to error: {e}")
            continue
        for row in sample_rows:
            append_jsonl(save_result_filename, row)
            rows_by_window.setdefault((row["layer_window_start"], row["layer_window_end"]), []).append(row)

    save_window_summary(rows_by_window, args.save_path)
    print(f"\nALL DONE (errors_skipped={error_count}) -> {save_result_filename}")


if __name__ == "__main__":
    main()
