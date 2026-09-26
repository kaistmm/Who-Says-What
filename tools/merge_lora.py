"""
Build the merged video-SALMONN 2+ (7B) checkpoint used by every analysis script.

video-SALMONN 2+ is distributed as a LoRA adapter (tsinghua-ee/video-SALMONN-2_plus_7B)
on top of the audio-augmented Qwen2.5-VL-7B-Instruct base ("Qwen2.5-VL-7B-Instruct-Audio",
produced with gen_audio_model.py of https://github.com/bytedance/video-SALMONN-2).
This script applies the adapter, merges it into the weights and saves a plain
``video_SALMONN2_plus`` checkpoint that can be loaded with ``from_pretrained``.
The logic mirrors the evaluation path of the official training script.
The merge itself runs on CPU, but a GPU must be visible: the import chain of the model file
(transformers -> deepspeed / triton kernels) fails without a CUDA driver.

Example
    python tools/merge_lora.py \
        --model_base checkpoints/Qwen2.5-VL-7B-Instruct-Audio \
        --lora_ckpt  checkpoints/video-SALMONN-2_plus_7B \
        --out        checkpoints/video-SALMONN-2_plus_7B-merged
"""
import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_base", type=str, required=True, help="Qwen2.5-VL-7B-Instruct-Audio directory")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="video-SALMONN-2_plus_7B adapter directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory of the merged checkpoint")
    args = parser.parse_args()

    from peft import PeftModel

    print(f"Loading base model from {args.model_base} ...")
    model = video_SALMONN2_plus.from_pretrained(
        args.model_base, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager",
    )
    # The adapter only targets the LLM; keep the Whisper encoder layers out of PEFT's reach.
    audio_layers = model.audio.layers
    del model.audio.layers
    print(f"Applying LoRA adapter from {args.lora_ckpt} ...")
    model = PeftModel.from_pretrained(model, args.lora_ckpt)
    model.model.audio.layers = audio_layers
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"Saved merged checkpoint to {args.out}")


if __name__ == "__main__":
    main()
