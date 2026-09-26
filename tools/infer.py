# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plain generation with video-SALMONN 2+ (used to measure primed / unprimed
accuracy of the trimodal binding task, Sec. 3.1 and App. A.1).

Example
    python tools/infer.py --json_path json_files/rsa/aavr.json --save_path exp/infer/aavr
    python tools/accuracy.py --jsonl exp/infer/aavr/aavr_results.jsonl --prefix-match
"""
import os
import sys
import json
import argparse
import random
import time

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import DataArguments
from transformers import AutoTokenizer, WhisperFeatureExtractor

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


def apply_liger_kernel_to_qwen2_5_vl():
    from qwenvl.model import modeling_qwen2_5_vl
    modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP


def collate_fn(batch):
    return batch[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,
                        help="Salmon-format JSON file (.json)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save result jsonl")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/video-SALMONN-2_plus_7B-merged",
                        help="Merged video-SALMONN 2+ checkpoint (tools/merge_lora.py)")
    parser.add_argument("--model_base", type=str, default="checkpoints/Qwen2.5-VL-7B-Instruct-Audio",
                        help="Base model for tokenizer / image processor")
    parser.add_argument("--modality", type=str, default=None, choices=["a", "v", "av", None])
    parser.add_argument("--mode", type=str, default=None, choices=["people", "sports", None])
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=131072)
    parser.add_argument("--video_max_frames", type=int, default=768)
    parser.add_argument("--video_min_frames", type=int, default=32)
    parser.add_argument("--base_interval", type=float, default=0.5)
    parser.add_argument("--max_pixels", type=int, default=61250)
    parser.add_argument("--min_pixels", type=int, default=784)
    parser.add_argument("--video_max_frame_pixels", type=int, default=61250)
    parser.add_argument("--video_min_frame_pixels", type=int, default=784)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    apply_liger_kernel_to_qwen2_5_vl()

    data_args = DataArguments(
        dataset_use=args.json_path,
        video_max_frames=args.video_max_frames,
        video_min_frames=args.video_min_frames,
        base_interval=args.base_interval,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        video_max_frame_pixels=args.video_max_frame_pixels,
        video_min_frame_pixels=args.video_min_frame_pixels,
        run_test=True,
        do_sample=args.do_sample,
    )
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(args.model_base)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        modality=args.modality,
        mode=args.mode,
    )

    print(f"Loading model from {args.ckpt_path} ...")
    model = video_SALMONN2_plus.from_pretrained(
        args.ckpt_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    if args.no_audio:
        del model.audio
    model.cuda().eval()

    save_file = os.path.join(
        args.save_path,
        os.path.basename(args.json_path).replace(".json", "_results.jsonl"),
    )
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if os.path.exists(save_file):
        os.remove(save_file)
    print(f"Results will be saved to {save_file}")

    test_data = data_module["train_dataset"]
    loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    for idx, inputs in tqdm(enumerate(loader), total=len(loader)):
        if not inputs:
            continue
        res_i = {
            "id": idx,
            "video": inputs.pop("video", None),
            "audio_path": inputs.pop("audio", None),
            "text": inputs.pop("prompt", {}).get("value", ""),
            "label": inputs.pop("ref", None),
        }
        inputs_original = inputs.copy()
        inputs = {
            k: v.to(f"cuda:{torch.cuda.current_device()}")
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
            )
        output_trimmed = outputs.sequences[0, len(inputs["input_ids"][0]):]
        output_text = tokenizer.decode(
            output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        res_i["text_output"] = output_text
        gen_tokens, gen_logprobs, gen_top5 = [], [], []
        for tok_id, step_scores in zip(output_trimmed.tolist(), outputs.scores):
            lp = torch.log_softmax(step_scores[0].float(), dim=-1)
            gen_tokens.append(tokenizer.decode([tok_id]))
            gen_logprobs.append(round(lp[tok_id].item(), 6))
            top_v, top_i = lp.topk(5)
            gen_top5.append(
                [[tokenizer.decode([i]), round(v, 6)] for v, i in zip(top_v.tolist(), top_i.tolist())]
            )
        res_i["gen_tokens"] = gen_tokens
        res_i["gen_logprobs"] = gen_logprobs
        res_i["gen_top5"] = gen_top5
        res_i["text"] = inputs_original.get("conversations", [{}])[0].get("value", "")
        res_i["label_name"] = inputs_original.get("label_name", "")
        res_i["choices"] = inputs_original.get("choices", {})

        with open(save_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(res_i, ensure_ascii=False) + "\n")

    print(f"\nDone — {len(loader)} samples processed. Results: {save_file}")


if __name__ == "__main__":
    main()
