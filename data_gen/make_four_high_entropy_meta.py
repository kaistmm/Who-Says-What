from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


DEFAULT_ANIMALS = ["monkey", "tiger", "rabbit", "panda"]
DEFAULT_VOICES = ["man1", "man2", "woman1", "woman2"]
DEFAULT_CONTENTS = ["canada", "egypt", "germany", "japan"]
SLOTS = ["tl", "tr", "bl", "br"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four_high_entropy meta.json with 16 variants (v00~v33), "
            "50 samples per variant, total 800 records."
        )
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("synthetic_animal_dataset/basic_asset/animal/video"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("synthetic_animal_dataset/basic_asset/animal/image"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("synthetic_animal_dataset/four_high_entropy"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-per-variant", type=int, default=50)
    return parser.parse_args()


def get_video_props(path: Path, cache: Dict[str, Tuple[int, int, float]]) -> Tuple[int, int, float]:
    key = str(path)
    if key in cache:
        return cache[key]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid video size: {path}")
    if fps <= 1e-6:
        fps = 25.0

    cache[key] = (w, h, fps)
    return cache[key]


def main() -> None:
    args = parse_args()

    animals: List[str] = DEFAULT_ANIMALS[:]
    voices: List[str] = DEFAULT_VOICES[:]
    contents: List[str] = DEFAULT_CONTENTS[:]
    variants = [f"v{i}{j}" for i in range(4) for j in range(4)]

    if len(animals) != 4 or len(voices) != 4 or len(contents) != 4:
        raise ValueError("This script expects exactly 4 animals, 4 voices, 4 speech contents.")

    out_root = args.out_root
    out_dir = out_root / "video"
    meta_path = out_root / "meta.json"
    combo_json_path = out_root / "four_high_entropy_combinations.json"

    out_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Asset checks
    for a in animals:
        sleep = args.image_dir / f"{a}_back.png"
        if not sleep.exists():
            raise FileNotFoundError(f"Missing sleep image: {sleep}")
    for a in animals:
        for c in contents:
            for v in voices:
                vp = args.video_dir / f"{a}_{c}_{v}.mp4"
                if not vp.exists():
                    raise FileNotFoundError(f"Missing source video: {vp}")

    rng = random.Random(args.seed)
    props_cache: Dict[str, Tuple[int, int, float]] = {}

    records = []
    combos = []
    combo_id = 1

    for variant in variants:
        pos_idx = int(variant[1])  # layout index for video1 animal
        ord_idx = int(variant[2])  # speech-order index for video1 animal
        fixed_slot = SLOTS[pos_idx]

        for _ in range(args.count_per_variant):
            ordered_animals = animals[:]
            rng.shuffle(ordered_animals)
            first_animal = ordered_animals[0]
            others = [a for a in ordered_animals if a != first_animal]

            shuffled_voices = voices[:]
            shuffled_contents = contents[:]
            rng.shuffle(shuffled_voices)
            rng.shuffle(shuffled_contents)

            videos = {}
            info_by_animal = {}
            ws, hs, fpss = [], [], []
            for i, animal in enumerate(ordered_animals, start=1):
                country = shuffled_contents[i - 1]
                voice = shuffled_voices[i - 1]
                video_path = args.video_dir / f"{animal}_{country}_{voice}.mp4"
                sleep_path = args.image_dir / f"{animal}_back.png"

                w, h, fps = get_video_props(video_path, props_cache)
                ws.append(w)
                hs.append(h)
                fpss.append(fps)

                info = {
                    "animal": animal,
                    "country": country,
                    "voice": voice,
                    "video": str(video_path).replace("\\", "/"),
                    "sleep_image": str(sleep_path).replace("\\", "/"),
                }
                videos[f"video{i}"] = info
                info_by_animal[animal] = info

            out_fps = min(fpss)
            slot_w = max(ws)
            slot_h = max(hs)

            rng.shuffle(others)
            layout_animal = {fixed_slot: first_animal}
            for slot, animal in zip([s for s in SLOTS if s != fixed_slot], others):
                layout_animal[slot] = animal

            rng.shuffle(others)
            speech_order_animal = [None] * 4
            speech_order_animal[ord_idx] = first_animal
            for idx, animal in zip([i for i in range(4) if i != ord_idx], others):
                speech_order_animal[idx] = animal

            layout_country = {slot: info_by_animal[a]["country"] for slot, a in layout_animal.items()}
            layout_voice = {slot: info_by_animal[a]["voice"] for slot, a in layout_animal.items()}
            speech_order_country = [info_by_animal[a]["country"] for a in speech_order_animal]
            speech_order_voice = [info_by_animal[a]["voice"] for a in speech_order_animal]

            output_name = f"combo_{combo_id:04d}_{variant}.mp4"
            output_path = out_dir / output_name

            rec = {
                "combo_id": combo_id,
                "pair_id": combo_id,
                "variant": variant,
                "variant_description": (
                    f"v{pos_idx}{ord_idx}: video1 animal fixed to layout index {pos_idx} ({fixed_slot}) "
                    f"and speech order index {ord_idx}"
                ),
                "video1": videos["video1"],
                "video2": videos["video2"],
                "video3": videos["video3"],
                "video4": videos["video4"],
                "layout_animal": layout_animal,
                "layout_country": layout_country,
                "layout_voice": layout_voice,
                "speech_order_animal": speech_order_animal,
                "speech_order_country": speech_order_country,
                "speech_order_voice": speech_order_voice,
                "output_name": output_name,
                "output_path": str(output_path).replace("\\", "/"),
                "fps": out_fps,
                "slot_size": {"width": slot_w, "height": slot_h},
                "output_size": {"width": slot_w * 2, "height": slot_h * 2},
                "audio_note": "Audio can be concatenated in speech_order_animal sequence.",
                "gap_seconds": 0.0,
                "skipped_existing": False,
            }
            records.append(rec)

            combos.append(
                {
                    "pair_id": combo_id,
                    "variant": variant,
                    "video1": videos["video1"],
                    "video2": videos["video2"],
                    "video3": videos["video3"],
                    "video4": videos["video4"],
                }
            )
            combo_id += 1

    meta = {
        "pair_csv": str(combo_json_path).replace("\\", "/"),
        "video_dir": str(args.video_dir).replace("\\", "/"),
        "image_dir": str(args.image_dir).replace("\\", "/"),
        "output_dir": str(out_dir).replace("\\", "/"),
        "variants": variants,
        "count_pairs": len(records),
        "count": len(records),
        "gap_seconds": 0.0,
        "with_audio": True,
        "start_combo": 1,
        "end_combo": None,
        "seed": args.seed,
        "country_rule": "4 distinct speech contents per video",
        "voice_rule": "4 distinct voices per video",
        "variant_rule": "variant vXY fixes video1 animal position to slot-index X and speech order index Y",
        "layout_index_map": {"0": "tl", "1": "tr", "2": "bl", "3": "br"},
        "records": records,
    }

    with combo_json_path.open("w", encoding="utf-8") as f:
        json.dump(combos, f, ensure_ascii=False, indent=2)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Validations
    expected = len(variants) * args.count_per_variant
    if len(records) != expected:
        raise AssertionError(f"Expected {expected} records, got {len(records)}")

    variant_counts = Counter(r["variant"] for r in records)
    if any(variant_counts[v] != args.count_per_variant for v in variants):
        raise AssertionError("Variant counts are not balanced.")

    for r in records:
        first = r["video1"]["animal"]
        px = int(r["variant"][1])
        ox = int(r["variant"][2])
        if r["layout_animal"][SLOTS[px]] != first:
            raise AssertionError("Variant position constraint violated.")
        if r["speech_order_animal"][ox] != first:
            raise AssertionError("Variant order constraint violated.")

    print(f"Done. records={len(records)}, variants={len(variants)}, each={args.count_per_variant}")
    print(f"Saved meta: {meta_path}")
    print(f"Saved combos: {combo_json_path}")


if __name__ == "__main__":
    main()
