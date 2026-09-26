from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


META_PATH = Path("synthetic_animal_dataset/cma_four/meta.json")
IMAGE_DIR = Path("synthetic_animal_dataset/basic_asset/animal/image")

SLOT_KEYS = ("tl", "tr", "bl", "br")
SLOT_OFFSETS = {"tl": (0, 0), "tr": (1, 0), "bl": (0, 1), "br": (1, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render cma_four videos from meta.json"
    )
    parser.add_argument("--meta-path", type=Path, default=META_PATH)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override output directory (default: use meta['output_dir'])",
    )
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--meta-only", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gap-seconds", type=float, default=1.0,
                        help="Silence between utterances (the released videos use 1.0 s)")
    parser.add_argument("--start-combo", type=int, default=1)
    parser.add_argument("--end-combo", type=int, default=None)
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument(
        "--error-log", type=Path, default=None,
        help="Path to error log txt file (default: <out_dir>/errors.txt)",
    )
    return parser.parse_args()


# ── video helpers ────────────────────────────────────────────────────────

def load_video_frames(path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 25.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in video: {path}")
    return frames, float(fps)


def get_video_props(path: Path) -> Tuple[int, int, float]:
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
    return w, h, fps


def resample_frames(frames: List[np.ndarray], src_fps: float, dst_fps: float) -> List[np.ndarray]:
    if abs(src_fps - dst_fps) < 1e-6:
        return frames
    duration = len(frames) / src_fps
    target_len = max(1, int(round(duration * dst_fps)))
    idx = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[i] for i in idx]


def fit_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# ── rendering helpers ────────────────────────────────────────────────────

def render_canvas(
    layout: Dict[str, str],
    active_animal: str,
    active_frame: np.ndarray,
    sleep_by_animal: Dict[str, np.ndarray],
) -> np.ndarray:
    tl = active_frame if layout["tl"] == active_animal else sleep_by_animal[layout["tl"]]
    tr = active_frame if layout["tr"] == active_animal else sleep_by_animal[layout["tr"]]
    bl = active_frame if layout["bl"] == active_animal else sleep_by_animal[layout["bl"]]
    br = active_frame if layout["br"] == active_animal else sleep_by_animal[layout["br"]]
    top = np.hstack((tl, tr))
    bottom = np.hstack((bl, br))
    return np.vstack((top, bottom))


def render_all_sleep(layout: Dict[str, str], sleep_by_animal: Dict[str, np.ndarray]) -> np.ndarray:
    top = np.hstack((sleep_by_animal[layout["tl"]], sleep_by_animal[layout["tr"]]))
    bottom = np.hstack((sleep_by_animal[layout["bl"]], sleep_by_animal[layout["br"]]))
    return np.vstack((top, bottom))


# ── audio ────────────────────────────────────────────────────────────────

def add_audio_track_with_gap(
    output_video_path: Path,
    ordered_video_paths: List[Path],
    gap_seconds: float,
) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or run with --no-audio.")
    if len(ordered_video_paths) == 0:
        raise ValueError("ordered_video_paths must not be empty")

    tmp_out = output_video_path.with_name(output_video_path.stem + "__with_audio_tmp.mp4")
    gap = max(0.0, float(gap_seconds))

    cmd = [ffmpeg_bin, "-y", "-i", str(output_video_path)]
    for p in ordered_video_paths:
        cmd.extend(["-i", str(p)])

    filter_parts: List[str] = []
    concat_nodes: List[str] = []
    for i in range(len(ordered_video_paths)):
        in_idx = i + 1
        a_tag = f"a{i+1}"
        filter_parts.append(
            f"[{in_idx}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[{a_tag}]"
        )
        concat_nodes.append(f"[{a_tag}]")
        if i < len(ordered_video_paths) - 1 and gap > 0:
            g_tag = f"g{i+1}"
            filter_parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{gap}[{g_tag}]")
            concat_nodes.append(f"[{g_tag}]")

    concat_n = len(concat_nodes)
    filter_parts.append("".join(concat_nodes) + f"concat=n={concat_n}:v=0:a=1[a_out]")
    filter_complex = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[a_out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(tmp_out),
    ])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        tail = "\n".join(stderr[-20:])
        raise RuntimeError(f"ffmpeg audio mux failed for {output_video_path}:\n{tail}")
    tmp_out.replace(output_video_path)


# ── main ─────────────────────────────────────────────────────────────────

def process_one_record(
    rec: dict,
    out_dir: Path,
    image_dir: Path,
    gap_seconds: float,
    no_audio: bool,
    meta_only: bool,
    skip_existing: bool,
) -> str:
    """Process a single record. Returns 'rendered', 'skipped', or raises."""

    layout = rec["layout_animal"]
    speech_order = rec["speech_order_animal"]

    # Build animals_info from video1..video4
    animals_info: Dict[str, dict] = {}
    for k in ("video1", "video2", "video3", "video4"):
        if k in rec and isinstance(rec[k], dict):
            info = rec[k]
            animal = info.get("animal")
            if animal:
                animals_info[animal] = {
                    "animal": animal,
                    "country": info.get("country"),
                    "voice": info.get("voice"),
                    "video": info.get("video"),
                }

    # Handle duplicate animal names by using compound keys
    if len(animals_info) < 4:
        animals_info = {}
        layout_country = rec.get("layout_country", {})
        speech_country = rec.get("speech_order_country", [])
        for k in ("video1", "video2", "video3", "video4"):
            if k in rec and isinstance(rec[k], dict):
                info = rec[k]
                ckey = f"{info['animal']}__{info['country']}"
                animals_info[ckey] = {
                    "animal": info["animal"],
                    "country": info.get("country"),
                    "voice": info.get("voice"),
                    "video": info.get("video"),
                }
        layout = {
            slot: f"{layout[slot]}__{layout_country.get(slot, '')}"
            for slot in layout
        }
        speech_order = [
            f"{a}__{speech_country[i]}" for i, a in enumerate(speech_order)
        ]

    if len(animals_info) < 4:
        raise ValueError(f"combo_id={rec.get('combo_id')} has insufficient animal info")

    for k in SLOT_KEYS:
        if k not in layout:
            raise KeyError(f"Missing layout key '{k}' in combo_id={rec['combo_id']}")

    output_name = rec.get("output_name", f"combo_{int(rec['combo_id']):06d}.mp4")
    output_path = out_dir / output_name

    if skip_existing and output_path.exists() and not meta_only:
        return "skipped"

    animal_video_paths: Dict[str, Path] = {}
    animal_sleep_paths: Dict[str, Path] = {}
    for key, info in animals_info.items():
        vpath = Path(info["video"])
        spath = image_dir / f"{info['animal']}_back.png"
        if not vpath.exists():
            raise FileNotFoundError(f"Missing source video: {vpath}")
        if not spath.exists():
            raise FileNotFoundError(f"Missing sleep image: {spath}")
        animal_video_paths[key] = vpath
        animal_sleep_paths[key] = spath

    if meta_only:
        return "rendered"

    # Load all frames
    frames_by_animal: Dict[str, List[np.ndarray]] = {}
    fps_by_animal: Dict[str, float] = {}
    for animal, p in animal_video_paths.items():
        frames, fps = load_video_frames(p)
        frames_by_animal[animal] = frames
        fps_by_animal[animal] = fps

    out_fps = min(fps_by_animal.values())
    for animal in list(frames_by_animal.keys()):
        frames_by_animal[animal] = resample_frames(
            frames_by_animal[animal], fps_by_animal[animal], out_fps
        )

    slot_w = max(f[0].shape[1] for f in frames_by_animal.values())
    slot_h = max(f[0].shape[0] for f in frames_by_animal.values())
    for animal in list(frames_by_animal.keys()):
        frames_by_animal[animal] = [fit_size(fr, slot_w, slot_h) for fr in frames_by_animal[animal]]

    sleep_by_animal: Dict[str, np.ndarray] = {}
    for animal, p in animal_sleep_paths.items():
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load sleep image: {p}")
        sleep_by_animal[animal] = fit_size(img, slot_w, slot_h)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (slot_w * 2, slot_h * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")

    try:
        gap_frames = max(0, int(round(gap_seconds * out_fps)))
        all_sleep = render_all_sleep(layout, sleep_by_animal)

        for stage_idx, active_animal in enumerate(speech_order):
            active_frames = frames_by_animal[active_animal]
            for fr in active_frames:
                canvas = render_canvas(layout, active_animal, fr, sleep_by_animal)
                writer.write(canvas)

            if stage_idx < len(speech_order) - 1 and gap_frames > 0:
                for _ in range(gap_frames):
                    writer.write(all_sleep.copy())
    finally:
        writer.release()

    if not no_audio:
        ordered_video_paths = [animal_video_paths[a] for a in speech_order]
        add_audio_track_with_gap(output_path, ordered_video_paths, gap_seconds)

    return "rendered"


def main() -> None:
    args = parse_args()

    with args.meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    records = meta.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"No valid records in {args.meta_path}")

    out_dir = args.out_dir if args.out_dir is not None else Path(meta["output_dir"])
    if not args.meta_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = args.error_log if args.error_log is not None else (out_dir / "errors.txt")

    # Filter records by combo_id
    indexed_records = []
    for rec in records:
        combo_id = int(rec["combo_id"])
        if combo_id < args.start_combo:
            continue
        if args.end_combo is not None and combo_id > args.end_combo:
            continue
        indexed_records.append(rec)

    if args.max_combos is not None:
        indexed_records = indexed_records[: args.max_combos]

    rendered = 0
    skipped = 0
    errors = 0
    error_lines: List[str] = []

    for rec in tqdm(indexed_records, desc="Rendering combos"):
        try:
            result = process_one_record(
                rec, out_dir, args.image_dir,
                args.gap_seconds, args.no_audio,
                args.meta_only, args.skip_existing,
            )
            if result == "skipped":
                skipped += 1
            else:
                rendered += 1
        except Exception as e:
            errors += 1
            combo_id = rec.get("combo_id", "?")
            output_name = rec.get("output_name", "?")
            tb = traceback.format_exc()
            error_msg = (
                f"[{datetime.now().isoformat()}] "
                f"combo_id={combo_id} output_name={output_name}\n"
                f"  Error: {e}\n"
                f"  Traceback:\n{tb}\n"
            )
            error_lines.append(error_msg)
            tqdm.write(f"ERROR combo_id={combo_id} ({output_name}): {e}")

    # Write error log
    if error_lines:
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"CMA Four rendering errors — {datetime.now().isoformat()}\n")
            f.write(f"Total errors: {errors}\n")
            f.write("=" * 80 + "\n\n")
            for line in error_lines:
                f.write(line)
        print(f"Error log written to: {error_log_path}")

    print(
        f"Done. selected={len(indexed_records)}, rendered={rendered}, "
        f"skipped_existing={skipped}, errors={errors}, meta_only={args.meta_only}"
    )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
