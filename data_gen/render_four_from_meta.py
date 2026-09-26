from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


META_PATH = Path("synthetic_animal_dataset/four_high_entropy/meta.json")
IMAGE_DIR = Path("synthetic_animal_dataset/basic_asset/animal/image")

SLOT_KEYS = ("tl", "tr", "bl", "br")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 2x2 four-animal videos (four_high_entropy) from meta.json"
    )
    parser.add_argument("--meta-path", type=Path, default=META_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output_dir in meta. If omitted, use meta['output_dir'].",
    )
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--meta-only", action="store_true", help="Do not render videos.")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable ffmpeg audio muxing. Video will contain no audio track.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rendering if output video already exists.",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0,
        help="Silence/pause duration between speech stages (the released videos use 1.0 s).",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Draw a green bounding box on the speaking animal's quadrant.",
    )
    parser.add_argument(
        "--bbox-thickness",
        type=int,
        default=None,
        help="BBox line width in pixels (default: scale from slot size).",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="Scale bbox size around slot center (1.0 = full slot).",
    )
    parser.add_argument(
        "--bbox-color",
        type=str,
        default="green",
        help="BBox color: 'green', 'red', or B,G,R triplet.",
    )
    parser.add_argument("--start-combo", type=int, default=1)
    parser.add_argument("--end-combo", type=int, default=None)
    parser.add_argument("--max-combos", type=int, default=None)
    return parser.parse_args()


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


SLOT_OFFSETS = {"tl": (0, 0), "tr": (1, 0), "bl": (0, 1), "br": (1, 1)}

COLOR_MAP = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
}


def parse_bbox_color(color_str: str) -> Tuple[int, int, int]:
    if color_str.lower() in COLOR_MAP:
        return COLOR_MAP[color_str.lower()]
    parts = [int(x.strip()) for x in color_str.split(",")]
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError(f"Invalid bbox-color: {color_str!r}")


def default_bbox_thickness(slot_w: int, slot_h: int) -> int:
    return max(12, min(slot_w, slot_h) // 15)


def draw_active_bbox(
    canvas: np.ndarray,
    layout: Dict[str, str],
    active_animal: str,
    slot_w: int,
    slot_h: int,
    thickness: int,
    scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    for slot_key, animal in layout.items():
        if animal != active_animal:
            continue
        ox, oy = SLOT_OFFSETS[slot_key]
        cx = ox * slot_w + slot_w // 2
        cy = oy * slot_h + slot_h // 2
        half_w = int(slot_w * scale / 2)
        half_h = int(slot_h * scale / 2)
        x1, y1 = cx - half_w, cy - half_h
        x2, y2 = cx + half_w, cy + half_h
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
    return canvas


def render_all_sleep(layout: Dict[str, str], sleep_by_animal: Dict[str, np.ndarray]) -> np.ndarray:
    top = np.hstack((sleep_by_animal[layout["tl"]], sleep_by_animal[layout["tr"]]))
    bottom = np.hstack((sleep_by_animal[layout["bl"]], sleep_by_animal[layout["br"]]))
    return np.vstack((top, bottom))


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

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[a_out]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(tmp_out),
        ]
    )

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        tail = "\n".join(stderr[-20:])
        raise RuntimeError(f"ffmpeg audio mux failed for {output_video_path}:\n{tail}")
    tmp_out.replace(output_video_path)


def main() -> None:
    args = parse_args()
    if args.start_combo < 1:
        raise ValueError("--start-combo must be >= 1")
    if args.end_combo is not None and args.end_combo < args.start_combo:
        raise ValueError("--end-combo must be >= --start-combo")
    if args.gap_seconds < 0:
        raise ValueError("--gap-seconds must be >= 0")
    if args.bbox_thickness is not None and args.bbox_thickness < 1:
        raise ValueError("--bbox-thickness must be >= 1")

    bbox_color = parse_bbox_color(args.bbox_color) if args.bbox else (0, 255, 0)
    bbox_scale = args.bbox_scale

    with args.meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    records = meta.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"No valid records in {args.meta_path}")

    out_dir = args.out_dir if args.out_dir is not None else Path(meta["output_dir"])
    if not args.meta_only:
        out_dir.mkdir(parents=True, exist_ok=True)

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

    for rec in tqdm(indexed_records, desc="Rendering combos"):
        layout = rec["layout_animal"]
        speech_order = rec["speech_order_animal"]
        if "animals" in rec and isinstance(rec["animals"], dict):
            animals_info = rec["animals"]
        else:
            animals_info = {}
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

        if args.skip_existing and output_path.exists() and not args.meta_only:
            skipped += 1
            continue

        animal_video_paths: Dict[str, Path] = {}
        animal_sleep_paths: Dict[str, Path] = {}
        for key, info in animals_info.items():
            vpath = Path(info["video"])
            spath = args.image_dir / f"{info['animal']}_back.png"
            if not vpath.exists():
                raise FileNotFoundError(f"Missing source video: {vpath}")
            if not spath.exists():
                raise FileNotFoundError(f"Missing sleep image: {spath}")
            animal_video_paths[key] = vpath
            animal_sleep_paths[key] = spath

        if args.meta_only:
            ws, hs, fpss = [], [], []
            for p in animal_video_paths.values():
                w, h, fps = get_video_props(p)
                ws.append(w)
                hs.append(h)
                fpss.append(fps)
            out_fps = min(fpss)
            slot_w = max(ws)
            slot_h = max(hs)
            rendered += 1
            continue

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

        slot_w = max(frames[0].shape[1] for frames in frames_by_animal.values())
        slot_h = max(frames[0].shape[0] for frames in frames_by_animal.values())
        for animal in list(frames_by_animal.keys()):
            frames_by_animal[animal] = [fit_size(fr, slot_w, slot_h) for fr in frames_by_animal[animal]]

        sleep_by_animal: Dict[str, np.ndarray] = {}
        for animal, p in animal_sleep_paths.items():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to load sleep image: {p}")
            sleep_by_animal[animal] = fit_size(img, slot_w, slot_h)

        bbox_t = (
            max(1, int(args.bbox_thickness))
            if args.bbox_thickness is not None
            else default_bbox_thickness(slot_w, slot_h)
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (slot_w * 2, slot_h * 2),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")

        try:
            gap_frames = max(0, int(round(args.gap_seconds * out_fps)))
            all_sleep = render_all_sleep(layout, sleep_by_animal)

            for stage_idx, active_animal in enumerate(speech_order):
                active_frames = frames_by_animal[active_animal]
                for fr in active_frames:
                    canvas = render_canvas(layout, active_animal, fr, sleep_by_animal)
                    if args.bbox:
                        draw_active_bbox(canvas, layout, active_animal, slot_w, slot_h, bbox_t, bbox_scale, bbox_color)
                    writer.write(canvas)

                if stage_idx < len(speech_order) - 1 and gap_frames > 0:
                    for _ in range(gap_frames):
                        writer.write(all_sleep)
        finally:
            writer.release()

        if not args.no_audio:
            ordered_video_paths = [animal_video_paths[a] for a in speech_order]
            add_audio_track_with_gap(output_path, ordered_video_paths, args.gap_seconds)

        rendered += 1

    print(
        f"Done. selected={len(indexed_records)}, rendered={rendered}, "
        f"skipped_existing={skipped}, meta_only={args.meta_only}"
    )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
