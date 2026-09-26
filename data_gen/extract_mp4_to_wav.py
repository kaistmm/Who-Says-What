#!/usr/bin/env python3
"""
Extract the audio track of every rendered video into a sibling wav directory.

The analysis json files reference the audio as ``.../audio/<name>.wav``
(four_high_entropy) or ``.../audios/<name>.wav`` (cma_four); the model reads the
wav, not the mp4 audio track, so this step must cover *all* videos.

Example
    python data_gen/extract_mp4_to_wav.py \
        --video_dir synthetic_animal_dataset/four_high_entropy/video \
        --audio_dir synthetic_animal_dataset/four_high_entropy/audio
    python data_gen/extract_mp4_to_wav.py \
        --video_dir synthetic_animal_dataset/cma_four/videos \
        --audio_dir synthetic_animal_dataset/cma_four/audios
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=None, help="Resample (default: keep the mp4 rate)")
    parser.add_argument("--channels", type=int, default=None, help="Channel count (default: keep)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Please install ffmpeg first.")
    mp4_files = sorted(args.video_dir.glob("*.mp4"))
    if not mp4_files:
        raise SystemExit(f"No .mp4 files found in: {args.video_dir}")
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    converted = skipped = failed = 0
    for mp4 in mp4_files:
        wav = args.audio_dir / (mp4.stem + ".wav")
        if wav.exists() and not args.overwrite:
            skipped += 1
            continue
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-vn", "-acodec", "pcm_s16le"]
        if args.sample_rate is not None:
            cmd += ["-ar", str(args.sample_rate)]
        if args.channels is not None:
            cmd += ["-ac", str(args.channels)]
        cmd.append(str(wav))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failed += 1
            print(f"FAILED {mp4}: {proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}")
        else:
            converted += 1
    print(f"converted={converted} skipped={skipped} failed={failed} -> {args.audio_dir}")


if __name__ == "__main__":
    main()
