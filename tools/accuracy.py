#!/usr/bin/env python3
"""Accuracy from result jsonl: normalize model text (punctuation, junk lines) before comparing."""
import argparse
import json
import re
from pathlib import Path

# Animal names of the synthetic dataset; extend with --animals
DEFAULT_ANIMALS = frozenset({"tiger", "panda", "monkey", "rabbit"})


def strip_assistant_prefix(s: str) -> str:
    if "\nassistant\n" in s:
        s = s.split("\nassistant\n")[-1]
    return s


def norm_label(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = strip_assistant_prefix(s)
    s = s.strip().strip('.,!?;:\'"`')
    return s


def norm_raw_pred(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = strip_assistant_prefix(s)
    return s


def extract_pred_token(raw: str, known: frozenset[str]) -> str:
    """Pick one answer word: last known animal name in text, else last long alphabetic token."""
    s = norm_raw_pred(raw)
    if not s:
        return ""
    tokens = re.findall(r"[a-z]+", s)
    if not tokens:
        return re.sub(r"[^a-z0-9]+", "", s)

    for t in reversed(tokens):
        if t in known:
            return t
    for t in reversed(tokens):
        if len(t) >= 3:
            return t
    return tokens[-1]


def is_match(label: str, pred_token: str, prefix_match: bool) -> bool:
    if not label or not pred_token:
        return False
    if pred_token == label:
        return True
    if prefix_match:
        n = min(2, len(label), len(pred_token))
        if n > 0 and pred_token[:n] == label[:n]:
            return True
    return False


def load_label_rows(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"Label JSONL decode error at line {line_no}: {e}")
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise SystemExit(f"Unsupported label format in: {path}")


def build_label_lookup(label_rows):
    by_video = {}
    by_id = {}
    by_index = []

    for idx, row in enumerate(label_rows):
        label = norm_label(row.get("label") or row.get("ref"))
        by_index.append(label)

        if "id" in row:
            by_id[str(row.get("id"))] = label

        video = row.get("video")
        if video is not None:
            by_video[str(video)] = label

    return by_id, by_video, by_index


POS_KEYS = ("tl", "tr", "bl", "br")
POS_INDEX = {k: i for i, k in enumerate(POS_KEYS)}


def load_meta(path: Path):
    """Load step0_*_rsa.json and index by video for layout/speech-order lookups."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list in meta file: {path}")
    by_video = {}
    for row in data:
        v = row.get("video")
        if v is not None:
            by_video[str(v)] = row
    return by_video


def find_target_positions(meta_row):
    """Return (layout_idx, speech_idx) for target_animal.

    target_animal may be an animal (visual_rsa case) or a country (audio_rsa case).
    Search in both layout_animal/layout_country and speech_order_animal/speech_order_country.
    Returns (None, None) if not found.
    """
    if not meta_row:
        return None, None
    target = str(meta_row.get("target_animal", "")).strip().lower()
    if not target:
        return None, None

    layout_idx = None
    for layout_key in ("layout_animal", "layout_country"):
        layout = meta_row.get(layout_key) or {}
        for k in POS_KEYS:
            if str(layout.get(k, "")).strip().lower() == target:
                layout_idx = POS_INDEX[k]
                break
        if layout_idx is not None:
            break

    speech_idx = None
    for so_key in ("speech_order_animal", "speech_order_country"):
        so = meta_row.get(so_key) or []
        for i, v in enumerate(so):
            if str(v).strip().lower() == target:
                speech_idx = i
                break
        if speech_idx is not None:
            break

    return layout_idx, speech_idx


def print_breakdown(title, counts, labels):
    total_c = sum(c for c, _ in counts.values())
    total_t = sum(t for _, t in counts.values())
    print(f"\n--- accuracy by {title} ---")
    for key in sorted(counts.keys(), key=lambda k: (k is None, k)):
        c, t = counts[key]
        name = labels.get(key, str(key))
        acc = c / t if t else 0.0
        print(f"  {name}: {c}/{t} = {acc:.4f} ({acc:.2%})")
    if total_t:
        acc = total_c / total_t
        print(f"  [overall] {total_c}/{total_t} = {acc:.4f} ({acc:.2%})")


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy from patched result jsonl.")
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Result jsonl written by tools/infer.py",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label file (.json or .jsonl). If set, prefer labels from this file.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="",
        help="Optional step0_*_rsa.json with layout_animal / speech_order_* fields; "
             "enables accuracy breakdown by target layout position and speech order.",
    )
    parser.add_argument(
        "--prefix-match",
        action="store_true",
        help="Also accept first-letter match on normalized words (e.g. tiger vs tiger.)",
    )
    parser.add_argument(
        "--animals",
        type=str,
        default="",
        help="Comma-separated extra animal names (lowercase) to recognize in noisy output",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Disable animal-token extraction; compare stripped lowercase strings only",
    )
    parser.add_argument(
        "--by-vxx",
        action="store_true",
        help="Break accuracy down by the _vXX suffix in the video filename (e.g. v00, v13).",
    )
    args = parser.parse_args()

    known = set(DEFAULT_ANIMALS)
    if args.animals.strip():
        known.update(a.strip().lower() for a in args.animals.split(",") if a.strip())
    known_frozen = frozenset(known)

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    label_by_id = {}
    label_by_video = {}
    label_by_index = []
    if args.label:
        label_path = Path(args.label)
        if not label_path.exists():
            raise SystemExit(f"Label file not found: {label_path}")
        label_rows = load_label_rows(label_path)
        label_by_id, label_by_video, label_by_index = build_label_lookup(label_rows)

    meta_by_video = {}
    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.exists():
            raise SystemExit(f"Meta file not found: {meta_path}")
        meta_by_video = load_meta(meta_path)

    total = 0
    correct = 0
    layout_counts = {}  # key -> [correct, total]
    speech_counts = {}
    vxx_counts = {}
    vxx_re = re.compile(r"_v(\d+)\.mp4$")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON decode error at line {line_no}: {e}")

            label = ""
            if args.label:
                row_id = row.get("id")
                row_video = row.get("video")
                if row_id is not None:
                    label = label_by_id.get(str(row_id), "")
                if not label and row_video is not None:
                    label = label_by_video.get(str(row_video), "")
                if not label and total < len(label_by_index):
                    label = label_by_index[total]

            if not label:
                label = norm_label(row.get("label") or row.get("ref"))

            raw_pred = row.get("text_output") or row.get("prediction") or ""
            if args.no_extract:
                pred = norm_label(raw_pred)
            else:
                pred = extract_pred_token(raw_pred, known_frozen)

            total += 1
            ok = is_match(label, pred, args.prefix_match)
            if ok:
                correct += 1

            if meta_by_video:
                meta_row = meta_by_video.get(str(row.get("video", "")))
                layout_idx, speech_idx = find_target_positions(meta_row)
                lc = layout_counts.setdefault(layout_idx, [0, 0])
                lc[0] += int(ok); lc[1] += 1
                sc = speech_counts.setdefault(speech_idx, [0, 0])
                sc[0] += int(ok); sc[1] += 1

            if args.by_vxx:
                m = vxx_re.search(str(row.get("video", "")))
                vkey = f"v{m.group(1)}" if m else "unknown"
                vc = vxx_counts.setdefault(vkey, [0, 0])
                vc[0] += int(ok); vc[1] += 1

    acc = correct / total if total else 0.0
    print(f"file={path}")
    if args.label:
        print(f"label_file={args.label}")
    if args.meta:
        print(f"meta_file={args.meta}")
    print(f"correct={correct}")
    print(f"total={total}")
    print(f"accuracy={acc:.6f} ({acc:.2%})")

    if meta_by_video:
        layout_labels = {0: "tl(0)", 1: "tr(1)", 2: "bl(2)", 3: "br(3)", None: "unknown"}
        speech_labels = {0: "speech#0", 1: "speech#1", 2: "speech#2", 3: "speech#3", None: "unknown"}
        layout_counts_t = {k: tuple(v) for k, v in layout_counts.items()}
        speech_counts_t = {k: tuple(v) for k, v in speech_counts.items()}
        print_breakdown("layout position of target", layout_counts_t, layout_labels)
        print_breakdown("speech-order index of target", speech_counts_t, speech_labels)

    if args.by_vxx:
        vxx_counts_t = {k: tuple(v) for k, v in vxx_counts.items()}
        vxx_labels = {k: k for k in vxx_counts_t}
        print_breakdown("video variant (vXX)", vxx_counts_t, vxx_labels)


if __name__ == "__main__":
    main()
