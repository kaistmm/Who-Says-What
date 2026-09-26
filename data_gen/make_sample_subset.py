"""
Extract a small demo subset (about 10 rows per json) of the synthetic dataset
together with the videos / wavs it references, so that the RSA / CMA scripts
can be smoke-tested without the full 30 GB dataset.

RSA rows are picked from different ``vXY`` variants so that the target's
position / temporal IDs are not constant (a constant ID would make the
hypothesis RSM degenerate).  CMA rows are the first rows of a few base videos.

Example (run from the repository root, with the full dataset available):
    python data_gen/make_sample_subset.py --src_root . --out_dir samples --json_out json_files/samples
"""
import argparse
import json
import os
import shutil
import subprocess

# one row per (position, temporal) variant: v00 v01 v02 v11 v12 v13 v22 v23 v30 v33
RSA_ROW_INDICES = [0, 50, 100, 250, 300, 350, 500, 550, 600, 750]
# base videos used for the CMA demo and how many rows (of 3) to keep per base
# (bases whose variant videos / wavs are all present; v11_n01 lacks the an2 wav)
CMA_BASES = [("v11_n02_base.mp4", 3), ("v22_n01_base.mp4", 3), ("v33_n01_base.mp4", 3), ("v44_n01_base.mp4", 1)]


def rewrite(path, src_prefix, dst_prefix):
    assert path.startswith(src_prefix), path
    return dst_prefix + path[len(src_prefix):]


def copy_media(src_root, rel_path, out_root, dst_rel):
    src = os.path.join(src_root, rel_path)
    dst = os.path.join(out_root, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if os.path.exists(src):
        shutil.copy2(src, dst)
    elif src.endswith(".wav"):
        # a few cma_four wavs were never extracted; take the audio track of the mp4
        mp4 = src.replace("/audios/", "/videos/").replace("/audio/", "/video/").replace(".wav", ".mp4")
        print(f"  wav missing, extracting from {mp4}")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", mp4, "-vn",
                        "-acodec", "pcm_s16le", dst], check=True)
    else:
        raise FileNotFoundError(src)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src_root", type=str, default=".",
                        help="Directory that contains synthetic_animal_dataset/ and json_files/")
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--json_out", type=str, default="json_files/samples")
    args = parser.parse_args()

    src_json = os.path.join(args.src_root, "json_files")
    os.makedirs(args.json_out, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- RSA ----------------
    for name in ["aavr", "vaar", "aavr_unprimed", "vaar_unprimed"]:
        rows = json.load(open(os.path.join(src_json, "rsa", f"{name}.json"), encoding="utf-8"))
        sub = []
        for i in RSA_ROW_INDICES:
            r = json.loads(json.dumps(rows[i]))
            for key in ("video", "audio"):
                new_rel = rewrite(r[key], "synthetic_animal_dataset/", f"{args.out_dir}/")
                copy_media(args.src_root, r[key], ".", new_rel)
                r[key] = new_rel
            sub.append(r)
        out = os.path.join(args.json_out, f"{name}.json")
        json.dump(sub, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {out} ({len(sub)} rows)")

    meta = json.load(open(os.path.join(args.src_root, "synthetic_animal_dataset/four_high_entropy/meta.json"), encoding="utf-8"))
    keep = {os.path.basename(json.load(open(os.path.join(src_json, "rsa/aavr.json")))[i]["video"]) for i in RSA_ROW_INDICES}
    meta["records"] = [r for r in meta["records"] if r["output_name"] in keep]
    meta["count"] = meta["count_pairs"] = len(meta["records"])
    json.dump(meta, open(os.path.join(args.out_dir, "four_high_entropy", "meta.json"), "w", encoding="utf-8"), indent=1)

    # ---------------- CMA ----------------
    used_names = set()
    for task in ("aavr", "vaar"):
        for m in ("position", "temporal", "semantic"):
            rows = json.load(open(os.path.join(src_json, "cma", f"{task}_{m}.json"), encoding="utf-8"))
            sub = []
            for base_name, n_keep in CMA_BASES:
                picked = [r for r in rows if os.path.basename(r["video"]) == base_name][:n_keep]
                for r in picked:
                    r = json.loads(json.dumps(r))
                    for key in ("video", "audio", "c2_video"):
                        new_rel = rewrite(r[key], "synthetic_animal_dataset/", f"{args.out_dir}/")
                        copy_media(args.src_root, r[key], ".", new_rel)
                        r[key] = new_rel
                    c2_audio = r["c2_video"].replace("/videos/", "/audios/").replace(".mp4", ".wav")
                    copy_media(args.src_root, rewrite(c2_audio, f"{args.out_dir}/", "synthetic_animal_dataset/"), ".", c2_audio)
                    used_names.add(os.path.basename(r["video"]))
                    used_names.add(os.path.basename(r["c2_video"]))
                    sub.append(r)
            out = os.path.join(args.json_out, f"{task}_{m}.json")
            json.dump(sub, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"wrote {out} ({len(sub)} rows)")

    meta = json.load(open(os.path.join(args.src_root, "synthetic_animal_dataset/cma_four/meta.json"), encoding="utf-8"))
    meta["records"] = [r for r in meta["records"] if r["output_name"] in used_names]
    meta["total_base"] = sum(r["record_type"] == "base" for r in meta["records"])
    meta["total_with_variants"] = len(meta["records"])
    json.dump(meta, open(os.path.join(args.out_dir, "cma_four", "meta.json"), "w", encoding="utf-8"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
