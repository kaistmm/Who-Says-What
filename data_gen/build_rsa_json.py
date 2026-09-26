"""
Build the RSA json files (Sec. 3.3) from ``four_high_entropy/meta.json``.

Every record of the meta file describes one 2x2 four-animal video.  The
*target event* of a video is always ``video1`` (its animal is placed at layout
slot X and speaks at order Y for variant ``vXY``).  For each video we write:

    aavr.json          Acoustically-Anchored Visual Retrieval, primed prompt
                       anchor = spoken country (json key ``target_animal``), label = animal
    vaar.json          Visually-Anchored Audio Retrieval, primed prompt
                       anchor = animal (``target_animal``), label = country
    aavr_unprimed.json / vaar_unprimed.json   same rows with the unprimed prompt

The symbolic IDs are:
    position_id  = layout slot of the target (tl=0, tr=1, bl=2, br=3)
    temporal_id  = speaking order of the target (0..3)
    semantic_id  = alphabetical index of the label among the four
                   animals / countries (the semantic content of the target attribute)

The three demonstration speakers of the prime are listed in a random order
(seeded), so regenerated prompts differ from the shipped json only in that order.

Example
    python data_gen/build_rsa_json.py \
        --meta synthetic_animal_dataset/four_high_entropy/meta.json --out_dir json_files/rsa
"""
import argparse
import json
import os
import random

SLOTS = ["tl", "tr", "bl", "br"]
INSTRUCTION = (
    "There are four animals in the clip, each speaking one word. Continue the sentence with the "
    "single most likely next word. Output exactly one word. Do not add any explanation, extra words, "
    "or punctuation. Sentence: "
)


def audio_path_of(video_path):
    return video_path.replace("/video/", "/audio/").replace(".mp4", ".wav")


def aavr_prompt(demos, target_country):
    # "Germany is said by the panda, canada is said by the monkey, japan is said by the tiger, and egypt is said by the "
    parts = []
    for i, (animal, country) in enumerate(demos):
        c = country.capitalize() if i == 0 else country
        parts.append(f"{c} is said by the {animal}")
    return INSTRUCTION + ", ".join(parts) + f", and {target_country} is said by the "


def vaar_prompt(demos, target_animal):
    # "The panda says germany, The monkey says canada, The tiger says japan, and the rabbit says "
    parts = [f"The {animal} says {country}" for animal, country in demos]
    return INSTRUCTION + ", ".join(parts) + f", and the {target_animal} says "


def make_row(rec, prompt, target, label, ids):
    video = rec["output_path"]
    return {
        "video": video,
        "audio": audio_path_of(video),
        "use_audio": True,
        "conversations": [
            {"from": "human", "value": "<video>\n" + prompt},
            {"from": "gpt", "value": label},
        ],
        "target_animal": target,
        "layout_animal": rec["layout_animal"],
        "layout_country": rec["layout_country"],
        "speech_order_animal": rec["speech_order_animal"],
        "speech_order_country": rec["speech_order_country"],
        "semantic_id": ids["semantic"],
        "position_id": ids["position"],
        "temporal_id": ids["temporal"],
        "label": label,
        "label_name": "",
        "choices": {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=str, default="synthetic_animal_dataset/four_high_entropy/meta.json")
    parser.add_argument("--out_dir", type=str, default="json_files/rsa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    records = meta["records"]
    animals = sorted({r["video1"]["animal"] for r in records} | {r["video2"]["animal"] for r in records}
                     | {r["video3"]["animal"] for r in records} | {r["video4"]["animal"] for r in records})
    countries = sorted({r[f"video{i}"]["country"] for r in records for i in range(1, 5)})

    aavr, vaar, aavr_unprimed, vaar_unprimed = [], [], [], []
    for rec in records:
        t_animal = rec["video1"]["animal"]
        t_country = rec["video1"]["country"]
        position = SLOTS.index(next(s for s in SLOTS if rec["layout_animal"][s] == t_animal))
        temporal = rec["speech_order_animal"].index(t_animal)

        others = [(a, rec["layout_country"][s]) for s in SLOTS for a in [rec["layout_animal"][s]] if a != t_animal]
        demos = others[:]
        rng.shuffle(demos)

        ids_aavr = {"semantic": animals.index(t_animal), "position": position, "temporal": temporal}
        ids_vaar = {"semantic": countries.index(t_country), "position": position, "temporal": temporal}

        aavr.append(make_row(rec, aavr_prompt(demos, t_country), t_country, t_animal, ids_aavr))
        vaar.append(make_row(rec, vaar_prompt(demos, t_animal), t_animal, t_country, ids_vaar))
        aavr_unprimed.append(make_row(rec, INSTRUCTION + f"{t_country} is said by the ", t_country, t_animal, ids_aavr))
        vaar_unprimed.append(make_row(rec, INSTRUCTION + f"The {t_animal} says ", t_animal, t_country, ids_vaar))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in [("aavr", aavr), ("vaar", vaar), ("aavr_unprimed", aavr_unprimed), ("vaar_unprimed", vaar_unprimed)]:
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
