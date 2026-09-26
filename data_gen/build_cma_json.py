"""
Build the CMA json files (Sec. 3.4) from ``cma_four/meta.json``.

The meta file holds, for every base video ``vXY_nNN_base`` (target animal at
layout slot X speaking at order Y), the rendered manipulated contexts c2:

    pZ   position ID manipulation : target swaps layout slot with the animal at slot Z
    tZ   temporal ID manipulation : target swaps speaking order with the animal at order Z
    anN  semantic manipulation (AAVR): target animal replaced by an unseen animal
    cnN  semantic manipulation (VAAR): target country replaced by an unseen country

Every row pairs the base video (``video``) with one c2 video (``c2_video``)
under the *same* primed prompt and stores
    base_ans / exp_ans  = y1 = y2  (answer of the unmodified query)
    causal_ans          = y1*      (expected answer after patching:
                          attribute of the swapped event, or the novel attribute)
    target_animal       = anchor word in the prompt (country for AAVR, animal for VAAR)

Outputs: aavr_{position,temporal,semantic}.json and vaar_{position,temporal,semantic}.json
(the three demonstration speakers of the prime are listed in a random, seeded order).

Example
    python data_gen/build_cma_json.py \
        --meta synthetic_animal_dataset/cma_four/meta.json --out_dir json_files/cma
"""
import argparse
import json
import os
import random
import re
from collections import OrderedDict

SLOTS = ["tl", "tr", "bl", "br"]
INSTRUCTION = (
    "There are four animals in the clip, each speaking one word. Continue the sentence with the "
    "single most likely next word. Output exactly one word. Do not add any explanation, extra words, "
    "or punctuation. Sentence: "
)


def audio_path_of(video_path):
    return video_path.replace("/videos/", "/audios/").replace(".mp4", ".wav")


def aavr_prompt(demos, target_country):
    # "Egypt is said by the monkey, Canada is said by the panda, Japan is said by the rabbit, and germany is said by the "
    parts = [f"{country.capitalize()} is said by the {animal}" for animal, country in demos]
    return INSTRUCTION + ", ".join(parts) + f", and {target_country} is said by the "


def vaar_prompt(demos, target_animal):
    # "the monkey says egypt, the panda says canada, the rabbit says japan, and the tiger says "
    parts = [f"the {animal} says {country}" for animal, country in demos]
    return INSTRUCTION + ", ".join(parts) + f", and the {target_animal} says "


def make_row(base, c2, prompt, target, label, causal_ans):
    video = base["output_path"]
    return {
        "video": video,
        "audio": audio_path_of(video),
        "use_audio": True,
        "conversations": [
            {"from": "human", "value": "<video>\n" + prompt},
            {"from": "gpt", "value": label},
        ],
        "label": label,
        "label_name": "",
        "choices": {},
        "c2_video": c2["output_path"],
        "base_ans": label,
        "exp_ans": label,
        "causal_ans": causal_ans,
        "target_animal": target,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=str, default="synthetic_animal_dataset/cma_four/meta.json")
    parser.add_argument("--out_dir", type=str, default="json_files/cma")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    an_repl = meta["an_replacements"]   # {"an1": "giraffe", ...}
    cn_repl = meta["cn_replacements"]   # {"cn1": "russia", ...}

    variants = OrderedDict()   # base name -> {suffix: record}
    bases = OrderedDict()
    for rec in meta["records"]:
        name = rec["output_name"]
        if rec["record_type"] == "base":
            bases[name] = rec
        else:
            variants.setdefault(rec["base_variant"], {})[re.sub(r"^.*_", "", name).replace(".mp4", "")] = rec

    out = {f"{task}_{m}": [] for task in ("aavr", "vaar") for m in ("position", "temporal", "semantic")}
    for base_name, base in bases.items():
        X, Y = int(base["variant"][1]), int(base["variant"][2])          # 1-based slot / order
        slot = SLOTS[X - 1]
        t_animal, t_country = base["layout_animal"][slot], base["layout_country"][slot]
        assert base["speech_order_animal"][Y - 1] == t_animal, base_name

        others = [(base["layout_animal"][s], base["layout_country"][s]) for s in SLOTS if s != slot]
        demos = others[:]
        rng.shuffle(demos)
        p_aavr, p_vaar = aavr_prompt(demos, t_country), vaar_prompt(demos, t_animal)
        vs = variants[base_name]

        for z in range(1, 5):
            if z == X or f"p{z}" not in vs:
                continue
            c2 = vs[f"p{z}"]
            out["aavr_position"].append(make_row(base, c2, p_aavr, t_country, t_animal, base["layout_animal"][SLOTS[z - 1]]))
            out["vaar_position"].append(make_row(base, c2, p_vaar, t_animal, t_country, base["layout_country"][SLOTS[z - 1]]))
        for z in range(1, 5):
            if z == Y or f"t{z}" not in vs:
                continue
            c2 = vs[f"t{z}"]
            out["aavr_temporal"].append(make_row(base, c2, p_aavr, t_country, t_animal, base["speech_order_animal"][z - 1]))
            out["vaar_temporal"].append(make_row(base, c2, p_vaar, t_animal, t_country, base["speech_order_country"][z - 1]))
        for key, novel in an_repl.items():
            if key in vs:
                out["aavr_semantic"].append(make_row(base, vs[key], p_aavr, t_country, t_animal, novel))
        for key, novel in cn_repl.items():
            if key in vs:
                out["vaar_semantic"].append(make_row(base, vs[key], p_vaar, t_animal, t_country, novel))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in out.items():
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
