"""
Paper-style CMA figure (Fig. 4): layer-wise mean Causal Mediation Score of the
three manipulations (temporal / position / semantic) at one probed token.

Each input is a ``*_cma_results.jsonl`` written by cma/cma_window.py.  The
series label/colour is inferred from the file name (``temporal``, ``position``
or ``semantic``).

Example
    python cma/plot_cma.py \
        exp/cma/aavr/temporal/anchor/aavr_temporal_cma_results.jsonl \
        exp/cma/aavr/position/anchor/aavr_position_cma_results.jsonl \
        exp/cma/aavr/semantic/anchor/aavr_semantic_cma_results.jsonl \
        --output-path figures/cma1.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"semantic": "#7F77DD", "position": "#EF9F27", "temporal": "#1D9E75"}
LINE_KW = {"linewidth": 4.95, "alpha": 0.95}
FALLBACK_COLOR = "#666666"
LABEL_FONTSIZE = 23.4
YLABEL_FONTSIZE = 19
TICK_FONTSIZE = 18


def infer_task_type(path: Path) -> str:
    name = path.name.lower()
    for key in ("semantic", "temporal", "position"):
        if key in name:
            return key
    text = str(path).lower()
    for key in ("semantic", "temporal", "position"):
        if key in text:
            return key
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl_paths", nargs="+", type=Path, help="*_cma_results.jsonl files (one per manipulation)")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Output figure (.pdf/.png). Default: <dir of first input>/combined_layer_mean_line.pdf")
    parser.add_argument("--std-scale", type=float, default=0.0,
                        help="Width of the shaded band as a multiple of the per-layer std (0 = no band)")
    return parser.parse_args()


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def layer_means(path: Path):
    values_by_layer = {}
    valid_sample_ids = set()
    n_total = 0
    for r in load_rows(path):
        n_total += 1
        val, layer = r.get("logits_diff_change"), r.get("layer")
        if val is None or layer is None:
            continue
        values_by_layer.setdefault(int(layer), []).append(float(val))
        if r.get("id") is not None:
            valid_sample_ids.add(r["id"])
    if not values_by_layer:
        raise ValueError(f"No valid logits_diff_change/layer values in {path}")
    layers = sorted(values_by_layer)
    means = [float(np.mean(values_by_layer[l])) for l in layers]
    stds = [float(np.std(values_by_layer[l])) for l in layers]
    return layers, means, stds, n_total, len(valid_sample_ids)


def main() -> None:
    args = parse_args()
    std_scale = float(args.std_scale)

    series = []
    for idx, path in enumerate(args.jsonl_paths):
        layers, means, stds, n_total, n_valid = layer_means(path)
        task = infer_task_type(path)
        series.append({
            "path": path, "layers": layers, "means": means, "stds": stds,
            "n_total": n_total, "n_valid": n_valid,
            "label": task.capitalize() if task != "unknown" else f"series{idx}",
            "color": PALETTE.get(task, FALLBACK_COLOR),
        })

    # legend / drawing order as in the RSA figure
    order = {"semantic": 0, "position": 1, "temporal": 2}
    series.sort(key=lambda s: order.get(s["label"].lower(), 9))

    output_path = args.output_path or (args.jsonl_paths[0].parent / "combined_layer_mean_line.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#FCFCFD")
    ax.set_facecolor("#FFFFFF")

    all_layers = []
    for s in series:
        layers = np.asarray(s["layers"])
        means = np.asarray(s["means"], dtype=float)
        stds = np.asarray(s["stds"], dtype=float)
        ax.plot(layers, means, color=s["color"], label=s["label"], **LINE_KW)
        if std_scale > 0:
            ax.fill_between(layers, means - stds * std_scale, means + stds * std_scale,
                            color=s["color"], alpha=0.18, linewidth=0)
        all_layers.extend(layers.tolist())

    if all_layers:
        x_ticks = np.arange(0, int(np.max(all_layers)) + 1, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=TICK_FONTSIZE)

    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("Layer", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Causal Mediation Score", fontsize=YLABEL_FONTSIZE)
    ax.axhline(0, color="#9AA0A6", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.12)
    legend = ax.legend(fontsize=LABEL_FONTSIZE, frameon=True, loc="upper left")
    legend.get_frame().set_edgecolor("#DADCE0")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # y tick labels with just enough decimals to be distinct
    fig.canvas.draw()
    ymin, ymax = ax.get_ylim()
    visible_ticks = [t for t in ax.get_yticks() if ymin <= t <= ymax]
    prec = 0
    for v in visible_ticks:
        for d in range(4):
            if abs(round(v, d) - v) < 1e-6:
                prec = max(prec, d)
                break
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, p=prec: f"{v:.{p}f}"))

    plt.savefig(output_path, dpi=220)
    plt.close()

    print(f"saved: {output_path}")
    for s in series:
        print(f"  {s['label']:<9} {s['path']}: rows={s['n_total']}, samples={s['n_valid']}")


if __name__ == "__main__":
    main()
