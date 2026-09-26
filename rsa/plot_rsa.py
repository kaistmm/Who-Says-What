"""
Paper-style RSA figure (Fig. 2) from the ``rsa_*_all.npy`` files written by
rsa_anchor_token.py / rsa_last_token.py.

Example
    python rsa/plot_rsa.py --result_dir exp/rsa/aavr/anchor --output figures/rsa1.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"semantic": "#7F77DD", "position": "#EF9F27", "temporal": "#1D9E75"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing rsa_{semantic,position,temporal}_all.npy")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (.pdf/.png). Default: <result_dir>/rsa_combined_all_pretty.pdf")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _auto_ylim(values, pad=0.05):
    finite_vals = np.asarray(values)[np.isfinite(values)]
    if finite_vals.size == 0:
        return -0.3, 0.6
    lo = max(-1.0, min(float(np.min(finite_vals)) - pad, -pad))
    hi = min(1.0, max(float(np.max(finite_vals)) + pad, pad))
    if hi <= lo:
        lo, hi = -0.3, 0.6
    return lo, hi


def plot_combined(score_dict, output_path, dpi=220):
    plt.style.use("seaborn-v0_8-whitegrid")
    label_fontsize = 23.4
    tick_fontsize = 18

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#FCFCFD")
    ax.set_facecolor("#FFFFFF")

    n_layers = None
    for name in ("semantic", "position", "temporal"):
        scores = np.asarray(score_dict[name])
        n_layers = len(scores)
        ax.plot(np.arange(n_layers), scores, label=name.capitalize(),
                color=PALETTE[name], linewidth=4.95, alpha=0.95)

    if n_layers is not None:
        x_ticks = np.arange(0, n_layers, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=tick_fontsize)

    all_scores = np.concatenate([np.asarray(v) for v in score_dict.values()])
    vmin, vmax = _auto_ylim(all_scores)
    ax.set_ylim(vmin, vmax)
    y_start = np.ceil(vmin / 0.2) * 0.2
    y_end = np.floor(vmax / 0.2) * 0.2
    if y_start <= y_end:
        ax.set_yticks(np.round(np.arange(y_start, y_end + 1e-9, 0.2), 2))
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    ax.set_xlabel("Layer", fontsize=label_fontsize)
    ax.set_ylabel("Correlation(r)", fontsize=label_fontsize)
    ax.axhline(0, color="#9AA0A6", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.12)

    legend = ax.legend(fontsize=label_fontsize, frameon=True, loc="upper left")
    legend.get_frame().set_edgecolor("#DADCE0")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    score_dict = {name: np.load(os.path.join(args.result_dir, f"rsa_{name}_all.npy"))
                  for name in ("semantic", "position", "temporal")}
    output = args.output or os.path.join(args.result_dir, "rsa_combined_all_pretty.pdf")
    plot_combined(score_dict, output, dpi=args.dpi)
    print(f"Saved: {output}")
