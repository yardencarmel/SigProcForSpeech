#!/usr/bin/env python3
"""
Phase 5.5 — Results Visualisation

Reads a metrics CSV (output of evaluate.py) and generates publication-ready
plots matching the style of the F5-TTS paper.

Supported ablations (--ablation flag):
  sway      — reproduces Figure 3: WER vs SIM-o for different s values
  cfg       — CFG strength vs WER
  emotion   — emotion_weight vs SIM-o and MCD (our Improvement 1 ablation)
  nfe       — NFE steps vs WER (also from Figure 3 / Table 6)

Usage:
    .venv/bin/python scripts/plot_results.py \\
        --csv results/sway_sampling/results_metrics.csv \\
        --ablation sway \\
        --outdir results/plots

    .venv/bin/python scripts/plot_results.py \\
        --csv results/emotion_transfer/results_metrics.csv \\
        --ablation emotion \\
        --outdir results/plots
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_OK = True
except ImportError:
    print("[ERROR] matplotlib not installed.  pip install matplotlib")
    MATPLOTLIB_OK = False


# Paper-style colour palette
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def plot_sway_sampling(rows, outdir):
    """
    Reproduce Figure 3 (right panel) from the paper:
    WER(%) vs SIM-o for different Sway Sampling coefficients s.
    One line per NFE step count.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Sway Sampling Ablation", fontsize=13, fontweight="bold")

    # Group by NFE
    by_nfe = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nfe = int(r.get("nfe_steps", 32))
        s   = safe_float(r.get("sway_coef"))
        wer = safe_float(r.get("wer"))
        sim = safe_float(r.get("sim_o"))
        if None not in (s, wer, sim):
            by_nfe[nfe][s].append((wer, sim))

    nfe_vals = sorted(by_nfe.keys())
    for ax_idx, metric_key in enumerate(["wer", "sim_o"]):
        ax = axes[ax_idx]
        for i, nfe in enumerate(nfe_vals):
            s_vals = sorted(by_nfe[nfe].keys())
            ys = []
            for s in s_vals:
                pairs = by_nfe[nfe][s]
                idx = 0 if metric_key == "wer" else 1
                ys.append(np.mean([p[idx] for p in pairs]))

            y_plot = [v * 100 for v in ys] if metric_key == "wer" else ys
            ax.plot(s_vals, y_plot,
                    color=COLORS[i % len(COLORS)],
                    marker=MARKERS[i % len(MARKERS)],
                    linewidth=2, markersize=6,
                    label=f"NFE={nfe}")

        ax.set_xlabel("Sway Sampling coefficient s", fontsize=11)
        ylabel = "WER (%)" if metric_key == "wer" else "SIM-o"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    out = Path(outdir) / "sway_sampling_ablation.pdf"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    plt.close()


def plot_cfg(rows, outdir):
    """CFG strength vs WER (and SIM-o on secondary axis)."""
    cfgs, wers, sims = [], [], []
    per_cfg = defaultdict(lambda: {"wer": [], "sim": []})
    for r in rows:
        cfg = safe_float(r.get("cfg_strength"))
        wer = safe_float(r.get("wer"))
        sim = safe_float(r.get("sim_o"))
        if cfg is not None:
            if wer is not None: per_cfg[cfg]["wer"].append(wer)
            if sim is not None: per_cfg[cfg]["sim"].append(sim)

    cfgs = sorted(per_cfg.keys())
    mean_wer = [np.mean(per_cfg[c]["wer"]) * 100 if per_cfg[c]["wer"] else None for c in cfgs]
    mean_sim = [np.mean(per_cfg[c]["sim"]) if per_cfg[c]["sim"] else None for c in cfgs]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    valid_cfgs_wer = [c for c, w in zip(cfgs, mean_wer) if w is not None]
    valid_wer      = [w for w in mean_wer if w is not None]
    valid_cfgs_sim = [c for c, s in zip(cfgs, mean_sim) if s is not None]
    valid_sim      = [s for s in mean_sim if s is not None]

    ax1.plot(valid_cfgs_wer, valid_wer, "o-", color=COLORS[0], linewidth=2, label="WER (%)")
    ax2.plot(valid_cfgs_sim, valid_sim, "s--", color=COLORS[1], linewidth=2, label="SIM-o")

    ax1.set_xlabel("CFG Strength", fontsize=11)
    ax1.set_ylabel("WER (%)", color=COLORS[0], fontsize=11)
    ax2.set_ylabel("SIM-o", color=COLORS[1], fontsize=11)
    ax1.set_title("CFG Strength Ablation", fontsize=12, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(outdir) / "cfg_ablation.pdf"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    plt.close()


def plot_emotion_weight(rows, outdir):
    """
    Emotion transfer ablation:
    emotion_weight vs SIM-o (identity similarity) and MCD (emotion distance).

    Design goal: sweet spot where emotion is perceptible (low MCD toward
    emotion ref) but identity is still preserved (high SIM-o).
    """
    per_w = defaultdict(lambda: {"sim": [], "mcd": [], "wer": []})
    for r in rows:
        w   = safe_float(r.get("emotion_weight"))
        sim = safe_float(r.get("sim_o"))
        mcd = safe_float(r.get("mcd"))
        wer = safe_float(r.get("wer"))
        if w is not None:
            if sim is not None: per_w[w]["sim"].append(sim)
            if mcd is not None: per_w[w]["mcd"].append(mcd)
            if wer is not None: per_w[w]["wer"].append(wer)

    weights = sorted(per_w.keys())
    mean_sim = [np.mean(per_w[w]["sim"]) if per_w[w]["sim"] else None for w in weights]
    mean_mcd = [np.mean(per_w[w]["mcd"]) if per_w[w]["mcd"] else None for w in weights]
    mean_wer = [np.mean(per_w[w]["wer"]) * 100 if per_w[w]["wer"] else None for w in weights]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Emotion Transfer Weight Ablation", fontsize=13, fontweight="bold")

    for ax, ys, ylabel, color in zip(
        axes,
        [mean_sim, mean_mcd, mean_wer],
        ["SIM-o (↑ identity preserved)", "MCD dB (↓ closer to emotion)", "WER % (↓ better)"],
        COLORS[:3],
    ):
        valid_w = [w for w, y in zip(weights, ys) if y is not None]
        valid_y = [y for y in ys if y is not None]
        ax.plot(valid_w, valid_y, "o-", color=color, linewidth=2, markersize=7)
        ax.set_xlabel("emotion_weight (w)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(weights)
        ax.grid(True, alpha=0.3)
        ax.axvline(0.35, color="grey", linestyle=":", linewidth=1.2,
                   label="default w=0.35")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path(outdir) / "emotion_weight_ablation.pdf"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    plt.close()


def plot_sway_pdf(outdir):
    """
    Plot the Sway Sampling probability density function π(t) for several
    values of s.  This reproduces the left panel of Figure 3 in the paper.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    t = np.linspace(0, 1, 500)

    sway_coefs = [0.4, 0.0, -0.4, -0.8]
    for i, s in enumerate(sway_coefs):
        # f_sway(u; s) = u + s * (cos(π/2 * u) - 1 + u)
        f = t + s * (np.cos(np.pi / 2 * t) - 1 + t)
        # Approximate PDF via finite differences of the inverse CDF
        # (simple numerical derivative of f_sway w.r.t. u)
        df = np.gradient(f, t)
        pdf = 1.0 / (np.abs(df) + 1e-8)
        pdf /= np.trapz(pdf, t)   # normalise
        ax.plot(t, pdf,
                color=COLORS[i % len(COLORS)],
                linewidth=2,
                label=f"s={s:+.1f}")

    ax.set_xlabel("Flow step t", fontsize=11)
    ax.set_ylabel("π(t)  [probability density]", fontsize=11)
    ax.set_title("Sway Sampling PDF  π(t) for different s", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    out = Path(outdir) / "sway_sampling_pdf.pdf"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Plot ablation results")
    p.add_argument("--csv", default=None,
                   help="Metrics CSV (output of evaluate.py).  "
                        "Not required for --ablation sway_pdf.")
    p.add_argument("--ablation",
                   choices=["sway", "cfg", "emotion", "sway_pdf", "all"],
                   default="all")
    p.add_argument("--outdir", "-o", default="results/plots")
    return p.parse_args()


def main():
    args = parse_args()

    if not MATPLOTLIB_OK:
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ablation == "sway_pdf":
        plot_sway_pdf(outdir)
        return 0

    if not args.csv:
        print("[ERROR] --csv is required for this ablation type")
        return 1

    rows = load_csv(args.csv)

    if args.ablation in ("sway", "all"):
        if any("sway_coef" in r for r in rows):
            plot_sway_sampling(rows, outdir)

    if args.ablation in ("cfg", "all"):
        if any("cfg_strength" in r for r in rows):
            plot_cfg(rows, outdir)

    if args.ablation in ("emotion", "all"):
        if any("emotion_weight" in r for r in rows):
            plot_emotion_weight(rows, outdir)

    # Always generate the PDF plot
    plot_sway_pdf(outdir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
