#!/usr/bin/env python3
"""
Step 7 — All Graphs
=====================
Unified graph generation script. Use --graphs to select which graphs to create.

Available graphs:
  baseline       Sway sampling, CFG strength, emotion weight, sway PDF
                 (from baseline reproduction — run_all_phases.py phase 6)
  extension1     Emotion transfer v2 plot (from run_phase4.py results)
  method_A       Heatmaps + tradeoff scatter (from run_noise_inject.py results)
  method_B       Method B sweep plot (from run_style_guidance.py results)
  method_C       Method C sweep plot (from run_scheduled_cond.py results)
  method_D       Method D sweep plot (from run_noise_stats.py results)
  comparison     Cross-method comparison plots (from compare_methods.py)
  all            Generate all of the above (default)

Usage:
    .venv/Scripts/python scripts/7_graphs.py
    .venv/Scripts/python scripts/7_graphs.py --graphs baseline comparison
    .venv/Scripts/python scripts/7_graphs.py --graphs method_A method_B
    .venv/Scripts/python scripts/7_graphs.py --graphs all
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

ALL_GRAPHS = [
    "baseline", "extension1",
    "method_A", "method_B", "method_C", "method_D",
    "comparison",
]


def run_baseline_graphs(args):
    """Generate baseline plots via run_all_phases.py --phases 6."""
    print("\n" + "=" * 60)
    print("Generating BASELINE graphs (sway, CFG, emotion, sway PDF)")
    print("=" * 60)
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "1_baseline.py"),
        "--ref_audio", args.ref_audio,
        "--outdir", str(Path(args.outdir) / "baseline"),
        "--phases", "6",
        "--plots", "all",
    ]
    subprocess.call(cmd)


def run_extension1_graphs(args):
    """Regenerate extension 1 plot from existing results CSV."""
    print("\n" + "=" * 60)
    print("Generating EXTENSION 1 graphs (emotion transfer v2)")
    print("=" * 60)
    results_dir = Path(args.outdir) / "extension_1"
    csv_path = results_dir / "results_metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No results CSV at {csv_path}")
        print("         Run step 2 (2_extension1.py) first.")
        return

    try:
        import csv as csv_mod
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(csv_path, encoding="utf-8") as fh:
            rows = list(csv_mod.DictReader(fh))

        v2_rows = []
        for r in rows:
            if r.get("method") == "direct_mel_injection":
                try:
                    wer = float(r["wer"])
                    v2_rows.append({
                        "weight": float(r["weight"]),
                        "wer": wer,
                        "sim_o": float(r["sim_o"]),
                        "mcd": float(r["mcd"]),
                    })
                except (ValueError, KeyError):
                    pass

        if len(v2_rows) >= 2:
            ws = [r["weight"] for r in v2_rows]
            wers = [r["wer"] for r in v2_rows]
            sims = [r["sim_o"] for r in v2_rows]
            mcds = [r["mcd"] for r in v2_rows]

            fig, axes = plt.subplots(3, 1, figsize=(6, 12))
            fig.suptitle("Emotion Transfer v2 — Direct Mel Injection", fontsize=13)
            axes[0].plot(ws, wers, "o-r"); axes[0].set_xlabel("Emotion weight"); axes[0].set_ylabel("WER (%)"); axes[0].set_title("Word Error Rate")
            axes[1].plot(ws, sims, "o-b"); axes[1].set_xlabel("Emotion weight"); axes[1].set_ylabel("SIM-o"); axes[1].set_title("Speaker Similarity (vs identity ref)")
            axes[2].plot(ws, mcds, "o-g"); axes[2].set_xlabel("Emotion weight"); axes[2].set_ylabel("MCD"); axes[2].set_title("Mel Cepstral Distortion")
            plt.tight_layout()
            plot_path = results_dir / "emotion_transfer.png"
            plt.savefig(str(plot_path), dpi=150)
            plt.close()
            print(f"  -> {plot_path}")
        else:
            print("  [SKIP] Not enough data points for extension 1 plot")
    except Exception as e:
        print(f"  [ERROR] {e}")


def run_method_graphs(method_key, args):
    """Regenerate per-method sweep plots from existing results CSVs."""
    method_scripts = {
        "method_A": "run_noise_inject.py",
        "method_B": "run_style_guidance.py",
        "method_C": "run_scheduled_cond.py",
        "method_D": "run_noise_stats.py",
    }
    method_names = {
        "method_A": "Method A — SDEdit Noise Injection",
        "method_B": "Method B — Style Guidance (2-Pass ODE)",
        "method_C": "Method C — Scheduled Conditioning",
        "method_D": "Method D — Noise Statistics Transfer",
    }
    method_dir_names = {
        "method_A": "method_A",
        "method_B": "method_B",
        "method_C": "method_C",
        "method_D": "method_D",
    }

    print(f"\n" + "=" * 60)
    print(f"Generating {method_names[method_key]} graphs")
    print("=" * 60)

    results_dir = Path(args.outdir) / "extension_2" / method_dir_names[method_key]
    csv_path = results_dir / "results_metrics.csv"
    if not csv_path.exists():
        letter = method_key[-1]
        print(f"  [SKIP] No results CSV at {csv_path}")
        step_num = {"A": 3, "B": 4, "C": 5, "D": 6}[letter]
        print(f"         Run step {step_num} ({step_num}_method_{letter}.py) first.")
        return

    try:
        import csv as csv_mod
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv_mod.DictReader(fh))

        # Convert numeric fields
        for row in rows:
            for field in ["wer", "sim_A", "sim_B", "mcd_A", "noise_level",
                          "guidance_scale", "switch_point", "sway_coef"]:
                if field in row and row[field] not in ("", "err", None):
                    try:
                        row[field] = float(row[field])
                    except ValueError:
                        pass

        # Determine x-axis parameter and sway values
        if method_key == "method_A":
            x_param = "noise_level"
            x_label = "noise_level (alpha)"
        elif method_key == "method_B":
            x_param = "guidance_scale"
            x_label = "guidance_scale"
        elif method_key == "method_C":
            x_param = "switch_point"
            x_label = "switch_point"
        elif method_key == "method_D":
            x_param = "noise_level"
            x_label = "noise_level (alpha)"

        sway_coefs = sorted(set(r["sway_coef"] for r in rows
                                if isinstance(r.get("sway_coef"), (int, float))))

        metrics = [
            ("wer",   "WER (%)",             "red"),
            ("sim_A", "SIM-A (identity)",    "blue"),
            ("sim_B", "SIM-B (style)",       "orange"),
            ("mcd_A", "MCD-A (vs identity)", "green"),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(7, 18))
        fig.suptitle(f"{method_names[method_key]}\nx-axis = {x_label}", fontsize=11)
        axes = axes.flatten()
        markers = ["o", "s", "^", "D"]

        for ax, (metric, ylabel, color) in zip(axes, metrics):
            for i, sway in enumerate(sway_coefs):
                subset = [r for r in rows if r.get("sway_coef") == sway
                          and isinstance(r.get(metric), (int, float))
                          and isinstance(r.get(x_param), (int, float))]
                if not subset:
                    continue
                subset.sort(key=lambda r: r[x_param])
                xs = [r[x_param] for r in subset]
                ys = [r[metric] for r in subset]
                ax.plot(xs, ys, marker=markers[i % len(markers)],
                        label=f"sway={sway:+.1f}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = results_dir / f"{method_dir_names[method_key]}_sweep.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"  -> {plot_path}")

        # For Method A, also generate heatmaps
        if method_key == "method_A":
            _generate_method_A_heatmaps(rows, results_dir)

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()


def _generate_method_A_heatmaps(rows, out_dir):
    """Generate heatmap plots for Method A (copied from run_noise_inject.py)."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        alphas = sorted(set(r["noise_level"] for r in rows
                            if isinstance(r.get("noise_level"), (int, float))))
        sways = sorted(set(r["sway_coef"] for r in rows
                           if isinstance(r.get("sway_coef"), (int, float))))

        metrics_cfg = [
            ("wer",   "WER (%)",             "RdYlGn_r", None, None),
            ("sim_A", "SIM-A (identity)",    "RdYlGn",   0.0,  1.0),
            ("sim_B", "SIM-B (style)",       "RdYlGn",   0.0,  1.0),
            ("mcd_A", "MCD-A (vs identity)", "RdYlGn_r", None, None),
        ]

        a_idx = {a: i for i, a in enumerate(alphas)}
        s_idx = {s: i for i, s in enumerate(sways)}

        for metric, title, cmap, vmin, vmax in metrics_cfg:
            data = np.full((len(sways), len(alphas)), np.nan)
            for r in rows:
                ai = a_idx.get(r.get("noise_level"))
                si = s_idx.get(r.get("sway_coef"))
                val = r.get(metric)
                if ai is not None and si is not None and isinstance(val, (int, float)):
                    data[si, ai] = val

            fig, ax = plt.subplots(figsize=(14, 7))
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            plt.colorbar(im, ax=ax, label=title)
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha="right")
            ax.set_yticks(range(len(sways)))
            ax.set_yticklabels([f"{s:+.1f}" for s in sways])
            ax.set_xlabel("noise_level alpha")
            ax.set_ylabel("sway coefficient")
            ax.set_title(f"Method A -- {title}")

            for si in range(len(sways)):
                for ai in range(len(alphas)):
                    v = data[si, ai]
                    if not np.isnan(v):
                        txt = f"{v:.0f}" if metric == "wer" else f"{v:.2f}"
                        brightness = (v - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-9)
                        txt_color = "white" if (brightness < 0.35 or brightness > 0.75) else "black"
                        ax.text(ai, si, txt, ha="center", va="center", fontsize=6, color=txt_color)

            plt.tight_layout()
            p = out_dir / f"heatmap_{metric}.png"
            plt.savefig(str(p), dpi=150)
            plt.close()
            print(f"  -> {p}")

    except Exception as e:
        print(f"  [heatmap error] {e}")


def run_comparison_graphs(args):
    """Run cross-method comparison via compare_methods.py."""
    print("\n" + "=" * 60)
    print("Generating COMPARISON graphs (cross-method)")
    print("=" * 60)
    compare_script = SCRIPTS_DIR / "compare_methods.py"
    cmd = [sys.executable, str(compare_script)] + args.compare_args
    subprocess.call(cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified graph generation for all experiment steps"
    )
    parser.add_argument(
        "--graphs", nargs="+",
        choices=ALL_GRAPHS + ["all"],
        default=["all"],
        help="Which graphs to generate (default: all)",
    )
    parser.add_argument(
        "--ref_audio", "-r",
        default="F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
        help="Reference audio for baseline plots",
    )
    parser.add_argument(
        "--outdir", "-o", default="results",
        help="Root results directory",
    )
    parser.add_argument(
        "--compare_args", nargs="*", default=[],
        help="Extra arguments to pass to compare_methods.py (e.g. --methods A B)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graphs = args.graphs

    if "all" in graphs:
        graphs = ALL_GRAPHS

    print(f"Generating graphs: {graphs}")

    if "baseline" in graphs:
        run_baseline_graphs(args)

    if "extension1" in graphs:
        run_extension1_graphs(args)

    for mk in ["method_A", "method_B", "method_C", "method_D"]:
        if mk in graphs:
            run_method_graphs(mk, args)

    if "comparison" in graphs:
        run_comparison_graphs(args)

    print("\n" + "=" * 60)
    print("ALL REQUESTED GRAPHS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
