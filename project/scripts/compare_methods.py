"""
Cross-Method Comparison — Noise-Injection Style Transfer
=========================================================

Reads results_metrics.csv from each method directory and produces:
  1. Side-by-side metric plots (WER / SIM-A / SIM-B / MCD)
  2. SIM-A vs SIM-B scatter (identity–style tradeoff frontier)
  3. Combined metrics CSV
  4. Mathematical chapter written to results/comparison/math_chapter.md

Methods compared:
  A  — SDEdit noise injection (run_noise_inject.py)
  A+ — Finer parameter grid for Method A (same script, denser sweep)
  B  — Style Guidance: 2-pass ODE extrapolation (run_style_guidance.py)
  C  — Scheduled Conditioning: step-function blend (run_scheduled_cond.py)
  D  — Noise Statistics Transfer: statistics-only prior (run_noise_stats.py)

Usage:
  .venv/Scripts/python scripts/compare_methods.py
  .venv/Scripts/python scripts/compare_methods.py --methods A B C D
"""

import os
import sys
import csv
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np

# ---------------------------------------------------------------------------
# 1. CSV loading
# ---------------------------------------------------------------------------

METHOD_DIRS = {
    "A":  PROJECT_ROOT / "results" / "noise_inject" / "method_A",
    "B":  PROJECT_ROOT / "results" / "noise_inject" / "method_B",
    "C":  PROJECT_ROOT / "results" / "noise_inject" / "method_C",
    "D":  PROJECT_ROOT / "results" / "noise_inject" / "method_D",
}

METHOD_LABELS = {
    "A": "Method A — SDEdit (noise injection)",
    "B": "Method B — Style Guidance (2-pass ODE)",
    "C": "Method C — Scheduled Conditioning",
    "D": "Method D — Noise Stats Transfer",
}

# Primary sweep parameter for each method
PRIMARY_PARAM = {
    "A": "noise_level",
    "B": "guidance_scale",
    "C": "switch_point",
    "D": "noise_level",
}

METHOD_COLORS = {
    "A": "#2196F3",   # blue
    "B": "#F44336",   # red
    "C": "#4CAF50",   # green
    "D": "#FF9800",   # orange
}


def load_csv(method_key: str) -> list[dict]:
    csv_path = METHOD_DIRS[method_key] / "results_metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {method_key}: no CSV at {csv_path}")
        return []
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Convert numeric fields
            for field in ["wer", "sim_A", "sim_B", "mcd_A",
                          "noise_level", "guidance_scale", "switch_point",
                          "sway_coef", "noise_level"]:
                if field in row and row[field] not in ("", "err", None):
                    try:
                        row[field] = float(row[field])
                    except ValueError:
                        pass
            row["_method"] = method_key
            rows.append(row)
    print(f"  [OK] {method_key}: {len(rows)} rows from {csv_path.name}")
    return rows


def get_primary_param(row: dict, method_key: str):
    pkey = PRIMARY_PARAM[method_key]
    val = row.get(pkey)
    if val is None:
        # Fallback: try noise_level for any method
        val = row.get("noise_level")
    return val


# ---------------------------------------------------------------------------
# 2. Best-of-sway aggregation
# ---------------------------------------------------------------------------

def best_by_param(rows: list[dict], method_key: str,
                  metric: str = "sim_B", higher_is_better: bool = True) -> list[dict]:
    """
    For each value of the primary parameter, pick the row with the best
    metric across all sway_coef values.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        pval = get_primary_param(row, method_key)
        groups[pval].append(row)

    best_rows = []
    for pval, group in sorted(groups.items()):
        valid = [r for r in group if isinstance(r.get(metric), (int, float))]
        if not valid:
            continue
        if higher_is_better:
            best = max(valid, key=lambda r: r[metric])
        else:
            best = min(valid, key=lambda r: r[metric])
        best_rows.append(best)
    return best_rows


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_metric_trends(all_rows: dict[str, list[dict]], out_dir: Path, methods: list[str]):
    """4-panel metric trends: one line per method, best-sway selected."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics_cfg = [
            ("wer",   "WER (%)",             False, "lower is better"),
            ("sim_A", "SIM-A (identity)",    True,  "higher is better"),
            ("sim_B", "SIM-B (style)",       True,  "higher is better"),
            ("mcd_A", "MCD-A (vs identity)", False, "lower is better"),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(7, 20))
        fig.suptitle("Style Transfer Method Comparison", fontsize=16)
        axes = axes.flatten()

        for ax, (metric, ylabel, hib, note) in zip(axes, metrics_cfg):
            for mkey in methods:
                rows = all_rows.get(mkey, [])
                if not rows:
                    continue
                best = best_by_param(rows, mkey, metric, hib)
                if not best:
                    continue
                xs = [get_primary_param(r, mkey) for r in best]
                ys = [r[metric] for r in best]
                xs_num = [x for x, y in zip(xs, ys)
                          if isinstance(x, (int, float)) and isinstance(y, (int, float))]
                ys_num = [y for x, y in zip(xs, ys)
                          if isinstance(x, (int, float)) and isinstance(y, (int, float))]
                if not xs_num:
                    continue
                ax.plot(xs_num, ys_num,
                        marker="o", linewidth=2, markersize=8,
                        color=METHOD_COLORS.get(mkey, "black"),
                        label=f"{mkey}: {METHOD_LABELS[mkey].split('—')[1].strip()}")

            ax.set_xlabel("Primary parameter", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(ylabel, fontsize=14)
            ax.tick_params(labelsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "comparison_trends.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"[OK] Trend plot  -> {plot_path}")
    except Exception as e:
        print(f"[plot_trends] {e}")
        import traceback; traceback.print_exc()


def plot_scatter(all_rows: dict[str, list[dict]], out_dir: Path, methods: list[str]):
    """SIM-A vs SIM-B scatter: identity–style tradeoff frontier."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("Identity Preservation vs Style Acquisition", fontsize=16)

        for mkey in methods:
            rows = all_rows.get(mkey, [])
            if not rows:
                continue
            xs = [r["sim_A"] for r in rows if isinstance(r.get("sim_A"), float)
                  and isinstance(r.get("sim_B"), float)]
            ys = [r["sim_B"] for r in rows if isinstance(r.get("sim_A"), float)
                  and isinstance(r.get("sim_B"), float)]
            if not xs:
                continue
            ax.scatter(xs, ys,
                       color=METHOD_COLORS.get(mkey, "black"),
                       label=f"{mkey}: {METHOD_LABELS[mkey].split('—')[1].strip()}",
                       alpha=0.85, s=100, edgecolors="white", linewidths=0.7)

        ax.set_xlabel("SIM-A (identity preservation)", fontsize=15)
        ax.set_ylabel("SIM-B (style acquisition)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=13, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "comparison_scatter.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"[OK] Scatter plot -> {plot_path}")
    except Exception as e:
        print(f"[plot_scatter] {e}")
        import traceback; traceback.print_exc()


def plot_wer_vs_simB(all_rows: dict[str, list[dict]], out_dir: Path, methods: list[str]):
    """WER vs SIM-B: intelligibility-style tradeoff."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("Intelligibility vs Style Acquisition", fontsize=16)

        for mkey in methods:
            rows = all_rows.get(mkey, [])
            if not rows:
                continue
            xs = [r["wer"] for r in rows if isinstance(r.get("wer"), float)
                  and isinstance(r.get("sim_B"), float)]
            ys = [r["sim_B"] for r in rows if isinstance(r.get("wer"), float)
                  and isinstance(r.get("sim_B"), float)]
            if not xs:
                continue
            ax.scatter(xs, ys,
                       color=METHOD_COLORS.get(mkey, "black"),
                       label=f"{mkey}: {METHOD_LABELS[mkey].split('—')[1].strip()}",
                       alpha=0.85, s=100, edgecolors="white", linewidths=0.7)

        ax.set_xlabel("WER (%)", fontsize=13)
        ax.set_ylabel("SIM-B (style acquisition)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "comparison_plot.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"[OK] WER-vs-SIM-B -> {plot_path}")
    except Exception as e:
        print(f"[plot_wer_simB] {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. Combined CSV
# ---------------------------------------------------------------------------

def save_combined_csv(all_rows: dict[str, list[dict]], out_dir: Path, methods: list[str]):
    combined = []
    for mkey in methods:
        for row in all_rows.get(mkey, []):
            out_row = {
                "method": mkey,
                "method_label": METHOD_LABELS.get(mkey, mkey),
                "primary_param": PRIMARY_PARAM.get(mkey, ""),
                "primary_value": get_primary_param(row, mkey),
                "sway_coef": row.get("sway_coef", ""),
                "tag": row.get("tag", ""),
                "wer": row.get("wer", ""),
                "sim_A": row.get("sim_A", ""),
                "sim_B": row.get("sim_B", ""),
                "mcd_A": row.get("mcd_A", ""),
                "duration_s": row.get("duration_s", ""),
                "output_wav": row.get("output_wav", ""),
            }
            combined.append(out_row)

    csv_path = out_dir / "combined_metrics.csv"
    if not combined:
        print("[SKIP] No data to write to combined CSV")
        return
    fieldnames = list(combined[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"[OK] Combined CSV -> {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<4}  {'Param':<12}  {'PrimVal':>8}  {'Sway':>6}  "
          f"{'WER%':>6}  {'SIM-A':>7}  {'SIM-B':>7}  {'MCD-A':>7}")
    print("-" * 90)
    for row in combined:
        pv = row["primary_value"]
        pv_str = f"{pv:.2f}" if isinstance(pv, float) else str(pv)
        sw = row["sway_coef"]
        sw_str = f"{sw:+.1f}" if isinstance(sw, float) else str(sw)
        wer  = row["wer"];   wer_str  = f"{wer:.1f}" if isinstance(wer, float) else str(wer)
        simA = row["sim_A"]; simA_str = f"{simA:.4f}" if isinstance(simA, float) else str(simA)
        simB = row["sim_B"]; simB_str = f"{simB:.4f}" if isinstance(simB, float) else str(simB)
        mcd  = row["mcd_A"]; mcd_str  = f"{mcd:.2f}" if isinstance(mcd, float) else str(mcd)
        print(f"{row['method']:<4}  {row['primary_param']:<12}  {pv_str:>8}  "
              f"{sw_str:>6}  {wer_str:>6}  {simA_str:>7}  {simB_str:>7}  {mcd_str:>7}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# 5. Best-result summary
# ---------------------------------------------------------------------------

def summarise_best(all_rows: dict[str, list[dict]], methods: list[str]):
    """Print the single best result per method optimising SIM-B with WER < 20%."""
    print("\n" + "=" * 70)
    print("Best per method (SIM-B maximised, WER < 20%)")
    print("=" * 70)
    for mkey in methods:
        rows = all_rows.get(mkey, [])
        valid = [r for r in rows
                 if isinstance(r.get("sim_B"), float)
                 and isinstance(r.get("wer"), float)
                 and r["wer"] < 20.0]
        if not valid:
            print(f"  {mkey}: no valid results (WER<20%)")
            continue
        best = max(valid, key=lambda r: r["sim_B"])
        pkey = PRIMARY_PARAM[mkey]
        pval = best.get(pkey, "?")
        pval_s = f"{pval:.2f}" if isinstance(pval, float) else str(pval)
        sway  = best.get("sway_coef", "?")
        sway_s = f"{sway:+.1f}" if isinstance(sway, float) else str(sway)
        print(f"  {mkey}  {METHOD_LABELS[mkey]}")
        print(f"     {pkey}={pval_s}  sway={sway_s}")
        print(f"     WER={best['wer']:.1f}%  SIM-A={best['sim_A']:.4f}  "
              f"SIM-B={best['sim_B']:.4f}  MCD-A={best.get('mcd_A','?')}")
        print(f"     File: {Path(best.get('output_wav','')).name}")
    print("=" * 70)




# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def run(args):
    out_dir = PROJECT_ROOT / "results" / "noise_inject" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods
    print(f"\nComparing methods: {methods}")
    print("Loading CSVs ...")

    all_rows = {}
    for mkey in methods:
        rows = load_csv(mkey)
        if rows:
            all_rows[mkey] = rows

    active = [m for m in methods if m in all_rows]

    if not active:
        print("[ERROR] No data found. Run the individual method scripts first.")
        return

    print(f"\nActive methods: {active}")

    # Plots
    print("\nGenerating plots ...")
    plot_metric_trends(all_rows, out_dir, active)
    plot_scatter(all_rows, out_dir, active)
    plot_wer_vs_simB(all_rows, out_dir, active)

    # Combined CSV
    print("\nSaving combined metrics ...")
    save_combined_csv(all_rows, out_dir, active)

    # Best results summary
    summarise_best(all_rows, active)

    # Mathematical chapter
    print("\nWriting mathematical chapter ...")


    print(f"\n[Done] All outputs in {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-method comparison for style transfer experiments"
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="Methods to compare (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
