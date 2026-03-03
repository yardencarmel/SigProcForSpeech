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
        fig.suptitle(
            "Style Transfer Method Comparison — F5-TTS Flow Matching\n"
            "Identity A = basic_ref_en  |  Style B = basic_ref_zh\n"
            "Best sway_coef selected per primary parameter value",
            fontsize=11,
        )
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
                        marker="o", linewidth=2,
                        color=METHOD_COLORS.get(mkey, "black"),
                        label=f"{mkey}: {METHOD_LABELS[mkey].split('—')[1].strip()}")

            ax.set_xlabel("Primary parameter (α / guidance_scale / switch_point)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}\n({note})")
            ax.legend(fontsize=7)
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
        ax.set_title(
            "Identity Preservation vs Style Acquisition\n"
            "SIM-A = cosine similarity to identity speaker A\n"
            "SIM-B = cosine similarity to style speaker B\n"
            "(ideal = high SIM-A and high SIM-B)",
            fontsize=10,
        )

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
                       alpha=0.8, s=60, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("SIM-A (identity preservation)  →  higher = better")
        ax.set_ylabel("SIM-B (style acquisition)  →  higher = better")
        ax.legend(fontsize=8, loc="best")
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
        ax.set_title(
            "Intelligibility vs Style Acquisition\n"
            "WER = word error rate (lower = better intelligibility)\n"
            "SIM-B = style similarity to B (higher = more style transferred)",
            fontsize=10,
        )

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
                       alpha=0.8, s=60, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("WER (%)  ←  lower = better intelligibility")
        ax.set_ylabel("SIM-B (style acquisition)  →  higher = more style")
        ax.legend(fontsize=8, loc="best")
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
# 6. Mathematical chapter
# ---------------------------------------------------------------------------

MATH_CHAPTER = r"""# Mathematical Analysis of Noise-Injection Style Transfer Methods

## Overview

This chapter provides a rigorous mathematical treatment of four methods for
acoustic style transfer in flow-matching TTS systems, specifically applied to
F5-TTS (Chen et al., ACL 2025).

All methods share a common foundation: they manipulate the inference process
of a Conditional Flow Matching (CFM) model without retraining, exploiting the
geometric structure of the ODE trajectory to transfer stylistic properties
from a reference speaker B to speech conditioned on speaker A's voice.

---

## Background: Conditional Flow Matching

F5-TTS trains a vector field $v_\theta(x, c, t)$ that transports samples from
a base distribution $p_0 = \mathcal{N}(0, I)$ to the data distribution $p_1$
of mel spectrograms, conditioned on a reference mel $c$.

The forward ODE is:
$$\frac{dx}{dt} = v_\theta(x_t, c, t), \quad x_0 \sim p_0, \quad t \in [0, 1]$$

During inference with classifier-free guidance (CFG):
$$v_\text{guided}(x, c, t) = v_\theta(x, \varnothing, t) + \lambda \left[ v_\theta(x, c, t) - v_\theta(x, \varnothing, t) \right]$$

where $\lambda$ is the CFG strength and $\varnothing$ is the unconditional (dropped) conditioning.

The goal of style transfer: generate speech with speaker A's voice characteristics
(identity, timbre) but carrying stylistic/emotional properties from speaker B
(prosody, energy contour, speaking rate).

---

## Method A — SDEdit Noise Injection

**Starting point modification only; vector field unchanged.**

### Formulation

$$x_0 = (1 - \alpha) \cdot \varepsilon + \alpha \cdot \text{mel}_B, \quad \varepsilon \sim \mathcal{N}(0, I)$$

The ODE integrates from $t_\text{start} = \alpha$ to $t = 1$, conditioned on $\text{mel}_A$:

$$\frac{dx}{dt} = v_\text{guided}(x_t, \text{mel}_A, t), \quad t \in [\alpha, 1]$$

### Geometric Interpretation

In flow matching, the trajectory from $x_0$ to $x_1$ passes through a
"noise-to-data" manifold. By biasing $x_0$ toward $\text{mel}_B$, we start
the ODE in a neighbourhood of B's mel in noise space. The identity conditioning
($\text{mel}_A$) then acts as a **gravitational pull** that re-orients the
trajectory toward A's data manifold.

The SDEdit connection: if $\alpha$ is small, the ODE sees mostly noise at
$t=\alpha$ and B's influence is mild; if $\alpha$ is large, B's structure
dominates and the ODE must "undo" it with only $1-\alpha$ integration time
available. This creates a tradeoff between **style bleeding** (high $\alpha$)
and **identity preservation** (low $\alpha$).

### Sway Sampling Interaction

Sway sampling modifies the timestep schedule:
$$t' = t + s \left( \cos\!\left(\frac{\pi}{2} t\right) - 1 + t \right)$$

Negative $s$ concentrates ODE steps near $t=0$ (the noisy end). Combined with
SDEdit, negative sway allocates more function evaluations to the critical
early-integration region where B's structure is being processed.

### Hyperparameters

- $\alpha \in [0, 1]$: blending weight; 0 = pure Gaussian, 1 = pure mel_B
- $s$: sway coefficient (typically $s \in [-1, 0]$)

---

## Method B — Style Guidance (2-Pass ODE Extrapolation)

**Vector field modification at every ODE step; starting point unchanged.**

### Formulation

At each ODE timestep $t$, the transformer is evaluated twice:

$$v_A(x, t) = v_\text{guided}(x, \text{mel}_A, t) \quad \text{(identity direction)}$$
$$v_B(x, t) = v_\text{guided}(x, \text{mel}_B, t) \quad \text{(style direction)}$$

The combined vector field is:
$$v_\text{style}(x, t) = v_A + g \cdot (v_B - v_A) = (1-g) \cdot v_A + g \cdot v_B$$

where $g$ is the **guidance scale**.

### Geometric Interpretation

This is a **linear extrapolation in vector-field space**. The direction
$v_B - v_A$ points from the identity trajectory toward the style trajectory.
Scaling it by $g$ controls how far along that direction we move:

- $g = 0$: pure identity (no style) — reduces to vanilla F5-TTS
- $g = 1$: equal blend of both fields
- $g > 1$: **extrapolation** beyond the style field (analogous to negative
  guidance in classifier-free guidance)

Crucially, this method never touches $x_0$ — style is injected entirely through
the velocity field. Because the velocity field is integrated continuously, the
effect is distributed across the full temporal extent of the mel spectrogram,
unlike Method A where style is concentrated at the starting frame.

### Analogy to Classifier-Free Guidance

Standard CFG extrapolates from unconditional to conditional:
$$v_\text{CFG} = v_\varnothing + \lambda (v_c - v_\varnothing)$$

Method B extrapolates from identity-conditional to style-conditional:
$$v_\text{style} = v_A + g (v_B - v_A)$$

The formal structure is identical, with the identity conditioning $\text{mel}_A$
playing the role of the "unconditional" baseline and $\text{mel}_B$ as the target condition.

### Computational Cost

2× transformer evaluations per ODE step → 2× wall-clock time vs baseline.
NFE stays the same; only the per-step cost doubles.

---

## Method C — Scheduled Conditioning Blend (Step Function)

**Time-varying conditioning; single pass per ODE step.**

### Formulation

The conditioning is switched at a user-specified $t^* \in [0, 1]$:

$$c(t) = \begin{cases} \text{mel}_B & \text{if } t < t^* \\ \text{mel}_A & \text{if } t \geq t^* \end{cases}$$

The ODE becomes:
$$\frac{dx}{dt} = v_\text{guided}(x_t, c(t), t)$$

### Geometric Interpretation

Flow matching ODE dynamics differ across the time axis:

- **Early steps** ($t$ small, near noise): the model resolves low-frequency
  spectral envelope, speaking rate, and prosodic shape — properties dominated
  by the training distribution of the conditioning signal.
- **Late steps** ($t$ large, near data): the model refines speaker-specific
  texture, fine formant structure, and micro-prosody.

By assigning $\text{mel}_B$ to early steps, we allow B's coarse prosodic
skeleton to inform the structural scaffold of the trajectory. By handing off to
$\text{mel}_A$ for late steps, we ask the model to "paint over" this scaffold
with A's acoustic identity.

### Key Properties

- **Single-pass**: same compute as vanilla F5-TTS
- **Non-smooth vector field**: the switch at $t^*$ creates a discontinuity in
  the conditioning, which the Euler ODE integrator handles without issue but
  may interact with higher-order solvers
- **Degenerate cases**: $t^*=0$ → always use A (identity baseline);
  $t^*=1$ → always use B (pure style, voice cloning from B)

### Extensions

The step function can be replaced by a smooth schedule:

*Linear:* $c(t) = (1-t) \cdot \text{mel}_B + t \cdot \text{mel}_A$

*Cosine:* $c(t) = \text{mel}_B + \frac{1 + \cos(\pi t)}{2} (\text{mel}_A - \text{mel}_B)$

These smooth schedules interpolate through the conditioning space, potentially
reducing the sharp acoustic discontinuity at $t^*$.

---

## Method D — Noise Statistics Transfer

**Starting point modification (statistics-only); vector field unchanged.**

### Formulation

$$x_0 = \frac{\varepsilon}{\|\varepsilon\|_\sigma} \cdot \sigma_\text{target} + \alpha \cdot \mu_B$$

where:
$$\sigma_\text{target} = \alpha \cdot \text{std}(\text{mel}_B) + (1-\alpha) \cdot \text{std}(\varepsilon)$$
$$\mu_B = \text{mean}(\text{mel}_B)$$

This rescales the Gaussian noise to have the same variance as a blend of
mel_B's spectral amplitude and native Gaussian variance, then adds a fraction
of mel_B's global mean offset.

### Comparison with Method A

| Property | Method A (SDEdit) | Method D (Stats) |
|---|---|---|
| x_0 construction | $(1-\alpha)\varepsilon + \alpha \cdot \text{mel}_B$ | rescaled $\varepsilon$ with B's stats |
| Temporal structure from B | Yes — B's frames appear at specific positions | No — only global amplitude shape |
| Maximum style leakage | Strong at high $\alpha$ | Weak; only envelope |
| WER degradation | Significant at $\alpha > 0.3$ | Expected milder |
| Interpretability | Direct frame blending | Spectral normalisation |

### Information-Theoretic View

Method D transfers the **first moment** ($\mu_B$) and **second moment** ($\sigma_B^2$)
of mel_B's amplitude distribution into the starting noise, while Method A
transfers up to **infinite-order statistics** (the exact joint distribution of
frames in mel_B). Method D is thus a strict information-theoretic lower bound
on how much style information can be injected via x_0 manipulation while
preserving the Gaussianity structure of the noise distribution.

---

## Comparison Summary

| Axis | Method A | Method B | Method C | Method D |
|---|---|---|---|---|
| Where style acts | $x_0$ (frames) | $v(x,t)$ (velocity) | $c(t)$ (condition) | $x_0$ (stats) |
| Temporal specificity | High (frame-level) | Distributed (all steps) | Step-coarse | None |
| Compute overhead | ~1× | ~2× | ~1× | ~1× |
| Identity conditioning | Pure (never blended) | Blended per step | Partial | Pure |
| Sway interaction | Strong | Moderate | Moderate | Mild |
| Primary control | $\alpha$ | $g$ | $t^*$ | $\alpha$ |

The richest information pathway is Method B (style enters at every ODE step
via the vector field), making it the most expressive but also the most
computationally expensive. Method A occupies a middle ground: B's temporal
structure seeds $x_0$ but identity conditioning dominates integration. Method C
creates a "coarse-to-fine" semantic switch without extra compute. Method D is
the lightest intervention, useful as an ablation to isolate the contribution
of amplitude statistics vs. temporal frame structure.

---

## References

1. Chen, S.-g., et al. (2025). F5-TTS: A Fairytaler that Fakes Fluent and
   Faithful Speech with Flow Matching. *ACL 2025*. arXiv:2410.06885.

2. Meng, C., et al. (2022). SDEdit: Guided Image Synthesis and Editing with
   Stochastic Differential Equations. *ICLR 2022*. arXiv:2108.01073.

3. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance.
   *NeurIPS 2022 Workshop on DGMs*. arXiv:2207.12598.

4. Lipman, Y., et al. (2023). Flow Matching for Generative Modeling.
   *ICLR 2023*. arXiv:2210.02747.

5. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations.
   *NeurIPS 2018*. arXiv:1806.07366.
"""


def write_math_chapter(out_dir: Path):
    md_path = out_dir / "math_chapter.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(MATH_CHAPTER)
    print(f"[OK] Math chapter -> {md_path}")


# ---------------------------------------------------------------------------
# 7. Main
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
    write_math_chapter(out_dir)

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
