"""
Parse metrics_log.txt → results_metrics.csv + noise_inject_sweep.png
(No model loading needed — all data is already in the log file.)
"""
import csv, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_PATH  = PROJECT_ROOT / "results" / "noise_inject" / "metrics_log.txt"
OUT_DIR   = PROJECT_ROOT / "results" / "noise_inject"

import numpy as np

log = LOG_PATH.read_text(encoding="utf-8")

# ── Parse WER lines ──────────────────────────────────────────────────────────
# e.g.:  a0.00_sn0.4           WER=  0.0%  "I don't..."
wer_re = re.compile(r"(a[\d.]+_s[np][\d.]+)\s+WER=\s*([\d.]+)%\s+\"([^\"]*)\"")
wer_data, hyp_data = {}, {}
for m in wer_re.finditer(log):
    stem, wer, hyp = m.group(1), float(m.group(2)), m.group(3)
    wer_data[stem] = wer
    hyp_data[stem] = hyp

# ── Parse SIM lines ──────────────────────────────────────────────────────────
# e.g.:  a0.00_sn0.4           SIM-A=0.8843  SIM-B=0.7672
sim_re = re.compile(r"(a[\d.]+_s[np][\d.]+)\s+SIM-A=([\d.]+)\s+SIM-B=([\d.]+)")
sim_A_data, sim_B_data = {}, {}
for m in sim_re.finditer(log):
    stem = m.group(1)
    sim_A_data[stem] = float(m.group(2))
    sim_B_data[stem] = float(m.group(3))

# ── Parse MCD lines ──────────────────────────────────────────────────────────
# e.g.:  a0.00_sn0.4           MCD-A=702.16
mcd_re = re.compile(r"(a[\d.]+_s[np][\d.]+)\s+MCD-A=([\d.]+)")
mcd_data = {}
for m in mcd_re.finditer(log):
    mcd_data[m.group(1)] = float(m.group(2))

# ── Parse alpha/sway from stem ───────────────────────────────────────────────
# stem format: a{alpha}_s{n/p}{|sway|}
# e.g. "a0.10_sn1.0"  ->  alpha=0.10, sway=-1.0
#      "a0.00_sp0.0"  ->  alpha=0.00, sway=0.0
def parse_tag(stem):
    alpha = float(stem[1:5])            # "0.10"
    sway_code = stem[7:]                # "n1.0" or "p0.0"
    sign = -1.0 if sway_code[0] == 'n' else 1.0
    sway = sign * float(sway_code[1:])
    return alpha, sway

stems = sorted(wer_data.keys())
assert len(stems) == 18, f"Expected 18, got {len(stems)}: {stems}"

rows = []
for stem in stems:
    alpha, sway = parse_tag(stem)
    rows.append({
        "noise_level": alpha,
        "sway_coef":   sway,
        "tag":         stem,
        "output_wav":  str(OUT_DIR / "audio" / f"{stem}.wav"),
        "wer":         wer_data.get(stem, "err"),
        "whisper_hyp": hyp_data.get(stem, ""),
        "sim_A":       sim_A_data.get(stem, "err"),
        "sim_B":       sim_B_data.get(stem, "err"),
        "mcd_A":       mcd_data.get(stem, "err"),
    })

# ── Save CSV ─────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "results_metrics.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"[OK] CSV -> {csv_path}  ({len(rows)} rows)")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

noise_levels = sorted(set(r["noise_level"] for r in rows))
sway_coefs   = sorted(set(r["sway_coef"]   for r in rows))
markers = ["o", "s", "^"]
colors  = ["tab:blue", "tab:orange", "tab:green"]

metrics_cfg = [
    ("wer",   "WER (%)",              "tab:red"),
    ("sim_A", "SIM-A (identity A)",   "tab:blue"),
    ("sim_B", "SIM-B (style B)",      "tab:orange"),
    ("mcd_A", "MCD-A (vs identity A)","tab:green"),
]

fig, axes = plt.subplots(4, 1, figsize=(7, 18))
fig.suptitle(
    "Noise-Injection Style Transfer — SDEdit on Flow Matching\n"
    "Identity A = basic_ref_en  |  Style B = basic_ref_zh\n"
    "Each line = one sway coefficient (s);  x-axis = noise level (α)",
    fontsize=10,
)
axes = axes.flatten()

for ax, (metric, ylabel, color) in zip(axes, metrics_cfg):
    for i, sway in enumerate(sway_coefs):
        subset = sorted([r for r in rows if r["sway_coef"] == sway
                         and isinstance(r[metric], (int, float))],
                        key=lambda r: r["noise_level"])
        if not subset: continue
        xs = [r["noise_level"] for r in subset]
        ys = [r[metric] for r in subset]
        ax.plot(xs, ys, marker=markers[i], color=colors[i],
                label=f"sway={sway:+.1f}", linewidth=1.8)
    ax.set_xlabel("noise_level (α)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "noise_inject_sweep.png"
plt.savefig(str(plot_path), dpi=150)
plt.close()
print(f"[OK] Plot -> {plot_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'alpha':>6}  {'sway':>6}  {'WER%':>6}  {'SIM-A':>7}  {'SIM-B':>7}  {'MCD-A':>7}  Whisper hypothesis")
print("-" * 80)
for r in sorted(rows, key=lambda x: (x["noise_level"], x["sway_coef"])):
    hyp_short = (r["whisper_hyp"][:45] + "...") if len(r["whisper_hyp"]) > 45 else r["whisper_hyp"]
    print(f"{r['noise_level']:>6.2f}  {r['sway_coef']:>+6.1f}  "
          f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
          f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}  "
          f"\"{hyp_short}\"")
print("=" * 80)
