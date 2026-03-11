"""
Compute metrics for already-generated noise_inject audio files.
Loads Whisper and WavLM once each, applies to all 18 files.
"""
import os, sys, csv, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import soundfile as sf
import torch

AUDIO_DIR   = PROJECT_ROOT / "results" / "noise_inject" / "audio"
OUT_DIR     = PROJECT_ROOT / "results" / "noise_inject"
REF_A       = str(PROJECT_ROOT / "F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav")
REF_B       = str(PROJECT_ROOT / "F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_zh.wav")
GEN_TEXT    = ("I don't really care what you call me. I've been a silent spectator, "
               "watching species evolve, empires rise and fall.")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Parse alpha and sway from filename like "a0.20_sn0.4.wav" or "a0.00_sp0.0.wav"
def parse_tag(name):
    # tag format: a{alpha}_s{p/n}{|sway|}
    alpha_str = name[1:5]   # e.g. "0.20"
    alpha = float(alpha_str)
    sway_part = name[6:]    # e.g. "n1.0" or "p0.0"
    sign = -1.0 if sway_part[0] == 'n' else 1.0
    sway = sign * float(sway_part[1:])
    return alpha, sway

wav_files = sorted(AUDIO_DIR.glob("a*.wav"))
print(f"Found {len(wav_files)} audio files.")

# ── 1. WER via Whisper (load once) ──────────────────────────────────────────
print("\n[1/3] WER via Whisper large-v3-turbo ...")
from transformers import pipeline as hf_pipeline
whisper = hf_pipeline("automatic-speech-recognition",
                       model="openai/whisper-large-v3-turbo",
                       device=0 if DEVICE == "cuda" else -1)

def tok(s): return re.sub(r"[^\w\s]", "", s.lower()).split()
def levenshtein_wer(ref, hyp):
    r_w, h_w = tok(ref), tok(hyp)
    r, h = len(r_w), len(h_w)
    d = np.zeros((r+1, h+1), dtype=int)
    for i in range(r+1): d[i][0] = i
    for j in range(h+1): d[0][j] = j
    for i in range(1, r+1):
        for j in range(1, h+1):
            if r_w[i-1] == h_w[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return round(100.0 * d[r][h] / max(r, 1), 1)

wer_results = {}
hyp_results = {}
for wav in wav_files:
    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1: data = data.mean(1)
    result = whisper({"sampling_rate": sr, "raw": data})
    hyp = result["text"].strip()
    wer = levenshtein_wer(GEN_TEXT, hyp)
    wer_results[wav.stem] = wer
    hyp_results[wav.stem] = hyp
    safe_hyp = hyp[:60].encode("ascii", "replace").decode()
    print(f"  {wav.stem:20s}  WER={wer:5.1f}%  \"{safe_hyp}\"")

del whisper
if DEVICE == "cuda": torch.cuda.empty_cache()

# ── 2. WavLM speaker similarity (load once) ──────────────────────────────────
print("\n[2/3] Speaker similarity via WavLM-base-plus ...")
from transformers import WavLMModel, AutoFeatureExtractor
fe  = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
wlm.eval()

def embed(path):
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1: data = data.mean(1)
    if sr != 16000:
        from scipy.signal import resample
        data = resample(data, int(len(data) * 16000 / sr))
    inputs = fe(data, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = wlm(**inputs).last_hidden_state.mean(1)
    return out / out.norm()

e_A = embed(REF_A)
e_B = embed(REF_B)

sim_A_results, sim_B_results = {}, {}
for wav in wav_files:
    e_gen = embed(str(wav))
    sim_A = round(float((e_gen * e_A).sum().item()), 4)
    sim_B = round(float((e_gen * e_B).sum().item()), 4)
    sim_A_results[wav.stem] = sim_A
    sim_B_results[wav.stem] = sim_B
    print(f"  {wav.stem:20s}  SIM-A={sim_A:.4f}  SIM-B={sim_B:.4f}")

del wlm, fe
if DEVICE == "cuda": torch.cuda.empty_cache()

# ── 3. MCD (no model needed) ─────────────────────────────────────────────────
print("\n[3/3] Mel Cepstral Distortion vs A ...")
import librosa

def mcd(gen_path, ref_path):
    y_g, _ = librosa.load(str(gen_path), sr=24000)
    y_r, _ = librosa.load(str(ref_path), sr=24000)
    m_g = librosa.feature.mfcc(y=y_g, sr=24000, n_mfcc=25)[1:]
    m_r = librosa.feature.mfcc(y=y_r, sr=24000, n_mfcc=25)[1:]
    T = min(m_g.shape[1], m_r.shape[1])
    diff = m_g[:, :T] - m_r[:, :T]
    return round(float((10/np.log(10)) * np.sqrt(2*(diff**2).sum(0)).mean()), 2)

mcd_results = {}
for wav in wav_files:
    val = mcd(str(wav), REF_A)
    mcd_results[wav.stem] = val
    print(f"  {wav.stem:20s}  MCD-A={val:.2f}")

# ── 4. Save CSV ───────────────────────────────────────────────────────────────
rows = []
for wav in wav_files:
    stem = wav.stem
    alpha, sway = parse_tag(stem)
    rows.append({
        "noise_level": alpha,
        "sway_coef":   sway,
        "tag":         stem,
        "output_wav":  str(wav),
        "wer":         wer_results.get(stem, "err"),
        "whisper_hyp": hyp_results.get(stem, ""),
        "sim_A":       sim_A_results.get(stem, "err"),
        "sim_B":       sim_B_results.get(stem, "err"),
        "mcd_A":       mcd_results.get(stem, "err"),
    })

csv_path = OUT_DIR / "results_metrics.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\n[OK] CSV -> {csv_path}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

noise_levels = sorted(set(r["noise_level"] for r in rows))
sway_coefs   = sorted(set(r["sway_coef"]   for r in rows))
markers = ["o", "s", "^"]

metrics = [
    ("wer",   "WER (%)",             "red"),
    ("sim_A", "SIM-A (identity A)",  "blue"),
    ("sim_B", "SIM-B (style B)",     "orange"),
    ("mcd_A", "MCD-A (vs identity)", "green"),
]

fig, axes = plt.subplots(4, 1, figsize=(7, 18))
fig.suptitle(
    "Noise-Injection Style Transfer — SDEdit on Flow Matching\n"
    "Identity A = basic_ref_en  |  Style B = basic_ref_zh\n"
    "Each line = one sway coefficient (s);  x-axis = noise level (α)",
    fontsize=10,
)
axes = axes.flatten()

for ax, (metric, ylabel, color) in zip(axes, metrics):
    for i, sway in enumerate(sway_coefs):
        subset = [r for r in rows if r["sway_coef"] == sway
                  and isinstance(r[metric], (int, float))]
        if not subset: continue
        xs = [r["noise_level"] for r in subset]
        ys = [r[metric] for r in subset]
        ax.plot(xs, ys, marker=markers[i % len(markers)],
                label=f"sway={sway:+.1f}")
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

# ── 6. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'alpha':>6}  {'sway':>6}  {'WER%':>6}  {'SIM-A':>7}  {'SIM-B':>7}  {'MCD-A':>7}")
print("-" * 80)
for r in sorted(rows, key=lambda x: (x["noise_level"], x["sway_coef"])):
    print(f"{r['noise_level']:>6.2f}  {r['sway_coef']:>+6.1f}  "
          f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
          f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}")
print("=" * 80)
