"""
Method A Sweep — SDEdit Noise Injection
========================================

Fine-grained parameter grid:

    alpha (noise_level) in {0.00, 0.05, 0.10, 0.15, ..., 0.70}  — 15 values
    sway  (sway_coef)   in {-1.0, -0.9, -0.8, ..., 1.0}         — 21 values
    Total: 15 × 21 = 315 runs

Output directory: results/extension_2/method_A/

Features:
  - Incremental CSV saves (resume-safe: skips already-generated WAVs)
  - 2-D heatmap visualisations for WER / SIM-A / SIM-B / MCD-A
  - Best-combination analysis table (maximise SIM-B s.t. WER < threshold)

Usage:
  .venv/Scripts/python scripts/run_noise_inject.py
  .venv/Scripts/python scripts/run_noise_inject.py --device cpu
  .venv/Scripts/python scripts/run_noise_inject.py --resume   # skip completed WAVs
"""

import os
import sys
import csv
import argparse
import itertools
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Environment patches (must come before any f5_tts imports)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_FFMPEG_BIN = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    r"Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin",
)
if os.path.isdir(_FFMPEG_BIN) and _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import soundfile as _sf
import torch as _torch
import torchaudio as _torchaudio


def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, backend=None, buffer_size=4096):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    t = _torch.from_numpy(data.T.copy())
    if frame_offset:
        t = t[:, frame_offset:]
    if num_frames > 0:
        t = t[:, :num_frames]
    return t, sr


_torchaudio.load = _sf_load

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

TARGET_SR  = 24_000
TARGET_RMS = 0.1


# ---------------------------------------------------------------------------
# 2. Audio helpers
# ---------------------------------------------------------------------------

def load_wav_simple(path: str, target_sr: int = TARGET_SR,
                    target_rms: float = TARGET_RMS) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != target_sr:
        from scipy.signal import resample
        n_out = int(audio.shape[-1] * target_sr / sr)
        audio_np = resample(audio.numpy(), n_out, axis=-1)
        audio = torch.from_numpy(audio_np.astype(np.float32))
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
    rms = torch.sqrt((audio ** 2).mean())
    if rms > 0:
        audio = audio * target_rms / rms
    return audio  # [1, T]


# ---------------------------------------------------------------------------
# 3. Model loading
# ---------------------------------------------------------------------------

def load_f5tts(device="cuda"):
    from importlib.resources import files as pkg_files
    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    yaml_path = str(pkg_files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
    model_cfg = OmegaConf.load(yaml_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_path = str(cached_path(
        "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    model = load_model(model_cls, model_arc, ckpt_path,
                       mel_spec_type="vocos", vocab_file="", device=device)
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False,
                           local_path="", device=device)
    return model, vocoder


# ---------------------------------------------------------------------------
# 4. Mel extraction
# ---------------------------------------------------------------------------

def audio_to_cond_mel(audio_1xT: torch.Tensor, mel_spec_module,
                      device: str) -> torch.Tensor:
    with torch.no_grad():
        mel = mel_spec_module(audio_1xT.to(device))
    return mel.permute(0, 2, 1)   # [1, T_mel, 100]


# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------

def infer_noise_inject(
    mel_A_3d, mel_B_3d, ref_text_A, gen_text,
    model, vocoder, device,
    noise_level=0.0, nfe_step=32, cfg_strength=2.0, sway_coef=-1.0, seed=42,
):
    """
    SDEdit-style noise injection using an external ODE loop.

    Biases the ODE starting point:
        y0 = (1 - noise_level) * randn + noise_level * mel_B_padded
    then integrates from t_start=noise_level to t=1, conditioned on mel_A.

    Uses torchdiffeq + model.transformer directly (no cfm.py modifications).
    """
    from f5_tts.model.utils import (
        convert_char_to_pinyin, list_str_to_idx, list_str_to_tensor,
        lens_to_mask,
    )
    from torchdiffeq import odeint

    if seed is not None:
        torch.manual_seed(seed)

    mel_A_3d = mel_A_3d.to(device)
    mel_B_3d = mel_B_3d.to(device)
    model.eval()

    # --- Text encoding ---
    text_list = [ref_text_A + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)
    if model.vocab_char_map is not None:
        text = list_str_to_idx(final_text_list, model.vocab_char_map).to(device)
    else:
        text = list_str_to_tensor(final_text_list).to(device)

    # --- Duration ---
    ref_audio_len = mel_A_3d.shape[1]
    ref_bytes = max(len(ref_text_A.encode("utf-8")), 1)
    gen_bytes = len(gen_text.encode("utf-8"))
    gen_len   = int(ref_audio_len / ref_bytes * gen_bytes)
    duration_val = ref_audio_len + gen_len
    duration = torch.full((1,), duration_val, device=device, dtype=torch.long)
    text_len = int((text != -1).sum().item())
    min_dur  = max(text_len, ref_audio_len) + 1
    max_duration = int(duration.clamp(min=min_dur).item())

    dtype = next(model.parameters()).dtype

    # --- Identity conditioning (A) ---
    cond_A = mel_A_3d.to(dtype)
    cond_seq_len_A = cond_A.shape[1]
    lens_A = torch.full((1,), cond_seq_len_A, device=device, dtype=torch.long)
    cond_mask_A = lens_to_mask(lens_A)
    cond_A_padded = F.pad(cond_A, (0, 0, 0, max_duration - cond_seq_len_A), value=0.0)
    cond_mask_A_padded = F.pad(cond_mask_A, (0, max_duration - cond_mask_A.shape[-1]), value=False)
    cond_mask_A_3d = cond_mask_A_padded.unsqueeze(-1)
    step_cond_A = torch.where(cond_mask_A_3d, cond_A_padded, torch.zeros_like(cond_A_padded))

    # --- Build y0: SDEdit-style noise + mel_B blend ---
    torch.manual_seed(seed)
    y0 = torch.randn(max_duration, model.num_channels, device=device, dtype=dtype)

    t_start = 0.0
    steps = nfe_step

    if noise_level > 0.0:
        # Pad / crop mel_B to match the generated sequence length
        cond_B = mel_B_3d.to(dtype)
        if cond_B.shape[1] < max_duration:
            mel_B_padded = F.pad(cond_B, (0, 0, 0, max_duration - cond_B.shape[1]), value=0.0)
        else:
            mel_B_padded = cond_B[:, :max_duration, :]
        mel_B_padded = mel_B_padded.squeeze(0)  # [max_duration, 100]

        t_start = float(noise_level)
        y0 = (1.0 - t_start) * y0 + t_start * mel_B_padded
        steps = max(1, int(nfe_step * (1.0 - t_start)))

    y0 = y0.unsqueeze(0)  # [1, max_duration, 100]

    # --- Timesteps ---
    t_steps = torch.linspace(t_start, 1.0, steps + 1, device=device, dtype=dtype)
    if sway_coef is not None:
        t_steps = t_steps + sway_coef * (
            torch.cos(torch.pi / 2 * t_steps) - 1 + t_steps)

    mask = None  # batch=1

    def fn(t, x):
        """Identity-conditioned vector field with CFG."""
        if cfg_strength < 1e-5:
            return model.transformer(
                x=x, cond=step_cond_A, text=text, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False, cache=False)
        pred_cfg = model.transformer(
            x=x, cond=step_cond_A, text=text, time=t, mask=mask,
            cfg_infer=True, cache=False)
        pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
        return pred + (pred - null_pred) * cfg_strength

    with torch.inference_mode():
        trajectory = odeint(fn, y0, t_steps, method="euler")

    model.transformer.clear_cache()

    # --- Post-process ---
    out = trajectory[-1]
    out = torch.where(cond_mask_A_3d, cond_A_padded, out)
    out = out.to(torch.float32)

    generated = out[:, ref_audio_len:, :]
    generated = generated.permute(0, 2, 1)

    with torch.no_grad():
        generated_wave = vocoder.decode(generated)

    wav  = generated_wave.squeeze().cpu().numpy()
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav, TARGET_SR


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def load_whisper(device):
    """Load Whisper ASR pipeline once at startup. Crashes immediately if it fails."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-large-v3-turbo",
                   device=0 if device == "cuda" else -1)
    print(f"  Whisper loaded on {device}")
    return asr


def compute_wer(wav_path, expected_text, asr):
    try:
        import re
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(1)
        result = asr({"sampling_rate": sr, "raw": data})
        hyp = result["text"].strip()

        def tok(s): return re.sub(r"[^\w\s]", "", s.lower()).split()
        ref_w, hyp_w = tok(expected_text), tok(hyp)
        r, h = len(ref_w), len(hyp_w)
        d = np.zeros((r + 1, h + 1), dtype=int)
        for i in range(r + 1): d[i][0] = i
        for j in range(h + 1): d[0][j] = j
        for i in range(1, r + 1):
            for j in range(1, h + 1):
                if ref_w[i - 1] == hyp_w[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return round(100.0 * d[r][h] / max(r, 1), 1), hyp
    except Exception as e:
        print(f"  [WER error] {e}")
        return -1.0, ""


def _wavlm_embed(wav_path, fe, wavlm_model, device):
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(1)
    if sr != 16000:
        from scipy.signal import resample
        data = resample(data, int(len(data) * 16000 / sr))
    inputs = fe(data, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        out = wavlm_model(**inputs).last_hidden_state.mean(1)
    return out / out.norm()


def load_wavlm(device):
    """Load WavLM once at startup. Crashes immediately if it fails."""
    from transformers import WavLMModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    try:
        wlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", use_safetensors=True)
    except Exception:
        wlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    wlm = wlm.to(device).eval()
    print(f"  WavLM loaded on {device}")
    return fe, wlm


def compute_sims(gen_wav, ref_A, ref_B, fe, wlm, device):
    e_gen = _wavlm_embed(gen_wav, fe, wlm, device)
    e_A   = _wavlm_embed(ref_A,   fe, wlm, device)
    e_B   = _wavlm_embed(ref_B,   fe, wlm, device)
    return round(float((e_gen * e_A).sum()), 4), round(float((e_gen * e_B).sum()), 4)


def compute_mcd(gen_wav, ref_wav):
    try:
        import librosa
        y_gen, sr_gen = librosa.load(str(gen_wav), sr=24000)
        y_ref, sr_ref = librosa.load(str(ref_wav), sr=24000)
        mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=sr_gen, n_mfcc=25)[1:]
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=25)[1:]
        T    = min(mfcc_gen.shape[1], mfcc_ref.shape[1])
        diff = mfcc_gen[:, :T] - mfcc_ref[:, :T]
        mcd  = (10 / np.log(10)) * np.sqrt(2 * (diff ** 2).sum(0)).mean()
        return round(float(mcd), 2)
    except Exception as e:
        print(f"  [MCD error] {e}")
        return -1.0


# ---------------------------------------------------------------------------
# 7. Heatmap plotting
# ---------------------------------------------------------------------------

def _plot_heatmaps(rows, alphas, sways, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        metrics_cfg = [
            ("wer",   "WER (%)",             "RdYlGn_r", None,    None),
            ("sim_A", "SIM-A (identity)",    "RdYlGn",   0.0,     1.0),
            ("sim_B", "SIM-B (style)",       "RdYlGn",   0.0,     1.0),
            ("mcd_A", "MCD-A (vs identity)", "RdYlGn_r", None,    None),
        ]

        # Build 2-D arrays: rows=sway (top=-1, bottom=+1), cols=alpha
        a_idx = {a: i for i, a in enumerate(alphas)}
        s_idx = {s: i for i, s in enumerate(sorted(sways))}

        for metric, title, cmap, vmin, vmax in metrics_cfg:
            data = np.full((len(sways), len(alphas)), np.nan)
            for r in rows:
                ai = a_idx.get(r["noise_level"])
                si = s_idx.get(r["sway_coef"])
                val = r[metric]
                if ai is not None and si is not None and isinstance(val, (int, float)):
                    data[si, ai] = val

            fig, ax = plt.subplots(figsize=(14, 7))
            im = ax.imshow(data, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           origin="upper")
            plt.colorbar(im, ax=ax, label=title)

            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha="right")
            ax.set_yticks(range(len(sways)))
            ax.set_yticklabels([f"{s:+.1f}" for s in sorted(sways)])
            ax.set_xlabel("noise_level α")
            ax.set_ylabel("sway coefficient")
            ax.set_title(f"Method A — {title}\n"
                         f"Identity A = basic_ref_en  |  Style B = basic_ref_zh")

            # Annotate cells with values
            for si in range(len(sways)):
                for ai in range(len(alphas)):
                    v = data[si, ai]
                    if not np.isnan(v):
                        txt = f"{v:.0f}" if metric == "wer" else f"{v:.2f}"
                        brightness = (v - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-9)
                        txt_color = "white" if (brightness < 0.35 or brightness > 0.75) else "black"
                        ax.text(ai, si, txt, ha="center", va="center",
                                fontsize=6, color=txt_color)

            plt.tight_layout()
            p = out_dir / f"heatmap_{metric}.png"
            plt.savefig(str(p), dpi=150)
            plt.close()
            print(f"[OK] Heatmap {metric} -> {p}")

    except Exception as e:
        print(f"[plot] {e}")


def _plot_tradeoff(rows, out_dir):
    """WER vs SIM-B scatter, coloured by alpha — split into two WER ranges."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        alphas_unique = sorted(set(r["noise_level"] for r in rows))
        cmap = cm.get_cmap("plasma", len(alphas_unique))
        color_map = {a: cmap(i) for i, a in enumerate(alphas_unique)}

        # Collect valid points
        points = []
        for r in rows:
            wer  = r["wer"]
            simb = r["sim_B"]
            if not isinstance(wer, (int, float)) or not isinstance(simb, (int, float)):
                continue
            points.append((wer, simb, r["noise_level"]))

        # --- Panel (a): WER 0–200 ---
        fig, ax = plt.subplots(figsize=(10, 7))
        pts_a = [(w, s, a) for w, s, a in points if w <= 200]
        for w, s, a in pts_a:
            ax.scatter(w, s, color=color_map[a],
                       s=120, alpha=0.85, edgecolors="white", linewidths=0.7)
        for a in alphas_unique:
            ax.scatter([], [], color=color_map[a], label=f"α={a:.2f}", s=100)
        ax.legend(title="noise_level α", fontsize=14, title_fontsize=15,
                  loc="upper right", ncol=2)
        ax.axvline(20, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(-5, 205)
        ax.set_xlabel("WER (%)", fontsize=17)
        ax.set_ylabel("SIM-B (style acquisition)", fontsize=17)
        ax.set_title("Method A: WER vs SIM-B Trade-off (WER ≤ 200%)", fontsize=19)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "tradeoff_wer_simb_a.png"
        plt.savefig(str(p), dpi=150)
        plt.close()
        print(f"[OK] Trade-off plot (a) -> {p}")

        # --- Panel (b): WER 500–2500 ---
        fig, ax = plt.subplots(figsize=(10, 7))
        pts_b = [(w, s, a) for w, s, a in points if w >= 500]
        for w, s, a in pts_b:
            ax.scatter(w, s, color=color_map[a],
                       s=120, alpha=0.85, edgecolors="white", linewidths=0.7)
        for a in alphas_unique:
            ax.scatter([], [], color=color_map[a], label=f"α={a:.2f}", s=100)
        ax.legend(title="noise_level α", fontsize=14, title_fontsize=15,
                  loc="upper right", ncol=2)
        ax.set_xlim(450, 2550)
        ax.set_xlabel("WER (%)", fontsize=17)
        ax.set_ylabel("SIM-B (style acquisition)", fontsize=17)
        ax.set_title("Method A: WER vs SIM-B Trade-off (WER ≥ 500%)", fontsize=19)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "tradeoff_wer_simb_b.png"
        plt.savefig(str(p), dpi=150)
        plt.close()
        print(f"[OK] Trade-off plot (b) -> {p}")

    except Exception as e:
        print(f"[tradeoff plot] {e}")


# ---------------------------------------------------------------------------
# 8. Best-combination analysis
# ---------------------------------------------------------------------------

def _print_best_table(rows, wer_threshold=20.0):
    print("\n" + "=" * 90)
    print(f"Best combinations (WER <= {wer_threshold}%) sorted by SIM-B descending")
    print("=" * 90)
    print(f"{'alpha':>6}  {'sway':>6}  {'WER%':>6}  {'SIM-A':>7}  "
          f"{'SIM-B':>7}  {'MCD-A':>7}  Whisper (truncated)")
    print("-" * 90)

    good = [r for r in rows
            if isinstance(r["wer"], (int, float)) and r["wer"] <= wer_threshold
            and isinstance(r["sim_B"], (int, float))]
    good.sort(key=lambda r: r["sim_B"], reverse=True)

    for r in good[:30]:   # top 30
        safe = str(r["whisper_hyp"])[:45].encode("ascii", "replace").decode("ascii")
        print(f"{r['noise_level']:>6.2f}  {r['sway_coef']:>+6.1f}  "
              f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
              f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}  \"{safe}\"")

    if not good:
        print("  (no combinations met the WER threshold)")
    print("=" * 90)

    # Also print best SIM-B ignoring WER
    print("\n" + "=" * 90)
    print("Top 10 by SIM-B regardless of WER")
    print("=" * 90)
    all_valid = [r for r in rows if isinstance(r["sim_B"], (int, float))]
    all_valid.sort(key=lambda r: r["sim_B"], reverse=True)
    for r in all_valid[:10]:
        safe = str(r["whisper_hyp"])[:45].encode("ascii", "replace").decode("ascii")
        print(f"{r['noise_level']:>6.2f}  {r['sway_coef']:>+6.1f}  "
              f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
              f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}  \"{safe}\"")
    print("=" * 90)


def _save_best_table(rows, out_dir, wer_threshold=20.0):
    """Write best_combinations.txt to disk."""
    lines = []
    lines.append(f"Method A -- Best Combinations (WER <= {wer_threshold}%)\n")
    lines.append("=" * 90 + "\n")
    lines.append(f"{'alpha':>6}  {'sway':>6}  {'WER%':>6}  {'SIM-A':>7}  "
                 f"{'SIM-B':>7}  {'MCD-A':>7}  Whisper\n")
    lines.append("-" * 90 + "\n")

    good = [r for r in rows
            if isinstance(r["wer"], (int, float)) and r["wer"] <= wer_threshold
            and isinstance(r["sim_B"], (int, float))]
    good.sort(key=lambda r: r["sim_B"], reverse=True)

    for r in good:
        safe = str(r["whisper_hyp"]).encode("ascii", "replace").decode("ascii")
        lines.append(f"{r['noise_level']:>6.2f}  {r['sway_coef']:>+6.1f}  "
                     f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
                     f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}  "
                     f"\"{safe}\"\n")

    path = out_dir / "best_combinations.txt"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] Best table -> {path}")


# ---------------------------------------------------------------------------
# 9. Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    device  = args.device
    out_dir = PROJECT_ROOT / "results" / "extension_2" / "method_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    alphas = [0.0] + [round(a, 2) for a in np.arange(0.05, 0.71, 0.05)]
    sways  = [round(s, 1) for s in np.arange(-1.0, 1.01, 0.1)]

    print("\n" + "=" * 70)
    print("Method A -- SDEdit Fine-Grained Sweep")
    print("=" * 70)
    print(f"  alphas : {alphas}")
    print(f"  sways  : {sways}")
    print(f"  Total  : {len(alphas)} x {len(sways)} = {len(alphas)*len(sways)} runs")

    # --- Load models ---
    print("\n[1/4] Loading F5-TTS model ...")
    model, vocoder = load_f5tts(device=device)
    mel_spec = model.mel_spec
    print("      Done.")

    print("      Loading Whisper (WER) ...")
    asr = load_whisper(device)

    print("      Loading WavLM (speaker similarity) ...")
    wavlm_fe, wavlm_model = load_wavlm(device)

    # --- Load reference audios ---
    id_path = Path(args.identity)
    em_path = Path(args.emotion)
    print(f"\n[2/4] Loading reference audios ...")
    print(f"      Identity A : {id_path.name}")
    print(f"      Style    B : {em_path.name}")

    audio_A = load_wav_simple(str(id_path)).to(device)
    audio_B = load_wav_simple(str(em_path)).to(device)
    mel_A   = audio_to_cond_mel(audio_A, mel_spec, device)
    mel_B   = audio_to_cond_mel(audio_B, mel_spec, device)
    print(f"      mel_A: {list(mel_A.shape)}  mel_B: {list(mel_B.shape)}")

    gen_text   = args.text
    ref_text_A = args.ref_text_id

    # --- Load existing CSV for resume ---
    csv_path  = out_dir / "results_metrics.csv"
    fieldnames = ["method", "noise_level", "sway_coef", "tag", "output_wav",
                  "duration_s", "peak_amp", "wer", "whisper_hyp",
                  "sim_A", "sim_B", "mcd_A"]

    existing = {}
    if args.resume and csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[row["tag"]] = row
        print(f"\n[resume] Loaded {len(existing)} existing rows from CSV.")

    # --- Inference sweep ---
    print(f"\n[3/4] Inference sweep ...")
    print(f"      Generate: \"{gen_text}\"")

    rows = []
    total = len(alphas) * len(sways)
    done  = 0

    for alpha, sway in itertools.product(alphas, sways):
        done += 1
        tag     = f"a{alpha:.2f}_s{sway:+.1f}".replace("+", "p").replace("-", "n")
        out_wav = audio_dir / f"{tag}.wav"

        # Resume: skip if WAV already exists and metrics are recorded
        if args.resume and tag in existing and out_wav.exists():
            rows.append({
                "method":     "noise_inject",
                "noise_level": float(existing[tag]["noise_level"]),
                "sway_coef":   float(existing[tag]["sway_coef"]),
                "tag":         tag,
                "output_wav":  str(out_wav),
                "duration_s":  float(existing[tag].get("duration_s", 0)),
                "peak_amp":    float(existing[tag].get("peak_amp", 0)),
                "wer":         _try_float(existing[tag].get("wer", "")),
                "whisper_hyp": existing[tag].get("whisper_hyp", ""),
                "sim_A":       _try_float(existing[tag].get("sim_A", "")),
                "sim_B":       _try_float(existing[tag].get("sim_B", "")),
                "mcd_A":       _try_float(existing[tag].get("mcd_A", "")),
            })
            print(f"  [{done:3d}/{total}] {tag}  (resumed)")
            continue

        print(f"\n  [{done:3d}/{total}] alpha={alpha:.2f}  sway={sway:+.1f}  -> {out_wav.name}")

        try:
            wav, sr = infer_noise_inject(
                mel_A, mel_B,
                ref_text_A=ref_text_A,
                gen_text=gen_text,
                model=model,
                vocoder=vocoder,
                device=device,
                noise_level=alpha,
                nfe_step=args.nfe,
                cfg_strength=args.cfg,
                sway_coef=sway,
                seed=42,
            )
            sf.write(str(out_wav), wav, sr)
            dur  = len(wav) / sr
            peak = float(np.abs(wav).max())
            print(f"     saved  dur={dur:.2f}s  peak={peak:.3f}")
        except Exception as e:
            print(f"     [ERROR] {e}")
            wav, sr = np.zeros(TARGET_SR, dtype=np.float32), TARGET_SR
            sf.write(str(out_wav), wav, sr)
            dur, peak = 0.0, 0.0

        rows.append({
            "method":     "noise_inject",
            "noise_level": alpha,
            "sway_coef":   sway,
            "tag":         tag,
            "output_wav":  str(out_wav),
            "duration_s":  round(dur, 3),
            "peak_amp":    round(peak, 4),
            "wer":         "",
            "whisper_hyp": "",
            "sim_A":       "",
            "sim_B":       "",
            "mcd_A":       "",
        })

    # --- Metrics ---
    print(f"\n[4/4] Computing metrics for {len(rows)} files ...")

    for i, row in enumerate(rows):
        tag      = row["tag"]
        wav_path = row["output_wav"]

        # Skip if ALL metrics are already valid numbers
        has_wer  = isinstance(row["wer"],  (int, float))
        has_simA = isinstance(row["sim_A"], (int, float))
        has_simB = isinstance(row["sim_B"], (int, float))
        has_mcd  = isinstance(row["mcd_A"], (int, float))

        if has_wer and has_simA and has_simB and has_mcd:
            print(f"  [{i+1:3d}/{len(rows)}] {tag}  (metrics already loaded)")
            continue

        print(f"  [{i+1:3d}/{len(rows)}] {tag}")

        # Only recompute missing metrics
        if not has_wer:
            wer_val, hyp = compute_wer(wav_path, gen_text, asr)
            row["wer"]        = wer_val if wer_val >= 0   else "err"
            row["whisper_hyp"] = hyp
        if not has_simA or not has_simB:
            sim_A, sim_B = compute_sims(wav_path, str(id_path), str(em_path), wavlm_fe, wavlm_model, device)
            row["sim_A"]      = sim_A   if sim_A   > -0.5 else "err"
            row["sim_B"]      = sim_B   if sim_B   > -0.5 else "err"
        if not has_mcd:
            mcd_A        = compute_mcd(wav_path, str(id_path))
            row["mcd_A"]      = mcd_A   if mcd_A   >= 0   else "err"

        safe_hyp = str(row.get("whisper_hyp", "")).encode("ascii", "replace").decode("ascii")
        print(f"     WER={row['wer']}%  SIM-A={row['sim_A']}  "
              f"SIM-B={row['sim_B']}  MCD-A={row['mcd_A']}")
        print(f"     Whisper: \"{safe_hyp}\"")

        # Incremental CSV save after every 10 completed rows
        if (i + 1) % 10 == 0:
            _save_csv(rows, csv_path, fieldnames)
            print(f"     [checkpoint] CSV saved ({i+1}/{len(rows)} metrics done)")

    # --- Final CSV save ---
    _save_csv(rows, csv_path, fieldnames)
    print(f"\n[OK] Results CSV -> {csv_path}")

    # --- Plots ---
    _plot_heatmaps(rows, alphas, sways, out_dir)
    _plot_tradeoff(rows, out_dir)

    # --- Best-combination analysis ---
    _print_best_table(rows, wer_threshold=20.0)
    _save_best_table(rows, out_dir, wer_threshold=20.0)

    print(f"\n[Done] All outputs in {out_dir}")
    return rows


def _try_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return val if val else ""


def _save_csv(rows, csv_path, fieldnames):
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 10. Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    f5_examples = (PROJECT_ROOT / "F5-TTS" / "src" / "f5_tts"
                   / "infer" / "examples" / "basic")
    default_id = str(f5_examples / "basic_ref_en.wav")
    default_em = str(f5_examples / "basic_ref_zh.wav")

    parser = argparse.ArgumentParser(
        description="Method A — SDEdit noise-injection sweep"
    )
    parser.add_argument("--identity", default=default_id)
    parser.add_argument("--emotion",  default=default_em)
    parser.add_argument("--text",
                        default=(
                            "I don't really care what you call me. "
                            "I've been a silent spectator, watching species "
                            "evolve, empires rise and fall."
                        ))
    parser.add_argument("--ref_text_id",
                        default="Some call me nature, others call me mother nature.")
    parser.add_argument("--nfe",    type=int,   default=32)
    parser.add_argument("--cfg",    type=float, default=2.0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Skip WAVs that already exist and have metrics in the CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device       : {args.device}")
    print(f"Identity ref : {Path(args.identity).name}")
    print(f"Style ref    : {Path(args.emotion).name}")
    print(f"Resume       : {args.resume}")
    run_sweep(args)
