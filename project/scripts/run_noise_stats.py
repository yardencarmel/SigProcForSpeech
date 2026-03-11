"""
Noise Statistics Transfer — Method D (Lightweight Baseline)
============================================================

A softer alternative to SDEdit (Method A): instead of directly blending
mel_B frames into x_0, we only transfer mel_B's global spectral statistics
(mean and standard deviation) to the starting noise.

Starting noise construction:
  randn    ~ N(0, 1)  [pure Gaussian, same shape as output]
  x_0 = (randn / randn.std()) * target_std + alpha * mel_B.mean()

  where:
    target_std = alpha * mel_B.std() + (1 - alpha) * randn.std()

  alpha = 0.0 : pure N(0,1) noise, no style information (baseline)
  alpha > 0   : noise rescaled to carry mel_B's spectral "envelope shape"
                without copying specific frames from B

Difference from Method A (SDEdit):
  - Method A blends actual mel_B frames: x_0 = (1-α)*randn + α*mel_B
    → specific prosodic patterns of B leak into x_0
  - Method D transfers only summary statistics:
    → only the amplitude envelope of B influences the noise distribution
    → finer temporal structure of B does NOT leak in

Hypothesis: Method D is a weaker prior than Method A, so SIM-B gains will
be smaller, but it may preserve intelligibility better at larger alpha.

Sweep:
  noise_level (alpha) in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}
  sway_coef           in {-1.0, 0.0}

Identity A: basic_ref_en.wav
Style    B: basic_ref_zh.wav

Usage:
  .venv/Scripts/python scripts/run_noise_stats.py
  .venv/Scripts/python scripts/run_noise_stats.py --device cpu
"""

import os
import sys
import csv
import argparse
import itertools
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Environment patches
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

# ---------------------------------------------------------------------------
# 2. Audio helpers
# ---------------------------------------------------------------------------
TARGET_SR = 24_000
TARGET_RMS = 0.1


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
    return audio


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
    return mel.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# 5. Noise statistics transfer inference
# ---------------------------------------------------------------------------

def infer_noise_stats(
    mel_A_3d: torch.Tensor,
    mel_B_3d: torch.Tensor,
    ref_text_A: str,
    gen_text: str,
    model,
    vocoder,
    device: str,
    noise_level: float = 0.0,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_coef: float | None = None,
    seed: int = 42,
) -> tuple:
    """
    Noise statistics transfer: rescale x_0 to match mel_B's statistics.

    x_0 = (randn / randn.std()) * target_std + alpha * mel_B.mean()
    target_std = alpha * mel_B.std() + (1 - alpha) * randn_native.std()

    The ODE is then integrated from t=0 to t=1, conditioned entirely on
    mel_A (identity). Only the noise distribution is modified — no direct
    mel_B frame mixing, no vector-field modification.

    Unlike Method A, this is a statistics-only prior: the temporal structure
    of mel_B does not leak into x_0.

    Returns (wav_numpy, sample_rate).
    """
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx, list_str_to_tensor, lens_to_mask
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
    gen_len = int(ref_audio_len / ref_bytes * gen_bytes)
    duration_val = ref_audio_len + gen_len
    duration = torch.full((1,), duration_val, device=device, dtype=torch.long)
    text_len = int((text != -1).sum().item())
    min_dur = max(text_len, ref_audio_len) + 1
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

    # --- Build noise statistics from mel_B ---
    # Pad/crop mel_B to max_duration for statistics computation
    cond_B = mel_B_3d.to(dtype)
    if cond_B.shape[1] < max_duration:
        mel_B_padded = F.pad(cond_B, (0, 0, 0, max_duration - cond_B.shape[1]), value=0.0)
    else:
        mel_B_padded = cond_B[:, :max_duration, :]  # [1, T, 100]

    # Use only the non-zero frames of mel_B for statistics
    mel_B_vals = mel_B_3d.squeeze(0)  # [T_B, 100]  (original, unpadded)
    mel_b_std  = float(mel_B_vals.std().item())
    mel_b_mean = float(mel_B_vals.mean().item())

    # --- Custom y0: statistics-transferred noise ---
    torch.manual_seed(seed)
    randn = torch.randn(max_duration, model.num_channels, device=device, dtype=dtype)
    native_std = float(randn.std().item())

    if noise_level > 0.0:
        # Transfer spectral envelope statistics of B to the noise
        target_std = noise_level * mel_b_std + (1.0 - noise_level) * native_std
        y0 = (randn / (randn.std() + 1e-8)) * target_std + noise_level * mel_b_mean
    else:
        # Pure Gaussian baseline
        y0 = randn

    y0 = y0.unsqueeze(0)  # [1, max_duration, 100]

    # --- Timesteps (always full 0→1, no t_start shortcut) ---
    t_steps = torch.linspace(0.0, 1.0, nfe_step + 1, device=device, dtype=dtype)
    if sway_coef is not None:
        t_steps = t_steps + sway_coef * (
            torch.cos(torch.pi / 2 * t_steps) - 1 + t_steps)

    mask = None  # batch=1

    def fn(t, x):
        """Standard identity-conditioned vector field (same as baseline)."""
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

    wav = generated_wave.squeeze().cpu().numpy()
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav, TARGET_SR


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def compute_wer(wav_path: str, expected_text: str, device: str) -> tuple:
    try:
        import re
        from transformers import pipeline
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(1)
        pipe = pipeline("automatic-speech-recognition",
                        model="openai/whisper-large-v3-turbo",
                        device=0 if device == "cuda" else -1)
        result = pipe({"sampling_rate": sr, "raw": data})
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


def _wavlm_embed(wav_path: str, fe, wavlm_model, device: str):
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


def compute_sims(gen_wav: str, ref_A: str, ref_B: str, device: str) -> tuple:
    try:
        from transformers import WavLMModel, AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        wlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
        wlm.eval()
        e_gen = _wavlm_embed(gen_wav, fe, wlm, device)
        e_A   = _wavlm_embed(ref_A,   fe, wlm, device)
        e_B   = _wavlm_embed(ref_B,   fe, wlm, device)
        return round(float((e_gen * e_A).sum().item()), 4), \
               round(float((e_gen * e_B).sum().item()), 4)
    except Exception as e:
        print(f"  [SIM error] {e}")
        return -1.0, -1.0


def compute_mcd(gen_wav: str, ref_wav: str) -> float:
    try:
        import librosa
        y_gen, _ = librosa.load(str(gen_wav), sr=24000)
        y_ref, _ = librosa.load(str(ref_wav), sr=24000)
        m_gen = librosa.feature.mfcc(y=y_gen, sr=24000, n_mfcc=25)[1:]
        m_ref = librosa.feature.mfcc(y=y_ref, sr=24000, n_mfcc=25)[1:]
        T = min(m_gen.shape[1], m_ref.shape[1])
        diff = m_gen[:, :T] - m_ref[:, :T]
        return round(float((10 / np.log(10)) * np.sqrt(2 * (diff ** 2).sum(0)).mean()), 2)
    except Exception as e:
        print(f"  [MCD error] {e}")
        return -1.0


# ---------------------------------------------------------------------------
# 7. Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    device = args.device
    out_dir = PROJECT_ROOT / "results" / "noise_inject" / "method_D"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("Noise Statistics Transfer (Method D)")
    print("=" * 70)

    print("\n[1/4] Loading F5-TTS model ...")
    model, vocoder = load_f5tts(device=device)
    mel_spec = model.mel_spec
    print("      Done.")

    id_path = Path(args.identity)
    em_path = Path(args.emotion)
    print(f"\n[2/4] Loading reference audios ...")
    print(f"      Identity A : {id_path.name}")
    print(f"      Style    B : {em_path.name}")

    audio_A = load_wav_simple(str(id_path)).to(device)
    audio_B = load_wav_simple(str(em_path)).to(device)
    mel_A = audio_to_cond_mel(audio_A, mel_spec, device)
    mel_B = audio_to_cond_mel(audio_B, mel_spec, device)
    print(f"      mel_A: {list(mel_A.shape)}  mel_B: {list(mel_B.shape)}")
    print(f"      mel_B stats: mean={mel_B.mean():.4f}  std={mel_B.std():.4f}")

    import shutil
    for src, dst_name in [(id_path, f"ref_A_{id_path.stem}.wav"),
                          (em_path, f"ref_B_{em_path.stem}.wav")]:
        dst = audio_dir / dst_name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    gen_text     = args.text
    ref_text_A   = args.ref_text_id
    noise_levels = args.noise_levels
    sway_coefs   = args.sways

    print(f"\n[3/4] Running sweep ...")
    print(f"      Generate    : \"{gen_text}\"")
    print(f"      noise_levels: {noise_levels}")
    print(f"      sway_coefs  : {sway_coefs}")
    print(f"      Total runs  : {len(noise_levels) * len(sway_coefs)}")

    rows = []

    for alpha, sway in itertools.product(noise_levels, sway_coefs):
        tag = (f"a{alpha:.2f}_s{sway:+.1f}"
               .replace("+", "p").replace("-", "n"))
        out_wav = audio_dir / f"{tag}.wav"
        print(f"\n  alpha={alpha:.2f}  sway={sway:+.1f}  -> {out_wav.name}")

        try:
            wav, sr = infer_noise_stats(
                mel_A, mel_B,
                ref_text_A=ref_text_A,
                gen_text=gen_text,
                model=model,
                vocoder=vocoder,
                device=device,
                noise_level=alpha,
                nfe_step=args.nfe,
                cfg_strength=args.cfg,
                sway_coef=sway if sway != 0.0 else None,
                seed=42,
            )
            sf.write(str(out_wav), wav, sr)
            dur = len(wav) / sr
            peak = float(np.abs(wav).max())
            print(f"     saved  dur={dur:.2f}s  peak={peak:.3f}")
        except Exception as e:
            print(f"     [ERROR] {e}")
            import traceback; traceback.print_exc()
            wav = np.zeros(TARGET_SR, dtype=np.float32)
            sf.write(str(out_wav), wav, TARGET_SR)
            dur, peak = 0.0, 0.0

        rows.append({
            "method": "noise_stats",
            "noise_level": alpha,
            "sway_coef": sway,
            "tag": tag,
            "output_wav": str(out_wav),
            "duration_s": round(dur, 3),
            "peak_amp": round(peak, 4),
            "wer": "",
            "whisper_hyp": "",
            "sim_A": "",
            "sim_B": "",
            "mcd_A": "",
        })

    print("\n[4/4] Computing metrics ...")
    for row in rows:
        tag = row["tag"]
        wav_path = row["output_wav"]
        print(f"  {tag}")
        wer_val, hyp = compute_wer(wav_path, gen_text, device)
        sim_A, sim_B = compute_sims(wav_path, str(id_path), str(em_path), device)
        mcd_A = compute_mcd(wav_path, str(id_path))
        row["wer"]         = wer_val if wer_val >= 0   else "err"
        row["whisper_hyp"] = hyp
        row["sim_A"]       = sim_A   if sim_A > -0.5   else "err"
        row["sim_B"]       = sim_B   if sim_B > -0.5   else "err"
        row["mcd_A"]       = mcd_A   if mcd_A >= 0     else "err"
        safe_hyp = hyp.encode("ascii", "replace").decode("ascii")
        print(f"     WER={row['wer']}%  SIM-A={row['sim_A']}  "
              f"SIM-B={row['sim_B']}  MCD-A={row['mcd_A']}")
        print(f"     Whisper: \"{safe_hyp}\"")

    csv_path = out_dir / "results_metrics.csv"
    fieldnames = ["method", "noise_level", "sway_coef", "tag", "output_wav",
                  "duration_s", "peak_amp", "wer", "whisper_hyp", "sim_A", "sim_B", "mcd_A"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Results CSV -> {csv_path}")

    _plot_results(rows, noise_levels, sway_coefs, out_dir)
    _print_summary(rows)
    return rows


def _plot_results(rows, noise_levels, sway_coefs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = [
            ("wer",   "WER (%)",             "red"),
            ("sim_A", "SIM-A (identity)",    "blue"),
            ("sim_B", "SIM-B (style)",       "orange"),
            ("mcd_A", "MCD-A (vs identity)", "green"),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(7, 18))
        fig.suptitle(
            "Noise Statistics Transfer (Method D)\n"
            "Identity A = basic_ref_en  |  Style B = basic_ref_zh\n"
            "Each line = one sway coefficient;  x-axis = noise_level (α)",
            fontsize=11,
        )
        axes = axes.flatten()
        markers = ["o", "s", "^", "D"]

        for ax, (metric, ylabel, color) in zip(axes, metrics):
            for i, sway in enumerate(sway_coefs):
                subset = [r for r in rows if r["sway_coef"] == sway
                          and isinstance(r[metric], (int, float))]
                if not subset:
                    continue
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
        plot_path = out_dir / "method_D_sweep.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"[OK] Plot -> {plot_path}")
    except Exception as e:
        print(f"[plot] {e}")


def _print_summary(rows):
    print("\n" + "=" * 70)
    print(f"{'alpha':>6}  {'sway':>6}  {'WER%':>6}  {'SIM-A':>7}  "
          f"{'SIM-B':>7}  {'MCD-A':>7}  Whisper")
    print("-" * 70)
    for r in rows:
        safe = str(r['whisper_hyp'])[:40].encode("ascii", "replace").decode("ascii")
        print(f"{r['noise_level']:>6.2f}  {r['sway_coef']:>+6.1f}  "
              f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
              f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}  "
              f"\"{safe}\"")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 8. Argument parsing and entry point
# ---------------------------------------------------------------------------

def parse_args():
    f5_examples = (PROJECT_ROOT / "F5-TTS" / "src" / "f5_tts"
                   / "infer" / "examples" / "basic")
    default_id = str(f5_examples / "basic_ref_en.wav")
    default_em = str(f5_examples / "basic_ref_zh.wav")

    parser = argparse.ArgumentParser(
        description="Noise Statistics Transfer sweep (Method D)"
    )
    parser.add_argument("--identity", default=default_id)
    parser.add_argument("--emotion", default=default_em)
    parser.add_argument("--text",
                        default=(
                            "I don't really care what you call me. "
                            "I've been a silent spectator, watching species "
                            "evolve, empires rise and fall."
                        ))
    parser.add_argument("--ref_text_id",
                        default="Some call me nature, others call me mother nature.")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--sways", type=float, nargs="+",
                        default=[-1.0, 0.0])
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device       : {args.device}")
    print(f"Identity ref : {Path(args.identity).name}")
    print(f"Style ref    : {Path(args.emotion).name}")
    run_sweep(args)
