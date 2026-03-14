"""
Style Guidance — 2-Pass ODE Extrapolation (Method B)
=====================================================

At each ODE timestep, the transformer is evaluated twice:
  vf_A = transformer(x, cond=mel_A, text, t)   # identity conditioning
  vf_B = transformer(x, cond=mel_B, text, t)   # style conditioning

The combined vector field is:
  vf = vf_A + guidance_scale * (vf_B - vf_A)

This is a direction-in-output-space extrapolation, analogous to CFG but
along the identity→style axis rather than cond→uncond.

  guidance_scale = 0.0 : pure identity baseline (reproduces vanilla F5-TTS)
  guidance_scale = 1.0 : equal blend at every ODE step
  guidance_scale > 1.0 : style over-expression (extrapolation beyond B)

The starting point x_0 is pure Gaussian noise (no mel_B blending) —
this method acts entirely through the vector field, not through x_0.

Sweep:
  guidance_scale in {0.0, 0.5, 1.0, 1.5, 2.0}
  sway_coef      in {-1.0, 0.0}

Cost: 2× transformer evaluations per ODE step compared to Method A.

Identity A: basic_ref_en.wav   (English female, F5-TTS canonical)
Style    B: basic_ref_zh.wav   (Mandarin animated-character, Ne Zha film)

Usage:
  .venv/Scripts/python scripts/run_style_guidance.py
  .venv/Scripts/python scripts/run_style_guidance.py --device cpu
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
    return mel.permute(0, 2, 1)  # [1, T_mel, 100]


# ---------------------------------------------------------------------------
# 5. ODE state setup helper
# ---------------------------------------------------------------------------

def _build_step_cond(mel_3d, max_duration, dtype, device):
    """Pad mel conditioning tensor and build the masked step_cond."""
    from f5_tts.model.utils import lens_to_mask

    cond_seq_len = mel_3d.shape[1]
    cond = mel_3d.to(dtype)
    lens = torch.full((1,), cond_seq_len, device=device, dtype=torch.long)
    cond_mask = lens_to_mask(lens)                              # [1, T]
    cond_padded = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
    cond_mask_padded = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
    cond_mask_3d = cond_mask_padded.unsqueeze(-1)               # [1, T, 1]
    step_cond = torch.where(cond_mask_3d, cond_padded, torch.zeros_like(cond_padded))
    return step_cond, cond_padded, cond_mask_3d


def _compute_max_duration(mel_A_3d, ref_text_A, gen_text, text_tensor, device):
    """Estimate output duration in mel frames (mirrors cfm.py logic)."""
    ref_audio_len = mel_A_3d.shape[1]
    ref_bytes = max(len(ref_text_A.encode("utf-8")), 1)
    gen_bytes = len(gen_text.encode("utf-8"))
    gen_len = int(ref_audio_len / ref_bytes * gen_bytes)
    duration_val = ref_audio_len + gen_len

    duration = torch.full((1,), duration_val, device=device, dtype=torch.long)
    # Ensure duration ≥ text length + 1 (cfm.py constraint)
    text_len = int((text_tensor != -1).sum().item())
    min_dur = max(text_len, ref_audio_len) + 1
    duration = duration.clamp(min=min_dur)
    return int(duration.item())


# ---------------------------------------------------------------------------
# 6. Style-guidance inference (2-pass ODE)
# ---------------------------------------------------------------------------

def infer_style_guidance(
    mel_A_3d: torch.Tensor,
    mel_B_3d: torch.Tensor,
    ref_text_A: str,
    gen_text: str,
    model,
    vocoder,
    device: str,
    guidance_scale: float = 1.0,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_coef: float | None = None,
    seed: int = 42,
) -> tuple:
    """
    Style guidance via 2-pass ODE extrapolation.

    At each ODE step:
      vf_A = transformer(x, cond=mel_A)   ← identity direction
      vf_B = transformer(x, cond=mel_B)   ← style direction
      vf   = vf_A + guidance_scale * (vf_B - vf_A)

    guidance_scale=0: reduces to pure identity (no style)
    guidance_scale=1: equal weighting of both fields
    guidance_scale>1: extrapolates beyond B's style

    Returns (wav_numpy, sample_rate).
    """
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx, list_str_to_tensor
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
    max_duration = _compute_max_duration(mel_A_3d, ref_text_A, gen_text, text, device)
    ref_audio_len = mel_A_3d.shape[1]
    dtype = next(model.parameters()).dtype

    # --- Build step_cond for identity (A) and style (B) ---
    step_cond_A, cond_A_padded, cond_mask_A_3d = _build_step_cond(
        mel_A_3d, max_duration, dtype, device)
    step_cond_B, _, _ = _build_step_cond(
        mel_B_3d, max_duration, dtype, device)

    # --- Initial noise (pure Gaussian — no mel_B blending) ---
    torch.manual_seed(seed)
    y0 = torch.randn(max_duration, model.num_channels,
                     device=device, dtype=dtype).unsqueeze(0)  # [1, T, 100]

    # --- Timesteps with optional sway ---
    t_steps = torch.linspace(0.0, 1.0, nfe_step + 1, device=device, dtype=dtype)
    if sway_coef is not None:
        t_steps = t_steps + sway_coef * (
            torch.cos(torch.pi / 2 * t_steps) - 1 + t_steps)

    mask = None  # batch=1 → no mask for efficiency

    def fn(t, x):
        """2-pass vector field: extrapolate from identity to style direction."""
        if cfg_strength < 1e-5:
            vf_A = model.transformer(
                x=x, cond=step_cond_A, text=text, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False, cache=False)
            vf_B = model.transformer(
                x=x, cond=step_cond_B, text=text, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False, cache=False)
        else:
            # Identity conditioning with CFG
            pred_cfg_A = model.transformer(
                x=x, cond=step_cond_A, text=text, time=t, mask=mask,
                cfg_infer=True, cache=False)
            pred_A, null_A = torch.chunk(pred_cfg_A, 2, dim=0)
            vf_A = pred_A + (pred_A - null_A) * cfg_strength

            # Style conditioning with CFG
            pred_cfg_B = model.transformer(
                x=x, cond=step_cond_B, text=text, time=t, mask=mask,
                cfg_infer=True, cache=False)
            pred_B, null_B = torch.chunk(pred_cfg_B, 2, dim=0)
            vf_B = pred_B + (pred_B - null_B) * cfg_strength

        # Extrapolate: identity + guidance_scale * (style - identity)
        return vf_A + guidance_scale * (vf_B - vf_A)

    with torch.inference_mode():
        trajectory = odeint(fn, y0, t_steps, method="euler")

    model.transformer.clear_cache()

    # --- Post-process ---
    out = trajectory[-1]                                         # [1, T, 100]
    out = torch.where(cond_mask_A_3d, cond_A_padded, out)       # restore ref region

    out = out.to(torch.float32)
    generated = out[:, ref_audio_len:, :]                       # strip ref portion
    generated = generated.permute(0, 2, 1)                      # [1, 100, gen_len]

    with torch.no_grad():
        generated_wave = vocoder.decode(generated)

    wav = generated_wave.squeeze().cpu().numpy()
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav, TARGET_SR


# ---------------------------------------------------------------------------
# 7. Metrics  (identical to run_noise_inject.py)
# ---------------------------------------------------------------------------

def load_whisper(device):
    """Load Whisper ASR pipeline once at startup. Crashes immediately if it fails."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-large-v3-turbo",
                   device=0 if device == "cuda" else -1)
    print(f"  Whisper loaded on {device}")
    return asr


def compute_wer(wav_path: str, expected_text: str, asr) -> tuple:
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


def compute_sims(gen_wav: str, ref_A: str, ref_B: str, fe, wlm, device: str) -> tuple:
    e_gen = _wavlm_embed(gen_wav, fe, wlm, device)
    e_A   = _wavlm_embed(ref_A,   fe, wlm, device)
    e_B   = _wavlm_embed(ref_B,   fe, wlm, device)
    return round(float((e_gen * e_A).sum().item()), 4), \
           round(float((e_gen * e_B).sum().item()), 4)


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
# 8. Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    device = args.device
    out_dir = PROJECT_ROOT / "results" / "extension_2" / "method_B"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("Style Guidance — 2-Pass ODE Extrapolation (Method B)")
    print("=" * 70)

    print("\n[1/4] Loading F5-TTS model ...")
    model, vocoder = load_f5tts(device=device)
    mel_spec = model.mel_spec
    print("      Done.")

    asr = load_whisper(device)
    wavlm_fe, wavlm_model = load_wavlm(device)

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

    # Copy reference audios to audio dir for later metric computation
    import shutil
    for src, dst_name in [(id_path, f"ref_A_{id_path.stem}.wav"),
                          (em_path, f"ref_B_{em_path.stem}.wav")]:
        dst = audio_dir / dst_name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    gen_text        = args.text
    ref_text_A      = args.ref_text_id
    guidance_scales = args.guidance_scales
    sway_coefs      = args.sways

    print(f"\n[3/4] Running sweep ...")
    print(f"      Generate       : \"{gen_text}\"")
    print(f"      guidance_scales: {guidance_scales}")
    print(f"      sway_coefs     : {sway_coefs}")
    print(f"      Total runs     : {len(guidance_scales) * len(sway_coefs)}")

    rows = []

    for gs, sway in itertools.product(guidance_scales, sway_coefs):
        tag = (f"gs{gs:.2f}_s{sway:+.1f}"
               .replace("+", "p").replace("-", "n"))
        out_wav = audio_dir / f"{tag}.wav"
        print(f"\n  guidance_scale={gs:.2f}  sway={sway:+.1f}  -> {out_wav.name}")

        try:
            wav, sr = infer_style_guidance(
                mel_A, mel_B,
                ref_text_A=ref_text_A,
                gen_text=gen_text,
                model=model,
                vocoder=vocoder,
                device=device,
                guidance_scale=gs,
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
            "method": "style_guidance",
            "guidance_scale": gs,
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
        wer_val, hyp = compute_wer(wav_path, gen_text, asr)
        sim_A, sim_B = compute_sims(wav_path, str(id_path), str(em_path), wavlm_fe, wavlm_model, device)
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

    # Save CSV
    csv_path = out_dir / "results_metrics.csv"
    fieldnames = ["method", "guidance_scale", "sway_coef", "tag", "output_wav",
                  "duration_s", "peak_amp", "wer", "whisper_hyp", "sim_A", "sim_B", "mcd_A"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Results CSV -> {csv_path}")

    _plot_results(rows, guidance_scales, sway_coefs, out_dir)
    _print_summary(rows)
    return rows


def _plot_results(rows, guidance_scales, sway_coefs, out_dir):
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
            "Style Guidance — 2-Pass ODE Extrapolation (Method B)\n"
            "Identity A = basic_ref_en  |  Style B = basic_ref_zh\n"
            "Each line = one sway coefficient;  x-axis = guidance_scale",
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
                xs = [r["guidance_scale"] for r in subset]
                ys = [r[metric] for r in subset]
                ax.plot(xs, ys, marker=markers[i % len(markers)],
                        label=f"sway={sway:+.1f}")
            ax.set_xlabel("guidance_scale")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "method_B_sweep.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"[OK] Plot -> {plot_path}")
    except Exception as e:
        print(f"[plot] {e}")


def _print_summary(rows):
    print("\n" + "=" * 70)
    print(f"{'gs':>5}  {'sway':>6}  {'WER%':>6}  {'SIM-A':>7}  "
          f"{'SIM-B':>7}  {'MCD-A':>7}  Whisper")
    print("-" * 70)
    for r in rows:
        safe = str(r['whisper_hyp'])[:40].encode("ascii", "replace").decode("ascii")
        print(f"{r['guidance_scale']:>5.2f}  {r['sway_coef']:>+6.1f}  "
              f"{str(r['wer']):>6}  {str(r['sim_A']):>7}  "
              f"{str(r['sim_B']):>7}  {str(r['mcd_A']):>7}  "
              f"\"{safe}\"")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 9. Argument parsing and entry point
# ---------------------------------------------------------------------------

def parse_args():
    f5_examples = (PROJECT_ROOT / "F5-TTS" / "src" / "f5_tts"
                   / "infer" / "examples" / "basic")
    default_id = str(f5_examples / "basic_ref_en.wav")
    default_em = str(f5_examples / "basic_ref_zh.wav")

    parser = argparse.ArgumentParser(
        description="Style Guidance sweep (Method B)"
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
    parser.add_argument("--guidance_scales", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0])
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
