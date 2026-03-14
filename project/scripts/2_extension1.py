"""
Phase 4: Emotion / Style Transfer via Direct Mel Injection
===========================================================

Problem with v1
---------------
The previous approach:
  blend log-mel spectrograms → decode via vocos → write temp WAV → re-encode in model
produced clipped / garbled audio at intermediate weights because vocos cannot faithfully
invert arbitrary blended mels (out-of-distribution inputs saturate the decoder).

Key insight (cfm.py, lines 106-109)
------------------------------------
    if cond.ndim == 2:              # raw waveform  [B, samples]
        cond = self.mel_spec(cond)  # model converts to mel internally
        cond = cond.permute(0, 2, 1)
    # BUT if cond is already 3D → used directly as conditioning [B, time, channels]

So we can inject the blended mel DIRECTLY into model.sample() as a 3D tensor,
completely skipping the vocos inversion step.

Methods implemented
-------------------
1. Direct-mel-injection (v2):  blend mels → pass [1, T, 100] tensor as cond
2. Zero-shot emotion transfer: use emotion audio directly as the sole reference
   (the model clones both voice and emotion of the reference speaker)
3. Concatenated-reference:     identity + emotion audios concatenated in time,
   used as a single reference (partial signal mixing)

Usage
-----
Run the full ablation experiment:
    .venv/Scripts/python scripts/run_phase4.py

Optionally specify custom reference audios and text:
    .venv/Scripts/python scripts/run_phase4.py \
        --identity samples/identity.wav \
        --emotion  samples/emotion.wav \
        --text "Your sentence here." \
        --ref_text_id "What the identity speaker says." \
        --ref_text_em "What the emotion speaker says."
"""

import os
import sys
import csv
import json
import argparse
import tempfile
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

# Patch torchaudio.load with soundfile (avoids torchcodec DLL requirement)
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

# ---------------------------------------------------------------------------
# 2. Audio helpers
# ---------------------------------------------------------------------------

TARGET_SR = 24_000
TARGET_RMS = 0.1


def load_wav_normalized(path: str, target_sr: int = TARGET_SR,
                        target_rms: float = TARGET_RMS) -> torch.Tensor:
    """Load WAV, resample to target_sr, normalize RMS. Returns [1, T]."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())          # [channels, T]
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)      # mono
    if sr != target_sr:
        audio = torch.from_numpy(
            np.array(sf.SoundFile(str(path)).read(dtype="float32"))
        )
        import torchaudio.functional as F_audio
        audio = F_audio.resample(audio.unsqueeze(0) if audio.ndim == 1 else audio,
                                 sr, target_sr)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
    rms = torch.sqrt((audio ** 2).mean())
    if rms > 0:
        audio = audio * target_rms / rms
    return audio  # [1, T]


def load_wav_simple(path: str, target_sr: int = TARGET_SR,
                    target_rms: float = TARGET_RMS) -> torch.Tensor:
    """Simple load + normalize using soundfile."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())   # [C, T]
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != target_sr:
        # simple linear resample
        n_out = int(audio.shape[-1] * target_sr / sr)
        audio_np = audio.numpy()
        from scipy.signal import resample
        audio_np = resample(audio_np, n_out, axis=-1)
        audio = torch.from_numpy(audio_np.astype(np.float32))
    rms = torch.sqrt((audio ** 2).mean())
    if rms > 0:
        audio = audio * target_rms / rms
    return audio  # [1, T]

# ---------------------------------------------------------------------------
# 3. Core model loading
# ---------------------------------------------------------------------------

def load_f5tts(device="cuda"):
    """Load pretrained F5-TTS model and vocoder (new API)."""
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
# 4. Mel-space blending (the key innovation in v2)
# ---------------------------------------------------------------------------

def get_mel_spec_module(model):
    """Extract the mel spec module from the CFM model."""
    return model.mel_spec


def audio_to_cond_mel(audio_1xT: torch.Tensor, mel_spec_module,
                      device: str) -> torch.Tensor:
    """
    Convert normalized audio [1, T] to conditioning mel [1, T_mel, 100].
    Uses F5-TTS's own MelSpec module so the representation is exact.
    """
    with torch.no_grad():
        mel = mel_spec_module(audio_1xT.to(device))  # [1, 100, T_mel]
    return mel.permute(0, 2, 1)  # [1, T_mel, 100]


def blend_cond_mels(mel_id: torch.Tensor, mel_em: torch.Tensor,
                    weight: float) -> torch.Tensor:
    """
    Linear interpolation of two conditioning mel tensors [1, T, 100].
    Both are time-aligned (truncated to the shorter one).
    weight = 0.0 -> pure identity, 1.0 -> pure emotion.
    """
    T = min(mel_id.shape[1], mel_em.shape[1])
    m1 = mel_id[:, :T, :]
    m2 = mel_em[:, :T, :]
    return (1.0 - weight) * m1 + weight * m2


# ---------------------------------------------------------------------------
# 5. Inference with blended mel (direct injection, bypassing vocos inversion)
# ---------------------------------------------------------------------------

def infer_with_cond_mel(
    cond_mel_3d: torch.Tensor,   # [1, T_ref, 100]
    ref_text: str,               # transcript of the reference audio (for duration)
    gen_text: str,               # text to generate
    model,
    vocoder,
    device: str,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_coef: float = -1.0,
    speed: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Run F5-TTS generation with a pre-computed 3D conditioning mel tensor.

    Because cond_mel_3d.ndim == 3, cfm.py bypasses the internal mel_spec()
    call and uses it directly as conditioning — no vocos inversion needed.

    Returns (audio_np [samples], sample_rate).
    """
    from f5_tts.model.utils import convert_char_to_pinyin

    if seed is not None:
        torch.manual_seed(seed)

    cond_mel_3d = cond_mel_3d.to(device)
    ref_audio_len = cond_mel_3d.shape[1]   # number of mel frames

    # Text processing (same as F5-TTS internals)
    text_list = [ref_text + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    # Duration estimation (mirrors infer_batch_process formula)
    ref_bytes = max(len(ref_text.encode("utf-8")), 1)
    gen_bytes = len(gen_text.encode("utf-8"))
    gen_len = int(ref_audio_len / ref_bytes * gen_bytes / speed)
    duration = ref_audio_len + gen_len

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=cond_mel_3d,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_coef,
            seed=seed,
        )

    # Slice off reference portion and decode
    generated = generated.to(torch.float32)
    generated = generated[:, ref_audio_len:, :]  # [1, gen_len, 100]
    generated = generated.permute(0, 2, 1)       # [1, 100, gen_len]

    with torch.no_grad():
        generated_wave = vocoder.decode(generated)  # [1, samples]

    wav = generated_wave.squeeze().cpu().numpy()
    # Peak-normalize output
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav, TARGET_SR


# ---------------------------------------------------------------------------
# 6. Metrics helpers (re-use Phase 5 approach)
# ---------------------------------------------------------------------------

def load_whisper(device):
    """Load Whisper ASR pipeline once at startup. Crashes immediately if it fails."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-large-v3-turbo",
                   device=0 if device == "cuda" else -1)
    print(f"  Whisper loaded on {device}")
    return asr


def compute_wer(wav_path: str, expected_text: str, asr) -> float:
    """WER via pre-loaded Whisper pipeline. Loads audio with soundfile to bypass FFmpeg."""
    try:
        import re
        # Load audio with soundfile → pass numpy array to avoid FFmpeg dependency
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(1)
        result = asr({"sampling_rate": sr, "raw": data})
        hyp = result["text"].strip()
        # Edit-distance WER
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
        return round(100.0 * d[r][h] / max(r, 1), 1)
    except Exception as e:
        print(f"  [WER error] {e}")
        return -1.0


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


def _wavlm_embed(wav_path: str, fe, wlm, device: str):
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(1)
    if sr != 16000:
        from scipy.signal import resample
        data = resample(data, int(len(data) * 16000 / sr))
    inputs = fe(data, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        out = wlm(**inputs).last_hidden_state.mean(1)
    return out / out.norm()


def compute_sim(wav_path: str, ref_wav_path: str, fe, wlm, device: str) -> float:
    """Speaker similarity via WavLM cosine similarity."""
    e1 = _wavlm_embed(wav_path, fe, wlm, device)
    e2 = _wavlm_embed(ref_wav_path, fe, wlm, device)
    return float((e1 * e2).sum().item())


def compute_mcd(gen_wav: str, ref_wav: str) -> float:
    """Mel Cepstral Distortion (24-MFCC, skip C0)."""
    try:
        import librosa
        y_gen, sr_gen = librosa.load(str(gen_wav), sr=24000)
        y_ref, sr_ref = librosa.load(str(ref_wav), sr=24000)
        mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=sr_gen, n_mfcc=25)[1:]  # skip C0
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=25)[1:]
        T = min(mfcc_gen.shape[1], mfcc_ref.shape[1])
        diff = mfcc_gen[:, :T] - mfcc_ref[:, :T]
        mcd = (10 / np.log(10)) * np.sqrt(2 * (diff ** 2).sum(0)).mean()
        return float(mcd)
    except Exception as e:
        print(f"  [MCD error] {e}")
        return -1.0


# ---------------------------------------------------------------------------
# 7. Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    device = args.device
    out_dir = PROJECT_ROOT / "results" / "extension_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("Phase 4 — Emotion Transfer via Direct Mel Injection")
    print("=" * 70)

    # --- Load model ---
    print("\n[1/4] Loading F5-TTS model ...")
    model, vocoder = load_f5tts(device=device)
    mel_spec_module = get_mel_spec_module(model)
    print("      Model loaded.")

    asr = load_whisper(device)
    wavlm_fe, wavlm_model = load_wavlm(device)

    # --- Load reference audios ---
    id_path = Path(args.identity)
    em_path = Path(args.emotion)
    print(f"\n[2/4] Loading reference audios ...")
    print(f"      Identity : {id_path.name}")
    print(f"      Emotion  : {em_path.name}")

    audio_id = load_wav_simple(str(id_path)).to(device)   # [1, T_id]
    audio_em = load_wav_simple(str(em_path)).to(device)   # [1, T_em]

    mel_id = audio_to_cond_mel(audio_id, mel_spec_module, device)  # [1, T, 100]
    mel_em = audio_to_cond_mel(audio_em, mel_spec_module, device)  # [1, T, 100]
    print(f"      mel_id: {list(mel_id.shape)}  mel_em: {list(mel_em.shape)}")

    # --- Prepare text ---
    gen_text = args.text
    ref_text_id = args.ref_text_id
    ref_text_em = args.ref_text_em

    print(f"\n[3/4] Running ablation ...")
    print(f"      Generate: \"{gen_text}\"")

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []

    for w in weights:
        print(f"\n  -- weight={w:.2f} --")

        # V2: direct mel injection (the new method)
        mel_blend = blend_cond_mels(mel_id, mel_em, weight=w)
        # Choose ref_text: identity for w<0.5, emotion for w>0.5, identity for w=0.5
        ref_text = ref_text_id if w <= 0.5 else ref_text_em

        wav, sr = infer_with_cond_mel(
            mel_blend, ref_text, gen_text,
            model, vocoder, device,
            nfe_step=args.nfe, cfg_strength=args.cfg,
            sway_coef=args.sway, seed=42,
        )

        out_wav = audio_dir / f"w{w:.2f}.wav"
        sf.write(str(out_wav), wav, sr)
        dur = len(wav) / sr
        peak = float(np.abs(wav).max())
        print(f"     saved: {out_wav.name}  dur={dur:.2f}s  peak={peak:.3f}")

        rows.append({
            "method": "direct_mel_injection",
            "weight": w,
            "gen_text": gen_text,
            "output_wav": str(out_wav),
            "identity_ref": str(id_path),
            "emotion_ref": str(em_path),
            "duration_s": round(dur, 3),
            "peak_amp": round(peak, 4),
            "wer": "",
            "sim_o": "",
            "mcd": "",
        })

    # Zero-shot transfer: use emotion audio directly as reference
    print(f"\n  -- zero-shot (emotion audio as sole reference) --")
    mel_zs = mel_em  # weight=1.0 but using em's ref_text
    wav_zs, sr = infer_with_cond_mel(
        mel_zs, ref_text_em, gen_text,
        model, vocoder, device,
        nfe_step=args.nfe, cfg_strength=args.cfg,
        sway_coef=args.sway, seed=42,
    )
    out_zs = audio_dir / "zeroshot_emotion.wav"
    sf.write(str(out_zs), wav_zs, sr)
    dur_zs = len(wav_zs) / sr
    print(f"     saved: {out_zs.name}  dur={dur_zs:.2f}s")
    rows.append({
        "method": "zero_shot_emotion",
        "weight": 1.0,
        "gen_text": gen_text,
        "output_wav": str(out_zs),
        "identity_ref": str(em_path),
        "emotion_ref": str(em_path),
        "duration_s": round(dur_zs, 3),
        "peak_amp": round(float(np.abs(wav_zs).max()), 4),
        "wer": "",
        "sim_o": "",
        "mcd": "",
    })

    # --- Metrics ---
    print("\n[4/4] Computing metrics ...")
    identity_wav_for_sim = str(id_path)

    for row in rows:
        w_str = f"w={row['weight']:.2f} ({row['method']})"
        print(f"  {w_str}")
        wer_val = compute_wer(row["output_wav"], gen_text, asr)
        sim_val = compute_sim(row["output_wav"], identity_wav_for_sim, wavlm_fe, wavlm_model, device)
        mcd_val = compute_mcd(row["output_wav"], str(id_path))
        row["wer"] = round(wer_val, 2) if wer_val >= 0 else "err"
        row["sim_o"] = round(sim_val, 4) if sim_val >= -1 else "err"
        row["mcd"] = round(mcd_val, 2) if mcd_val >= 0 else "err"
        print(f"     WER={row['wer']}%  SIM-o={row['sim_o']}  MCD={row['mcd']}")

    # --- Save CSV ---
    csv_path = out_dir / "results_metrics.csv"
    fieldnames = ["method", "weight", "gen_text", "output_wav",
                  "identity_ref", "emotion_ref",
                  "duration_s", "peak_amp", "wer", "sim_o", "mcd"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] Results saved to {csv_path}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        v2_rows = [r for r in rows if r["method"] == "direct_mel_injection"
                   and isinstance(r["wer"], float)]
        if len(v2_rows) >= 2:
            ws = [r["weight"] for r in v2_rows]
            wers = [r["wer"] for r in v2_rows]
            sims = [r["sim_o"] for r in v2_rows]
            mcds = [r["mcd"] for r in v2_rows]

            fig, axes = plt.subplots(3, 1, figsize=(6, 12))
            fig.suptitle("Emotion Transfer v2 — Direct Mel Injection", fontsize=13)

            axes[0].plot(ws, wers, "o-r")
            axes[0].set_xlabel("Emotion weight")
            axes[0].set_ylabel("WER (%)")
            axes[0].set_title("Word Error Rate")

            axes[1].plot(ws, sims, "o-b")
            axes[1].set_xlabel("Emotion weight")
            axes[1].set_ylabel("SIM-o")
            axes[1].set_title("Speaker Similarity (vs identity ref)")

            axes[2].plot(ws, mcds, "o-g")
            axes[2].set_xlabel("Emotion weight")
            axes[2].set_ylabel("MCD")
            axes[2].set_title("Mel Cepstral Distortion")

            plt.tight_layout()
            plot_path = out_dir / "emotion_transfer.png"
            plt.savefig(str(plot_path), dpi=150)
            plt.close()
            print(f"[OK] Plot saved to {plot_path}")
    except Exception as e:
        print(f"[plot] {e}")

    print("\n" + "=" * 70)
    print("Summary:")
    for r in rows:
        print(f"  {r['method']:28s}  w={r['weight']:.2f}  "
              f"dur={r['duration_s']:.2f}s  peak={r['peak_amp']:.3f}  "
              f"WER={r['wer']}%  SIM={r['sim_o']}  MCD={r['mcd']}")
    print("=" * 70)
    return rows


# ---------------------------------------------------------------------------
# 8. Argument parsing and entry point
# ---------------------------------------------------------------------------

def parse_args():
    # Default references: F5-TTS canonical examples for maximum acoustic contrast.
    # Identity A: English female speaker (the paper's own reference clip).
    # Style   B: Mandarin animated-character voice (very different prosody/timbre).
    # To use different refs supply --identity / --emotion / --ref_text_id / --ref_text_em.
    f5_examples = PROJECT_ROOT / "F5-TTS" / "src" / "f5_tts" / "infer" / "examples" / "basic"
    default_id = str(f5_examples / "basic_ref_en.wav")
    default_em = str(f5_examples / "basic_ref_zh.wav")

    parser = argparse.ArgumentParser(
        description="Phase 4: Emotion/Style Transfer via Direct Mel Injection"
    )
    parser.add_argument("--identity", default=default_id,
                        help="Identity/Voice reference WAV (speaker A)")
    parser.add_argument("--emotion", default=default_em,
                        help="Emotion/Style reference WAV (speaker B)")
    parser.add_argument("--text",
                        default="I don't really care what you call me. "
                                "I've been a silent spectator, watching species "
                                "evolve, empires rise and fall.",
                        help="Text to generate (English)")
    parser.add_argument("--ref_text_id",
                        default="Some call me nature, others call me mother nature.",
                        help="Transcript of identity reference audio (for duration est.)")
    parser.add_argument("--ref_text_em",
                        default="\u5bf9,\u8fd9\u5c31\u662f\u6211\u4e07\u4eba\u656c\u4ef0\u7684\u592a\u4e59\u771f\u4eba\u3002",
                        help="Transcript of emotion reference audio (for duration est.)")
    parser.add_argument("--nfe", type=int, default=32, help="NFE steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG strength")
    parser.add_argument("--sway", type=float, default=-1.0, help="Sway coef")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"Identity ref: {Path(args.identity).name}")
    print(f"Emotion  ref: {Path(args.emotion).name}")
    run_experiment(args)
