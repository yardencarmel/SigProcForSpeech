#!/usr/bin/env python3
"""
End-to-end pipeline for the F5-TTS Acoustic Style Transfer project.

Runs all phases using the current F5-TTS API (v1, OmegaConf-based loading):
  Phase 1  — English baseline
  Phase 2  — Sway Sampling ablation + CFG ablation (reproduce paper results)
  Phase 4  — Style transfer (mel-space blending)
  Phase 5  — Evaluate (WER, SIM-o, MCD) + plots

Usage:
    .venv/Scripts/python scripts/run_all_phases.py \
        --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
        --outdir results
"""

import argparse
import csv
import os
import sys
import tempfile
import traceback
import warnings
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore")

# Add FFmpeg to PATH so Whisper transcription works
_FFMPEG_BIN = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    r"Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
)
if os.path.isdir(_FFMPEG_BIN) and _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Monkey-patch torchaudio.load to use soundfile for WAV/FLAC (torchaudio 2.10
# on Windows requires FFmpeg shared DLLs for torchcodec which we don't have).
# soundfile handles WAV/FLAC natively; this patch is applied before F5-TTS imports.
# ---------------------------------------------------------------------------
import torchaudio as _torchaudio
import soundfile as _sf
import torch as _torch
import numpy as _np

def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, backend=None, buffer_size=4096):
    """soundfile-based torchaudio.load replacement for WAV/FLAC/OGG files."""
    try:
        data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        # data: (frames, channels)
        tensor = _torch.from_numpy(data.T.copy())  # (channels, frames)
        if frame_offset:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        return tensor, sr
    except Exception as e:
        raise RuntimeError(f"soundfile could not load {path}: {e}") from e

_torchaudio.load = _sf_load

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "F5-TTS" / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

# ---------------------------------------------------------------------------
# F5-TTS model loading (new API)
# ---------------------------------------------------------------------------

def load_f5tts(device="cuda", vocab_file=""):
    from importlib.resources import files as pkg_files
    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    yaml_path = str(pkg_files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
    model_cfg = OmegaConf.load(yaml_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    print("[INFO] Downloading / loading F5TTS_v1_Base checkpoint ...")
    ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    model = load_model(model_cls, model_arc, ckpt_path,
                       mel_spec_type="vocos", vocab_file=vocab_file, device=device)

    print("[INFO] Loading vocos vocoder ...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device)

    return model, vocoder


def infer(ref_wav, ref_txt, gen_txt, model, vocoder, device,
          nfe_step=32, cfg_strength=2.0, sway_coef=-1.0, speed=1.0, seed=42):
    """Single inference call returning (audio_np, sr)."""
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    if seed is not None:
        torch.manual_seed(seed)
    ref_audio, ref_text = preprocess_ref_audio_text(ref_wav, ref_txt)
    audio, sr, _ = infer_process(
        ref_audio, ref_text, gen_txt, model, vocoder,
        mel_spec_type="vocos",
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_coef,
        speed=speed,
        device=device,
    )
    return audio, sr


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="F5-TTS Acoustic Style Transfer — full pipeline")
    p.add_argument("--ref_audio", "-r", required=True,
                   help="Reference WAV file for voice cloning")
    p.add_argument("--ref_text", default="",
                   help="Transcription of ref_audio (leave blank to auto-transcribe)")
    p.add_argument("--outdir", "-o", default="results",
                   help="Root output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--phases", nargs="+", type=int, default=[1, 2, 4, 5, 6],
                   help="Which phases to run (1: English, 2: Ablations, 4: Emotion, 5: Eval, 6: Plots)")
    p.add_argument("--plots", nargs="+", default=["all"],
                   choices=["all", "sway_sampling", "cfg_strength", "emotion_weight", "sway_pdf"],
                   help="Which single figures to generate when running phase 6 (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 — English baseline
# ---------------------------------------------------------------------------

ENGLISH_SENTENCES = [
    ("en01", "The weather today is quite pleasant, with clear skies and a gentle breeze."),
    ("en02", "She quickly realized the answer had been right in front of her all along."),
    ("en03", "Flow matching with diffusion transformers achieves state of the art results."),
    ("en04", "Can you repeat that one more time? I didn't quite catch what you said."),
    ("en05", "The conference will be held in San Francisco next month, and registration is now open."),
]

def run_phase1(model, vocoder, ref_audio, ref_text, outdir, device, seed):
    print("\n" + "="*60)
    print("PHASE 1: English baseline")
    print("="*60)

    en_dir = Path(outdir) / "phase1" / "english"
    en_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[1.1] English zero-shot TTS -> {en_dir}")
    for sent_id, text in ENGLISH_SENTENCES:
        out_path = en_dir / f"{sent_id}.wav"
        print(f"  Generating: {text[:55]}...")
        audio, sr = infer(ref_audio, ref_text, text, model, vocoder, device,
                          nfe_step=32, seed=seed)
        sf.write(str(out_path), audio, sr)
        print(f"  -> {out_path}  ({len(audio)/sr:.2f}s)")

    print("\n[Phase 1 complete]")


# ---------------------------------------------------------------------------
# Phase 2 — Sway Sampling + CFG ablations
# ---------------------------------------------------------------------------

SWAY_COEFS  = [0.4, 0.0, -0.4, -0.8, -1.0]
NFE_STEPS   = [8, 16, 32]
ABLATION_SENTENCES = [
    ("ab01", "The cat sat on the mat and looked out the window."),
    ("ab02", "Scientists have discovered a new species of deep-sea fish."),
    ("ab03", "Flow matching with diffusion transformers achieves excellent results."),
]
CFG_STRENGTHS = [1.0, 1.5, 2.0, 2.5, 3.0]


def run_phase2(model, vocoder, ref_audio, ref_text, outdir, device, seed, skip_existing):
    print("\n" + "="*60)
    print("PHASE 2: Sway Sampling + CFG ablations (reproduce paper)")
    print("="*60)

    # --- 2.1 / 2.3: Sway Sampling sweep ---
    sway_dir = Path(outdir) / "sway_sampling"
    audio_dir = sway_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    total = len(SWAY_COEFS) * len(NFE_STEPS) * len(ABLATION_SENTENCES)
    print(f"\n[2.1] Sway Sampling sweep: {total} inferences")

    sway_rows = []
    done = 0
    for sway, nfe, (sid, text) in product(SWAY_COEFS, NFE_STEPS, ABLATION_SENTENCES):
        done += 1
        out_name = f"{sid}_s{sway:+.1f}_nfe{nfe:02d}.wav"
        out_path = audio_dir / out_name
        print(f"  [{done:03d}/{total}] s={sway:+.1f} NFE={nfe} {sid}", end="  ")
        if skip_existing and out_path.exists():
            print("[SKIP]")
        else:
            audio, sr = infer(ref_audio, ref_text, text, model, vocoder, device,
                              nfe_step=nfe, sway_coef=sway, seed=seed)
            sf.write(str(out_path), audio, sr)
            print(f"({len(audio)/sr:.2f}s)")
        sway_rows.append({
            "sway_coef": sway, "nfe_steps": nfe, "sentence_id": sid,
            "transcript": text, "output_wav": str(out_path), "wer": "", "sim_o": "",
        })

    # Write manifest
    sway_csv = sway_dir / "results.csv"
    with open(sway_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sway_rows[0].keys()))
        w.writeheader(); w.writerows(sway_rows)
    print(f"\n  Manifest -> {sway_csv}")

    # --- 2.2: CFG sweep ---
    cfg_dir = Path(outdir) / "cfg_strength"
    cfg_audio = cfg_dir / "audio"
    cfg_audio.mkdir(parents=True, exist_ok=True)

    cfg_total = len(CFG_STRENGTHS) * len(ABLATION_SENTENCES)
    print(f"\n[2.2] CFG strength sweep: {cfg_total} inferences")
    cfg_rows = []
    done = 0
    for cfg, (sid, text) in product(CFG_STRENGTHS, ABLATION_SENTENCES):
        done += 1
        out_name = f"{sid}_cfg{cfg:.1f}.wav"
        out_path = cfg_audio / out_name
        print(f"  [{done:02d}/{cfg_total}] cfg={cfg:.1f} {sid}", end="  ")
        if skip_existing and out_path.exists():
            print("[SKIP]")
        else:
            audio, sr = infer(ref_audio, ref_text, text, model, vocoder, device,
                              nfe_step=32, cfg_strength=cfg, sway_coef=-1.0, seed=seed)
            sf.write(str(out_path), audio, sr)
            print(f"({len(audio)/sr:.2f}s)")
        cfg_rows.append({
            "cfg_strength": cfg, "sentence_id": sid,
            "transcript": text, "output_wav": str(out_path), "wer": "", "sim_o": "",
        })

    cfg_csv = cfg_dir / "results.csv"
    with open(cfg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cfg_rows[0].keys()))
        w.writeheader(); w.writerows(cfg_rows)
    print(f"  Manifest -> {cfg_csv}")

    print("\n[Phase 2 complete]")
    return sway_csv, cfg_csv


# ---------------------------------------------------------------------------
# Phase 4 — Emotion / style transfer (mel-space blending)
# ---------------------------------------------------------------------------

def blend_mels(mel_id, mel_em, weight):
    T = min(mel_id.shape[-1], mel_em.shape[-1])
    return (1.0 - weight) * mel_id[..., :T] + weight * mel_em[..., :T]


def audio_to_mel(wav_path, sr=24000, n_fft=1024, hop=256, win=1024, n_mels=100):
    audio, orig_sr = torchaudio.load(wav_path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if orig_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, f_min=0.0, f_max=8000.0,
        power=1.0, norm="slaney", mel_scale="slaney",
    )
    mel = mel_t(audio.squeeze(0).unsqueeze(0)).squeeze(0)
    return torch.log(torch.clamp(mel, min=1e-5))


def make_emotion_ref_wav(id_wav, em_wav, weight, vocoder, device):
    mel_id = audio_to_mel(id_wav)
    mel_em = audio_to_mel(em_wav)
    mel_blend = blend_mels(mel_id, mel_em, weight)
    with torch.no_grad():
        audio_np = vocoder.decode(mel_blend.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_np, 24000)
    return tmp.name


def run_phase4(model, vocoder, ref_audio_id, ref_text_id,
               ref_audio_em, ref_text_em, outdir, device, seed):
    print("\n" + "="*60)
    print("PHASE 4: Emotion / style transfer — mel-space blending")
    print("="*60)

    # Sweep emotion weights using the two reference recordings
    em_dir = Path(outdir) / "emotion_transfer"
    em_dir.mkdir(parents=True, exist_ok=True)

    gen_text = "The situation has become completely unacceptable and we need to act now."
    weights = [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]

    print(f"  Identity ref  : {Path(ref_audio_id).name}")
    print(f"  Emotion ref   : {Path(ref_audio_em).name}")
    print(f"  Text          : {gen_text[:60]}")
    print(f"  Weights sweep : {weights}\n")

    rows = []
    for w in weights:
        out_path = em_dir / f"emo_w{w:.2f}.wav"
        print(f"  weight={w:.2f}", end="  ")
        if w == 0.0:
            eff_ref, eff_txt = ref_audio_id, ref_text_id
        elif w == 1.0:
            eff_ref, eff_txt = ref_audio_em, ref_text_em
        else:
            eff_ref = make_emotion_ref_wav(ref_audio_id, ref_audio_em, w, vocoder, device)
            eff_txt = ref_text_id  # blended mel; assign identity transcript

        audio, sr = infer(eff_ref, eff_txt, gen_text, model, vocoder, device,
                          nfe_step=32, seed=seed)
        sf.write(str(out_path), audio, sr)
        print(f"({len(audio)/sr:.2f}s)")

        if w not in (0.0, 1.0) and eff_ref != ref_audio_id and eff_ref != ref_audio_em:
            os.unlink(eff_ref)

        rows.append({
            "emotion_weight": w,
            "gen_text": gen_text,
            "output_wav": str(out_path),
            "identity_ref": ref_audio_id,
            "emotion_ref": ref_audio_em,
            "wer": "", "sim_o": "", "mcd": "",
        })

    em_csv = em_dir / "results.csv"
    with open(em_csv, "w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w2.writeheader(); w2.writerows(rows)
    print(f"\n  Manifest -> {em_csv}")
    print("\n[Phase 4 complete]")
    return em_csv


# ---------------------------------------------------------------------------
# Phase 5 — Evaluate (WER with Whisper, SIM-o with WavLM)
# ---------------------------------------------------------------------------

def compute_wer_whisper(wav_path, reference_text, language="en", device="cuda"):
    """Compute WER using Whisper-large-v3-turbo."""
    try:
        from transformers import pipeline as hf_pipeline
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=0 if device == "cuda" else -1,
        )
        result = asr(wav_path, generate_kwargs={"language": language})
        hypothesis = result["text"].strip().lower()
        reference = reference_text.strip().lower()
        # Simple WER: edit distance / len(ref)
        from difflib import SequenceMatcher
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if not ref_words:
            return 0.0
        sm = SequenceMatcher(None, ref_words, hyp_words)
        match = sum(block.size for block in sm.get_matching_blocks())
        wer = 1.0 - match / len(ref_words)
        return round(wer * 100, 2)
    except Exception as e:
        print(f"    WER error: {e}")
        return None


def compute_sim_wavlm(wav1, wav2, device="cuda"):
    """Compute speaker similarity using WavLM-large embeddings."""
    try:
        from transformers import AutoFeatureExtractor, WavLMModel
        import torch.nn.functional as F

        model_id = "microsoft/wavlm-base-plus"  # smaller than large, still good
        fe = AutoFeatureExtractor.from_pretrained(model_id)
        wavlm = WavLMModel.from_pretrained(model_id).to(device).eval()

        def embed(path):
            audio, sr = torchaudio.load(path)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            inputs = fe(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = wavlm(**inputs).last_hidden_state.mean(1)
            return F.normalize(out, dim=-1)

        e1 = embed(wav1)
        e2 = embed(wav2)
        return round(float(torch.cosine_similarity(e1, e2).item()), 4)
    except Exception as e:
        print(f"    SIM-o error: {e}")
        return None


def compute_mcd(wav1, wav2, sr=24000, n_mfcc=24):
    """Mel Cepstral Distortion between two WAV files."""
    try:
        def get_mfcc(path):
            audio, orig_sr = torchaudio.load(path)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if orig_sr != sr:
                audio = torchaudio.functional.resample(audio, orig_sr, sr)
            mfcc_t = torchaudio.transforms.MFCC(
                sample_rate=sr, n_mfcc=n_mfcc,
                melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
            )
            return mfcc_t(audio.squeeze(0))  # (n_mfcc, T)

        m1 = get_mfcc(wav1)
        m2 = get_mfcc(wav2)
        T = min(m1.shape[1], m2.shape[1])
        m1, m2 = m1[1:, :T], m2[1:, :T]  # skip C0
        diff = (m1 - m2) ** 2
        mcd = float(torch.sqrt(2 * diff.sum(0)).mean()) * (10 / np.log(10))
        return round(mcd, 4)
    except Exception as e:
        print(f"    MCD error: {e}")
        return None


def run_phase5(outdir, ref_audio, device):
    print("\n" + "="*60)
    print("PHASE 5: Evaluation — WER, SIM-o, MCD")
    print("="*60)

    results_dir = Path(outdir)
    summary_rows = []

    # --- Evaluate sway sampling ---
    sway_csv = results_dir / "sway_sampling" / "results.csv"
    if sway_csv.exists():
        print("\n[5.1] Evaluating sway sampling ablation ...")
        with open(sway_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        updated = []
        for row in rows:
            wav = row["output_wav"]
            transcript = row["transcript"]
            if os.path.exists(wav):
                wer = compute_wer_whisper(wav, transcript, "en", device)
                sim = compute_sim_wavlm(wav, ref_audio, device)
                row["wer"] = wer if wer is not None else ""
                row["sim_o"] = sim if sim is not None else ""
                print(f"  s={row['sway_coef']:5s} NFE={row['nfe_steps']:2s} "
                      f"WER={row['wer']}% SIM-o={row['sim_o']}")
                summary_rows.append({
                    "phase": "sway", "config": f"s={row['sway_coef']}_nfe{row['nfe_steps']}",
                    "wer": row["wer"], "sim_o": row["sim_o"], "mcd": "",
                })
            updated.append(row)
        sway_metrics_csv = results_dir / "sway_sampling" / "results_metrics.csv"
        with open(sway_metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(updated[0].keys()))
            w.writeheader(); w.writerows(updated)
        print(f"  -> {sway_metrics_csv}")

    # --- Evaluate CFG ---
    cfg_csv = results_dir / "cfg_strength" / "results.csv"
    if cfg_csv.exists():
        print("\n[5.2] Evaluating CFG ablation ...")
        with open(cfg_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        updated = []
        for row in rows:
            wav = row["output_wav"]
            if os.path.exists(wav):
                wer = compute_wer_whisper(wav, row["transcript"], "en", device)
                sim = compute_sim_wavlm(wav, ref_audio, device)
                row["wer"] = wer if wer is not None else ""
                row["sim_o"] = sim if sim is not None else ""
                print(f"  cfg={row['cfg_strength']:4s} WER={row['wer']}% SIM-o={row['sim_o']}")
                summary_rows.append({
                    "phase": "cfg", "config": f"cfg={row['cfg_strength']}",
                    "wer": row["wer"], "sim_o": row["sim_o"], "mcd": "",
                })
            updated.append(row)
        cfg_metrics_csv = results_dir / "cfg_strength" / "results_metrics.csv"
        with open(cfg_metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(updated[0].keys()))
            w.writeheader(); w.writerows(updated)

    # --- Evaluate emotion transfer ---
    em_csv = results_dir / "emotion_transfer" / "results.csv"
    if em_csv.exists():
        print("\n[5.3] Evaluating emotion transfer ...")
        with open(em_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        updated = []
        for row in rows:
            wav = row["output_wav"]
            if os.path.exists(wav):
                wer = compute_wer_whisper(wav, row["gen_text"], "en", device)
                sim = compute_sim_wavlm(wav, ref_audio, device)
                mcd = compute_mcd(wav, row["emotion_ref"])
                row["wer"] = wer if wer is not None else ""
                row["sim_o"] = sim if sim is not None else ""
                row["mcd"] = mcd if mcd is not None else ""
                print(f"  w={float(row['emotion_weight']):.2f} WER={row['wer']}% "
                      f"SIM-o={row['sim_o']} MCD={row['mcd']}")
                summary_rows.append({
                    "phase": "emotion", "config": f"w={row['emotion_weight']}",
                    "wer": row["wer"], "sim_o": row["sim_o"], "mcd": row["mcd"],
                })
            updated.append(row)
        em_metrics_csv = results_dir / "emotion_transfer" / "results_metrics.csv"
        with open(em_metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(updated[0].keys()))
            w.writeheader(); w.writerows(updated)

    # --- Summary table ---
    if summary_rows:
        summary_csv = results_dir / "summary_table.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "config", "wer", "sim_o", "mcd"])
            w.writeheader(); w.writerows(summary_rows)
        print(f"\n  Summary -> {summary_csv}")

    print("\n[Phase 5 complete]")


# ---------------------------------------------------------------------------
# Phase 5b — Plots
# ---------------------------------------------------------------------------

def plot_sway_sampling(results_dir, plots_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        sway_csv = results_dir / "sway_sampling" / "results_metrics.csv"
        if sway_csv.exists():
            with open(sway_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # Aggregate by (sway_coef, nfe_steps) -> mean WER & SIM-o
            from collections import defaultdict
            agg = defaultdict(list)
            for row in rows:
                try:
                    key = (float(row["sway_coef"]), int(row["nfe_steps"]))
                    wer = float(row["wer"]) if row["wer"] else None
                    sim = float(row["sim_o"]) if row["sim_o"] else None
                    if wer is not None and sim is not None:
                        agg[key].append((wer, sim))
                except Exception:
                    pass
            if agg:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
                # WER vs sway_coef for each NFE
                for nfe in [8, 16, 32]:
                    keys = sorted(k for k in agg if k[1] == nfe)
                    xs = [k[0] for k in keys]
                    wers = [np.mean([v[0] for v in agg[k]]) for k in keys]
                    sims = [np.mean([v[1] for v in agg[k]]) for k in keys]
                    ax1.plot(xs, wers, marker="o", label=f"NFE={nfe}")
                    ax2.plot(xs, sims, marker="s", label=f"NFE={nfe}")
                ax1.set_xlabel("Sway Sampling coefficient s")
                ax1.set_ylabel("WER (%)")
                ax1.set_title("WER vs Sway Sampling")
                ax1.legend(); ax1.grid(True)
                ax2.set_xlabel("Sway Sampling coefficient s")
                ax2.set_ylabel("SIM-o")
                ax2.set_title("SIM-o vs Sway Sampling")
                ax2.legend(); ax2.grid(True)
                plt.tight_layout()
                fig.savefig(str(plots_dir / "sway_sampling.png"), dpi=120)
                plt.close()
                print(f"  -> {plots_dir}/sway_sampling.png")
    except Exception as e:
        print(f"  Error plot_sway_sampling: {e}")
        traceback.print_exc()

def plot_cfg_strength(results_dir, plots_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        cfg_csv = results_dir / "cfg_strength" / "results_metrics.csv"
        if cfg_csv.exists():
            with open(cfg_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            from collections import defaultdict
            agg = defaultdict(list)
            for row in rows:
                try:
                    cfg = float(row["cfg_strength"])
                    wer = float(row["wer"]) if row["wer"] else None
                    sim = float(row["sim_o"]) if row["sim_o"] else None
                    if wer is not None and sim is not None:
                        agg[cfg].append((wer, sim))
                except Exception:
                    pass
            if agg:
                cfgs = sorted(agg.keys())
                wers = [np.mean([v[0] for v in agg[c]]) for c in cfgs]
                sims = [np.mean([v[1] for v in agg[c]]) for c in cfgs]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
                ax1.plot(cfgs, wers, "o-", color="royalblue")
                ax1.set_xlabel("CFG strength"); ax1.set_ylabel("WER (%)")
                ax1.set_title("WER vs CFG strength"); ax1.grid(True)
                ax2.plot(cfgs, sims, "s-", color="darkorange")
                ax2.set_xlabel("CFG strength"); ax2.set_ylabel("SIM-o")
                ax2.set_title("SIM-o vs CFG strength"); ax2.grid(True)
                plt.tight_layout()
                fig.savefig(str(plots_dir / "cfg_strength.png"), dpi=120)
                plt.close()
                print(f"  -> {plots_dir}/cfg_strength.png")
    except Exception as e:
        print(f"  Error plot_cfg_strength: {e}")
        traceback.print_exc()

def plot_emotion_weight(results_dir, plots_dir):
    try:
        import matplotlib.pyplot as plt
        em_csv = results_dir / "emotion_transfer" / "results_metrics.csv"
        if em_csv.exists():
            with open(em_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ws, sims, mcds = [], [], []
            for row in rows:
                try:
                    w = float(row["emotion_weight"])
                    sim = float(row["sim_o"]) if row["sim_o"] else None
                    mcd = float(row["mcd"]) if row["mcd"] else None
                    if sim is not None:
                        ws.append(w); sims.append(sim)
                    if mcd is not None:
                        mcds.append(mcd)
                except Exception:
                    pass
            if ws:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
                ax1.plot(ws[:len(sims)], sims, "o-", color="steelblue")
                ax1.set_xlabel("Emotion weight"); ax1.set_ylabel("SIM-o")
                ax1.set_title("Speaker similarity vs emotion weight"); ax1.grid(True)
                if mcds:
                    ax2.plot(ws[:len(mcds)], mcds, "s-", color="firebrick")
                    ax2.set_xlabel("Emotion weight"); ax2.set_ylabel("MCD (dB)")
                    ax2.set_title("MCD vs emotion weight"); ax2.grid(True)
                plt.tight_layout()
                fig.savefig(str(plots_dir / "emotion_weight.png"), dpi=120)
                plt.close()
                print(f"  -> {plots_dir}/emotion_weight.png")
    except Exception as e:
        print(f"  Error plot_emotion_weight: {e}")
        traceback.print_exc()

def plot_sway_pdf(plots_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        t = np.linspace(0, 1, 500)
        fig, ax = plt.subplots(figsize=(7, 4))
        for s, alpha in [(1.0, 0.45), (0.8, 0.25)]:
            f = t + s * (np.cos(np.pi / 2 * t) - 1 + t)
            dt = np.diff(f)
            dt = np.where(dt > 0, dt, 1e-6)
            pdf = 1.0 / np.maximum(dt * len(t), 1e-6)
            ax.plot(f[1:], pdf, ls="--", alpha=alpha, color="grey",
                    label=f"s={s:+.1f} (not tested)")
        for s in [0.4, 0.0, -0.4, -0.8, -1.0]:
            f = t + s * (np.cos(np.pi / 2 * t) - 1 + t)
            dt = np.diff(f)
            dt = np.where(dt > 0, dt, 1e-6)
            pdf = 1.0 / np.maximum(dt * len(t), 1e-6)
            ax.plot(f[1:], pdf, label=f"s={s:+.1f}")
        ax.set_xlabel("t"); ax.set_ylabel("π(t) density")
        ax.set_ylim(10**-0.5, 10**1.5)
        ax.set_title("Sway Sampling: ODE timestep distribution π(t)")
        ax.set_yscale("log")
        ax.legend(); ax.grid(True, alpha=0.4)
        plt.tight_layout()
        fig.savefig(str(plots_dir / "sway_pdf.png"), dpi=120)
        plt.close()
        print(f"  -> {plots_dir}/sway_pdf.png")
    except Exception as e:
        print(f"  Error plot_sway_pdf: {e}")
        traceback.print_exc()

def run_plots(outdir, plots_to_run=["all"]):
    print("\n" + "="*60)
    print(f"PHASE 6: Generating plots (plots={plots_to_run})")
    print("="*60)
    try:
        import matplotlib
        matplotlib.use("Agg")
        
        results_dir = Path(outdir)
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        if "all" in plots_to_run:
            plots_to_run = ["sway_sampling", "cfg_strength", "emotion_weight", "sway_pdf"]
            
        if "sway_sampling" in plots_to_run:
            plot_sway_sampling(results_dir, plots_dir)
        if "cfg_strength" in plots_to_run:
            plot_cfg_strength(results_dir, plots_dir)
        if "emotion_weight" in plots_to_run:
            plot_emotion_weight(results_dir, plots_dir)
        if "sway_pdf" in plots_to_run:
            plot_sway_pdf(plots_dir)

    except Exception as e:
        print(f"  Plot error: {e}")
        traceback.print_exc()

    print("\n[Plots complete]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ref_audio = str(Path(args.ref_audio).resolve())
    # Use the Mandarin clip as style reference (maximum acoustic contrast)
    ref_audio_emotion = str(
        Path(PROJECT_DIR) / "F5-TTS" / "src" / "f5_tts" / "infer"
        / "examples" / "basic" / "basic_ref_zh.wav"
    )
    if not os.path.exists(ref_audio_emotion):
        ref_audio_emotion = ref_audio  # fallback

    ref_text = args.ref_text  # "" = auto-transcribe

    print(f"\nDevice       : {args.device}")
    print(f"Ref audio    : {ref_audio}")
    print(f"Emotion ref  : {ref_audio_emotion}")
    print(f"Output dir   : {args.outdir}")
    print(f"Phases       : {args.phases}\n")

    # Load model only if needed (prevents hanging when just creating graphs)
    if any(p in args.phases for p in [1, 2, 4]):
        model, vocoder = load_f5tts(device=args.device)
    else:
        model, vocoder = None, None

    if 1 in args.phases:
        run_phase1(model, vocoder, ref_audio, ref_text, args.outdir, args.device, args.seed)

    if 2 in args.phases:
        sway_csv, cfg_csv = run_phase2(
            model, vocoder, ref_audio, ref_text, args.outdir, args.device,
            args.seed, args.skip_existing)

    if 4 in args.phases:
        em_csv = run_phase4(
            model, vocoder, ref_audio, ref_text,
            ref_audio_emotion, ref_text,  # use same text for emotion ref too
            args.outdir, args.device, args.seed)

    if 5 in args.phases:
        run_phase5(args.outdir, ref_audio, args.device)

    if 6 in args.phases or 5 in args.phases:
        run_plots(args.outdir, args.plots)

    print("\n" + "="*60)
    print("ALL PHASES COMPLETE")
    print(f"Results in: {args.outdir}/")
    print("="*60)


if __name__ == "__main__":
    main()
