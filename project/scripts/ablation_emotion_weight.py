#!/usr/bin/env python3
"""
Phase 4.4 — Emotion Transfer Weight Ablation

Sweeps emotion_weight w ∈ {0.0, 0.2, 0.3, 0.5, 0.7, 1.0} for a fixed pair
of (identity, emotion) references.  Measures how the blend affects:
  - SIM-o: speaker similarity vs. identity reference (should decrease with w)
  - MCD:   mel cepstral distortion vs. emotion reference (should decrease with w)

w=0 → pure identity (baseline voice cloning)
w=1 → pure emotion  (baseline: ignore identity)

The cross-over point where the emotion is perceptible but identity is
preserved is the design goal — typically 0.3–0.5.

Usage:
    .venv/bin/python scripts/ablation_emotion_weight.py \\
        --ref_identity samples/neutral_speaker.wav \\
        --ref_emotion  samples/angry_english.wav \\
        --text "The situation has become completely unacceptable." \\
        --outdir results/emotion_transfer

Cross-lingual example (Hebrew text + English emotion):
    .venv/bin/python scripts/ablation_emotion_weight.py \\
        --ref_identity samples/hebrew_speaker.wav \\
        --ref_emotion  samples/english_scream.wav \\
        --text "המצב הפך לבלתי נסבל לחלוטין." \\
        --outdir results/emotion_crosslingual \\
        --language he
"""

import sys
import csv
import argparse
import tempfile
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "F5-TTS" / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

try:
    from f5_tts.infer.utils_infer import (
        load_model, load_vocoder, preprocess_ref_audio_text, infer_process,
    )
    from f5_tts.model import DiT
    F5_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] {e}")
    F5_AVAILABLE = False

from run_emotion_transfer import (
    load_audio, audio_to_mel, mel_to_audio, blend_mels,
    F5TTSWithStyleTransfer,
)
from hebrew_utils import normalize_hebrew_text

WEIGHT_VALUES = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]


def parse_args():
    p = argparse.ArgumentParser(description="Emotion weight ablation")
    p.add_argument("--ref_identity", required=True,
                   help="Identity reference WAV (the speaker's voice to preserve)")
    p.add_argument("--ref_emotion", required=True,
                   help="Emotion reference WAV (the style/affect to inject)")
    p.add_argument("--text", required=True,
                   help="Text to synthesize")
    p.add_argument("--outdir", "-o", default="results/emotion_transfer")
    p.add_argument("--model_path", default=None,
                   help="Fine-tuned checkpoint (optional)")
    p.add_argument("--vocab_path", default=None)
    p.add_argument("--nfe_steps", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_coef", type=float, default=-1.0)
    p.add_argument("--language", default=None,
                   help="Language hint for WER (e.g. 'he', 'en')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    if not F5_AVAILABLE:
        print("[ERROR] Install F5-TTS first.")
        return 1

    outdir = Path(args.outdir)
    audio_dir = outdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading model ...")
    tts = F5TTSWithStyleTransfer(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        device=args.device,
    )

    rows = []
    print(f"\n[INFO] Sweeping emotion_weight over {WEIGHT_VALUES}")
    print(f"  Identity ref : {args.ref_identity}")
    print(f"  Emotion ref  : {args.ref_emotion}")
    print(f"  Text         : {args.text}\n")

    for w in WEIGHT_VALUES:
        torch.manual_seed(args.seed)
        out_path = audio_dir / f"weight_{w:.1f}.wav"
        print(f"  w={w:.1f}", end="  ")

        audio, sr = tts.generate(
            text=args.text,
            ref_audio_path=args.ref_identity,
            ref_audio_emotion_path=args.ref_emotion if w > 0 else None,
            emotion_weight=w,
            nfe_steps=args.nfe_steps,
            cfg_strength=args.cfg_strength,
            sway_coef=args.sway_coef,
            seed=args.seed,
        )
        sf.write(str(out_path), audio, sr)
        print(f"→ {out_path.name}  ({len(audio)/sr:.2f}s)")

        rows.append({
            "emotion_weight": w,
            "transcript": args.text,
            "output_wav": str(out_path),
            "reference_wav": args.ref_identity,
            "emotion_wav": args.ref_emotion,
            "language": args.language or "",
            "sim_o": "",  # filled by evaluate.py
            "mcd": "",    # filled by evaluate.py
            "wer": "",    # filled by evaluate.py
        })

    csv_path = outdir / "results.csv"
    fieldnames = ["emotion_weight", "transcript", "output_wav", "reference_wav",
                  "emotion_wav", "language", "sim_o", "mcd", "wer"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Audio in {audio_dir}/")
    print(f"     Manifest:  {csv_path}")
    print(
        f"\nNext: evaluate\n"
        f"  .venv/bin/python scripts/evaluate.py --csv {csv_path} --metric all"
        + (f" --language {args.language}" if args.language else "")
    )
    print(
        "\nThen plot:\n"
        f"  .venv/bin/python scripts/plot_results.py "
        f"--csv {csv_path} --ablation emotion_weight"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
