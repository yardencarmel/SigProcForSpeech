#!/usr/bin/env python3
"""
Phase 1.2 — Demonstrate base F5-TTS on English (zero-shot TTS).

Generates several English samples using the pretrained F5TTS_v1_Base model.
These are used to:
  - Confirm the pipeline works end-to-end
  - Establish a quality baseline before any Hebrew work
  - Provide reference samples for Sway Sampling and CFG ablations

Usage:
    .venv/bin/python scripts/demo_english.py \\
        --ref_audio samples/ref_english.wav \\
        --outdir results/phase1/english

The script generates 5 fixed sentences from the LibriSpeech-PC style prompt
list used in the paper and saves them to outdir/.

If you don't have a reference WAV, any 3–10 second clean speech recording
(WAV/MP3/FLAC, 16 kHz+) works.  LibriSpeech samples are ideal.
"""

import sys
import argparse
from pathlib import Path

import soundfile as sf

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
    print(f"[ERROR] F5-TTS not available: {e}")
    F5_AVAILABLE = False

import torch

# Sentences that exercise prosody, punctuation, and longer utterances
DEMO_SENTENCES = [
    "The weather today is quite pleasant, with clear skies and a gentle breeze.",
    "She quickly realized that the answer had been right in front of her all along.",
    "Flow matching with diffusion transformers achieves state of the art results.",
    "Can you repeat that one more time? I didn't quite catch what you said.",
    "The conference will be held in San Francisco next month, and registration is now open.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Demo: English zero-shot TTS with F5-TTS")
    p.add_argument("--ref_audio", "-r", required=True,
                   help="3–10 s reference WAV for voice cloning")
    p.add_argument("--outdir", "-o", default="results/phase1/english",
                   help="Output directory")
    p.add_argument("--nfe_steps", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_coef", type=float, default=-1.0,
                   help="Sway Sampling coefficient (paper default -1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    if not F5_AVAILABLE:
        print("[ERROR] Install F5-TTS first:  pip install -e F5-TTS/")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading model ...")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model("F5TTS_v1_Base", DiT, model_cfg).to(args.device).eval()
    vocoder = load_vocoder(is_local=False, local_path=None)

    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        args.ref_audio, "", device=args.device
    )

    print(f"\n[INFO] Generating {len(DEMO_SENTENCES)} English samples ...")
    for i, sentence in enumerate(DEMO_SENTENCES, 1):
        torch.manual_seed(args.seed + i)
        print(f"  [{i}/{len(DEMO_SENTENCES)}] {sentence[:60]}...")

        audio, sr, _ = infer_process(
            ref_audio_proc, ref_text_proc, sentence,
            model, vocoder,
            mel_spec_type="vocos",
            nfe_step=args.nfe_steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_coef,
            device=args.device,
        )

        out_path = outdir / f"en_demo_{i:02d}.wav"
        sf.write(str(out_path), audio, sr)
        print(f"         → {out_path}  ({len(audio)/sr:.2f}s)")

    print(f"\n[OK] All samples saved to {outdir}/")
    print("     Listen to them to verify the pipeline is working before continuing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
