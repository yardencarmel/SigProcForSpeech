#!/usr/bin/env python3
"""
Phase 2.2 — Classifier-Free Guidance (CFG) Strength Ablation

Sweeps cfg_strength over a fixed set of prompts with the paper-recommended
sway coefficient s=-1 and 32 NFE steps.

CFG trades off faithfulness to the text prompt vs. naturalness:
  - Low CFG (1.0): more diverse but potentially misaligned
  - High CFG (3.0): more faithful to text, potentially over-sharpened

Usage:
    .venv/bin/python scripts/ablation_cfg.py \\
        --ref_audio samples/ref_english.wav \\
        --outdir results/cfg_strength
"""

import sys
import csv
import argparse
from pathlib import Path

import torch
import soundfile as sf

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "F5-TTS" / "src"))

try:
    from f5_tts.infer.utils_infer import (
        load_model, load_vocoder, preprocess_ref_audio_text, infer_process,
    )
    from f5_tts.model import DiT
    F5_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] {e}")
    F5_AVAILABLE = False

CFG_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
NFE_STEPS  = 32
SWAY_COEF  = -1.0

TEST_SENTENCES = [
    ("en01", "The quick brown fox jumps over the lazy dog near the river bank."),
    ("en02", "She whispered softly so as not to wake the sleeping child."),
    ("en03", "The experiment was a complete success, confirming all our predictions."),
]


def parse_args():
    p = argparse.ArgumentParser(description="CFG strength ablation")
    p.add_argument("--ref_audio", "-r", required=True)
    p.add_argument("--outdir", "-o", default="results/cfg_strength")
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
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model("F5TTS_v1_Base", DiT, model_cfg).to(args.device).eval()
    vocoder = load_vocoder(is_local=False, local_path=None)

    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        args.ref_audio, "", device=args.device
    )

    rows = []
    total = len(CFG_VALUES) * len(TEST_SENTENCES)
    done = 0

    for cfg in CFG_VALUES:
        for sent_id, text in TEST_SENTENCES:
            done += 1
            out_path = audio_dir / f"{sent_id}_cfg{cfg:.1f}.wav"
            print(f"  [{done:02d}/{total}] cfg={cfg:.1f}  {sent_id}", end="  ")

            torch.manual_seed(args.seed)
            audio, sr, _ = infer_process(
                ref_audio_proc, ref_text_proc, text,
                model, vocoder,
                mel_spec_type="vocos",
                nfe_step=NFE_STEPS,
                cfg_strength=cfg,
                sway_sampling_coef=SWAY_COEF,
                device=args.device,
            )
            sf.write(str(out_path), audio, sr)
            print(f"({len(audio)/sr:.2f}s)")

            rows.append({
                "cfg_strength": cfg,
                "sentence_id": sent_id,
                "transcript": text,
                "output_wav": str(out_path),
                "wer": "",
                "sim_o": "",
            })

    csv_path = outdir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cfg_strength", "sentence_id", "transcript",
                           "output_wav", "wer", "sim_o"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Results manifest: {csv_path}")
    print(f"     Run metrics: .venv/bin/python scripts/evaluate.py --csv {csv_path} --metric all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
