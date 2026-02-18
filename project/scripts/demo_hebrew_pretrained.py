#!/usr/bin/env python3
"""
Phase 1.3 — Try Hebrew with the PRETRAINED (English+Chinese) F5-TTS model.

Expected outcome: garbled / incoherent output, because Hebrew characters are
outside the training vocabulary of the pretrained model.

This is intentional — the failure motivates Hebrew fine-tuning (Phase 3).
Document the WER of these outputs (it will be very high) as a baseline.

Usage:
    .venv/bin/python scripts/demo_hebrew_pretrained.py \\
        --ref_audio samples/ref_hebrew.wav \\
        --outdir results/phase1/hebrew_pretrained
"""

import sys
import argparse
from pathlib import Path

import soundfile as sf
import torch

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

from hebrew_utils import normalize_hebrew_text

# Short sentences covering a range of phonemes
HEBREW_SENTENCES = [
    "שלום עולם",                                      # Hello world
    "מה שלומך היום?",                                  # How are you today?
    "אני אוהב ללמוד שפות חדשות.",                      # I love learning new languages.
    "ירושלים היא עיר עתיקה ויפה.",                    # Jerusalem is an ancient and beautiful city.
    "המחקר מראה תוצאות מעניינות מאוד.",                # The research shows very interesting results.
]

HEBREW_SENTENCES_ROMANIZED = [
    "shalom olam",
    "ma shlomcha hayom",
    "ani ohev lilmod safot chadashot",
    "yerushalayim hi ir atika veyafa",
    "hamechkar mare totzaot meanyenot meod",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Demo: Hebrew TTS with PRETRAINED model (expected to fail)"
    )
    p.add_argument("--ref_audio", "-r", required=True,
                   help="3–10 s reference WAV (any speaker)")
    p.add_argument("--outdir", "-o", default="results/phase1/hebrew_pretrained")
    p.add_argument("--nfe_steps", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_coef", type=float, default=-1.0)
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

    print("[INFO] Loading pretrained model (English+Chinese only) ...")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model("F5TTS_v1_Base", DiT, model_cfg).to(args.device).eval()
    vocoder = load_vocoder(is_local=False, local_path=None)

    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        args.ref_audio, "", device=args.device
    )

    print(
        "\n[INFO] NOTE: Pretrained model was trained on English + Chinese only."
        "\n             Hebrew characters are outside its vocabulary."
        "\n             Expect garbled / repetitive / silent output."
        "\n             This failure is the baseline that motivates fine-tuning.\n"
    )

    results = []
    for i, (sentence, romanized) in enumerate(
        zip(HEBREW_SENTENCES, HEBREW_SENTENCES_ROMANIZED), 1
    ):
        normalized = normalize_hebrew_text(sentence)
        torch.manual_seed(args.seed + i)
        print(f"  [{i}/{len(HEBREW_SENTENCES)}] {sentence}  ({romanized})")

        try:
            audio, sr, _ = infer_process(
                ref_audio_proc, ref_text_proc, normalized,
                model, vocoder,
                mel_spec_type="vocos",
                nfe_step=args.nfe_steps,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway_coef,
                device=args.device,
            )
            out_path = outdir / f"he_pretrained_{i:02d}.wav"
            sf.write(str(out_path), audio, sr)
            results.append((sentence, romanized, str(out_path), "OK"))
            print(f"         → {out_path}")
        except Exception as e:
            results.append((sentence, romanized, "FAILED", str(e)))
            print(f"         → FAILED: {e}")

    # Save a summary CSV for later WER computation
    summary_path = outdir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("index,hebrew,romanized,output_file,status\n")
        for i, (heb, rom, path, status) in enumerate(results, 1):
            f.write(f'{i},"{heb}","{rom}",{path},{status}\n')

    print(f"\n[OK] Summary saved to {summary_path}")
    print("     Next: run scripts/evaluate.py --wer on these files")
    print("     Expected WER: very high (model cannot handle Hebrew chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
