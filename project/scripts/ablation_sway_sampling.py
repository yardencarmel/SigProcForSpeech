#!/usr/bin/env python3
"""
Phase 2.1 / 2.3 — Sway Sampling Ablation  (reproduces Figure 3 from the paper)

Sweeps Sway Sampling coefficient s and NFE step count over a fixed set of
English test prompts.  Computes WER and SIM-o for each setting.

The sway function from the paper (Eq. 7):
    f_sway(u; s) = u + s * (cos(π/2 * u) - 1 + u)

where:
    s  < 0  → biases toward smaller t (more density on early flow steps)
    s  = 0  → uniform sampling (baseline)
    s  > 0  → biases toward larger t

Paper reports best results at s = -1 (32 NFE Euler solver).

Usage:
    .venv/bin/python scripts/ablation_sway_sampling.py \\
        --ref_audio samples/ref_english.wav \\
        --outdir results/sway_sampling

After running, use scripts/plot_results.py to visualise the results.
"""

import sys
import csv
import argparse
from pathlib import Path
from itertools import product

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

# Sway Sampling coefficient values to sweep
SWAY_COEFS  = [0.4, 0.0, -0.4, -0.8, -1.0]

# NFE step counts to test (8 = fast, 32 = paper default)
NFE_STEPS   = [8, 16, 32]

# Fixed test sentences (short enough for fast runs; varied enough to test prosody)
TEST_SENTENCES = [
    ("en01", "The cat sat on the mat and looked out the window."),
    ("en02", "Scientists have discovered a new species of deep-sea fish."),
    ("en03", "She opened the letter and read it twice before responding."),
    ("en04", "Flow matching with diffusion transformers achieves excellent results."),
    ("en05", "The conference is scheduled for early next year in Berlin."),
]


def parse_args():
    p = argparse.ArgumentParser(description="Sway Sampling ablation")
    p.add_argument("--ref_audio", "-r", required=True,
                   help="Identity reference WAV (3–10 s clean speech)")
    p.add_argument("--outdir", "-o", default="results/sway_sampling")
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip generation if output file already exists")
    return p.parse_args()


def main():
    args = parse_args()

    if not F5_AVAILABLE:
        print("[ERROR] Install F5-TTS first.")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audio_dir = outdir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("[INFO] Loading model ...")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model("F5TTS_v1_Base", DiT, model_cfg).to(args.device).eval()
    vocoder = load_vocoder(is_local=False, local_path=None)

    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        args.ref_audio, "", device=args.device
    )

    total = len(SWAY_COEFS) * len(NFE_STEPS) * len(TEST_SENTENCES)
    print(f"[INFO] Running {total} inference calls "
          f"({len(SWAY_COEFS)} sway × {len(NFE_STEPS)} NFE × {len(TEST_SENTENCES)} sentences)\n")

    results_csv = outdir / "results.csv"
    fieldnames = [
        "sway_coef", "nfe_steps", "sentence_id", "transcript",
        "output_wav", "duration_s",
        "wer", "sim_o",   # filled in by evaluate.py
    ]

    rows = []
    done = 0
    for sway, nfe, (sent_id, text) in product(SWAY_COEFS, NFE_STEPS, TEST_SENTENCES):
        done += 1
        out_name = f"{sent_id}_s{sway:+.1f}_nfe{nfe:02d}.wav"
        out_path = audio_dir / out_name

        print(f"  [{done:03d}/{total}] s={sway:+.1f}  NFE={nfe}  {sent_id}", end="")

        if args.skip_existing and out_path.exists():
            duration = None
            print("  [SKIP]")
        else:
            torch.manual_seed(args.seed)
            audio, sr, _ = infer_process(
                ref_audio_proc, ref_text_proc, text,
                model, vocoder,
                mel_spec_type="vocos",
                nfe_step=nfe,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=sway,
                device=args.device,
            )
            sf.write(str(out_path), audio, sr)
            duration = len(audio) / sr
            print(f"  ({duration:.2f}s)")

        rows.append({
            "sway_coef": sway,
            "nfe_steps": nfe,
            "sentence_id": sent_id,
            "transcript": text,
            "output_wav": str(out_path),
            "duration_s": duration or "",
            "wer": "",
            "sim_o": "",
        })

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Audio saved to {audio_dir}/")
    print(f"     Results manifest: {results_csv}")
    print(
        "\nNext step: compute metrics\n"
        f"  .venv/bin/python scripts/evaluate.py --csv {results_csv} "
        f"--metric all --language en\n"
        "Then plot:\n"
        f"  .venv/bin/python scripts/plot_results.py --csv {results_csv}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
