#!/usr/bin/env python3
"""
Step 8 — Run Full Experiment
==============================
Orchestrator that runs all experiment steps sequentially:

  1. Baseline reproduction (English TTS + sway/CFG ablations + eval)
  2. Extension 1: Direct Mel Injection
  3. Extension 2, Method A: SDEdit Noise Injection
  4. Extension 2, Method B: Style Guidance (2-Pass ODE)
  5. Extension 2, Method C: Scheduled Conditioning Blend
  6. Extension 2, Method D: Noise Statistics Transfer
  7. All graphs + cross-method comparison

Each step is run as a subprocess so environment patches don't conflict.
Use --steps to run only specific steps, or --skip to skip steps.

Usage:
    # Run the entire experiment end-to-end:
    .venv/Scripts/python scripts/8_run_experiment.py \
        --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav

    # Run only steps 1 and 2:
    .venv/Scripts/python scripts/8_run_experiment.py \
        --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
        --steps 1 2

    # Skip the baseline (already done) and graphs:
    .venv/Scripts/python scripts/8_run_experiment.py \
        --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
        --skip 1 7

    # Use CPU:
    .venv/Scripts/python scripts/8_run_experiment.py \
        --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
        --device cpu
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEP_DESCRIPTIONS = {
    1: "Baseline reproduction (English TTS + sway/CFG ablations + eval)",
    2: "Extension 1: Direct Mel Injection",
    3: "Extension 2, Method A: SDEdit Noise Injection",
    4: "Extension 2, Method B: Style Guidance (2-Pass ODE)",
    5: "Extension 2, Method C: Scheduled Conditioning Blend",
    6: "Extension 2, Method D: Noise Statistics Transfer",
    7: "All graphs + cross-method comparison",
}


def run_step(step_num, python, args):
    """Run a single experiment step. Returns (success, elapsed_seconds)."""
    print("\n" + "#" * 70)
    print(f"# STEP {step_num}: {STEP_DESCRIPTIONS[step_num]}")
    print("#" * 70 + "\n")

    t0 = time.time()

    if step_num == 1:
        cmd = [
            python, str(SCRIPTS_DIR / "1_baseline.py"),
            "--ref_audio", args.ref_audio,
            "--outdir", str(Path(args.outdir) / "baseline"),
            "--device", args.device,
        ]

    elif step_num == 2:
        cmd = [
            python, str(SCRIPTS_DIR / "2_extension1.py"),
            "--device", args.device,
        ]

    elif step_num == 3:
        cmd = [
            python, str(SCRIPTS_DIR / "3_method_A.py"),
            "--device", args.device,
            "--resume",
        ]

    elif step_num == 4:
        cmd = [
            python, str(SCRIPTS_DIR / "4_method_B.py"),
            "--device", args.device,
        ]

    elif step_num == 5:
        cmd = [
            python, str(SCRIPTS_DIR / "5_method_C.py"),
            "--device", args.device,
        ]

    elif step_num == 6:
        cmd = [
            python, str(SCRIPTS_DIR / "6_method_D.py"),
            "--device", args.device,
        ]

    elif step_num == 7:
        cmd = [
            python, str(SCRIPTS_DIR / "7_graphs.py"),
            "--graphs", "all",
            "--ref_audio", args.ref_audio,
            "--outdir", args.outdir,
        ]

    else:
        print(f"  Unknown step {step_num}, skipping.")
        return False, 0.0

    print(f"  CMD: {' '.join(cmd)}\n")
    returncode = subprocess.call(cmd)
    elapsed = time.time() - t0

    if returncode != 0:
        print(f"\n  [WARNING] Step {step_num} exited with code {returncode}")
        return False, elapsed

    print(f"\n  [OK] Step {step_num} completed in {elapsed:.1f}s")
    return True, elapsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full F5-TTS Acoustic Style Transfer experiment"
    )
    parser.add_argument(
        "--ref_audio", "-r",
        default="F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
        help="Reference WAV file for voice cloning (used by baseline)",
    )
    parser.add_argument(
        "--outdir", "-o", default="results",
        help="Root output directory for results",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to use: cuda or cpu",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Run only these steps (e.g. --steps 1 2 3). Default: all.",
    )
    parser.add_argument(
        "--skip", nargs="+", type=int, default=[],
        help="Skip these steps (e.g. --skip 1 7)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python = sys.executable

    all_steps = [1, 2, 3, 4, 5, 6, 7]
    if args.steps is not None:
        steps_to_run = [s for s in args.steps if s in all_steps]
    else:
        steps_to_run = all_steps

    steps_to_run = [s for s in steps_to_run if s not in args.skip]

    print("=" * 70)
    print("F5-TTS Acoustic Style Transfer — Full Experiment")
    print("=" * 70)
    print(f"  Reference audio : {args.ref_audio}")
    print(f"  Output dir      : {args.outdir}")
    print(f"  Device          : {args.device}")
    print(f"  Steps to run    : {steps_to_run}")
    print()

    results = {}
    total_t0 = time.time()

    for step in steps_to_run:
        success, elapsed = run_step(step, python, args)
        results[step] = (success, elapsed)

    total_elapsed = time.time() - total_t0

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for step in steps_to_run:
        success, elapsed = results[step]
        status = "OK" if success else "FAILED"
        print(f"  Step {step}: [{status:6s}] {STEP_DESCRIPTIONS[step]:<55s} ({elapsed:.1f}s)")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    failed = [s for s in steps_to_run if not results[s][0]]
    if failed:
        print(f"\n  WARNING: Steps {failed} had errors. Check output above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
