Phase 2 — CFG Strength Ablation
================================

Purpose:
  Sweep Classifier-Free Guidance (CFG) strength to find the best trade-off
  between text faithfulness (WER) and speaker identity preservation (SIM-o).
  Fixed settings: Sway s=-1.0, NFE=32.

CFG controls how strongly the model follows the text conditioning vs.
the acoustic reference. Higher CFG → more text-faithful, lower SIM-o.

Sweep:
  cfg_strength ∈ {1.0, 1.5, 2.0, 2.5, 3.0}
  × 3 sentences = 15 WAVs

Contents:
  audio/     15 WAV files named ab0{1-3}_cfg{strength}.wav
  results_metrics.csv   WER (%) and SIM-o for every WAV

Key finding:
  cfg=1.0  → avg WER 17.2%, SIM-o 0.835  (good identity, less text-faithful)
  cfg=2.0  → avg WER  7.4%, SIM-o 0.802  (paper default, good balance)
  cfg=3.0  → avg WER  3.7%, SIM-o 0.774  (most text-faithful, lower identity)

Paper default cfg=2.0 confirmed as a good operating point.

See: results/plots/cfg_strength.png
Script: scripts/run_all_phases.py --phases 2
