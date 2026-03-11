Phase 2 — Sway Sampling Ablation
=================================

Purpose:
  Reproduce Figure 3 / Table 2 of the F5-TTS paper (Chen et al., ACL 2025).
  Sweep the Sway Sampling coefficient s and number of function evaluations
  (NFE) to verify that our inference pipeline matches the paper's results and
  to identify the best operating point.

Sway Sampling: a modified ODE timestep schedule that biases integration
steps toward the noisy end of the trajectory:
    f(t) = t + s * (cos(pi/2 * t) - 1 + t)
Negative s concentrates steps near t=0, improving quality at the cost of
no extra compute — the number of steps (NFE) stays the same.

Sweep:
  s   ∈ {+0.4, 0.0, -0.4, -0.8, -1.0}
  NFE ∈ {8, 16, 32}
  × 3 sentences = 45 WAVs

Contents:
  audio/      45 WAV files named ab0{1-3}_s{coef}_nfe{steps}.wav
  results_metrics.csv   WER (%) and SIM-o for every WAV

Key finding: s=-0.8 with NFE=32 achieves 0% WER on all 3 test sentences.
At NFE=8, s=-1.0 is best (20.7% WER), showing biased sampling helps most
when very few steps are used. Paper default s=-1.0 confirmed.

See: results/plots/sway_sampling.png, results/plots/sway_pdf.png
Script: scripts/run_all_phases.py --phases 2
