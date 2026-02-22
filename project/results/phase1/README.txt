Phase 1 — English Baseline
===========================

Purpose:
  Establish that the pretrained F5-TTS model works correctly for English
  (its training language). These samples confirm the full pipeline is
  functional before running any extension experiments.

Contents:
  english/
    en01.wav  "The weather today is quite pleasant..."
    en02.wav  "She quickly realized the answer..."
    en03.wav  "Flow matching with diffusion transformers..."
    en04.wav  "Can you repeat that one more time?..."
    en05.wav  "The conference will be held in San Francisco..."
    → All generated via zero-shot voice cloning from the reference audio.
      These should sound clear and natural; they confirm the pipeline works.

Script: scripts/run_all_phases.py --phases 1
