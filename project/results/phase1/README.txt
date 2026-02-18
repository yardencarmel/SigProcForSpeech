Phase 1 — Baseline Demos
========================

Purpose:
  Establish that the pretrained F5-TTS model works correctly for English and
  deliberately fails for Hebrew. The failure motivates the Hebrew fine-tuning
  extension (Phase 3.5).

Contents:
  english/
    en01.wav  "The weather today is quite pleasant..."
    en02.wav  "She quickly realized the answer..."
    en03.wav  "Flow matching with diffusion transformers..."
    en04.wav  "Can you repeat that one more time?..."
    en05.wav  "The conference will be held in San Francisco..."
    → All generated via zero-shot voice cloning from the Hebrew CV reference.
      These should sound clear and natural; they confirm the pipeline works.

  hebrew_pretrained/
    he01.wav  "שלום עולם, איך הולך לך היום?"
    he02.wav  "ירושלים היא עיר עתיקה ויפה מאוד."
    he03.wav  "המחקר מראה תוצאות מעניינות..."
    → Generated with the same pretrained EN+ZH model, Hebrew text.
      Expected to sound garbled — Hebrew characters are not in the pretrained
      vocabulary, so the model generates noise conditioned on unknown tokens.
      These are the "before fine-tuning" baseline.

Script: scripts/run_all_phases.py --phases 1
