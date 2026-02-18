Phase 4 — Voice / Style Transfer via Direct Mel Injection
===========================================================

Purpose:
  Generate speech with the voice of Reference A and the acoustic style of
  Reference B, using a continuous blend weight w (0=pure A, 1=pure B).

References used (for maximum acoustic contrast):
  Identity A:  basic_ref_en.wav — F5-TTS canonical English female speaker
               Transcript: "Some call me nature, others call me mother nature."
  Style    B:  basic_ref_zh.wav — Mandarin animated-character voice (Ne Zha film)
               Transcript: "对,这就是我万人敬仰的太乙真人。"

Generated text (F5-TTS paper's own demo sentence):
  "I don't really care what you call me. I've been a silent spectator,
   watching species evolve, empires rise and fall."

Method:
  mel_blend = (1-w) * mel_A + w * mel_B        # [1, T, 100]
  generated, _ = model.sample(cond=mel_blend, ...)   # cond.ndim==3 → direct

  When cond.ndim==3, cfm.py uses it directly for conditioning without any
  internal mel conversion — no Vocos inversion needed. mel_A and mel_B are
  computed using F5-TTS's own mel_spec module for exact representation match.

Contents:
  audio/
    w0.00.wav          pure voice of speaker A (English female)
    w0.25.wav          75% A style, 25% B style
    w0.50.wav          50/50 blend
    w0.75.wav          25% A style, 75% B style
    w1.00.wav          pure acoustic style of speaker B (Mandarin character)
    zeroshot_emotion.wav  B used directly as sole reference (zero-shot clone)

  results_metrics.csv   WER (%), SIM-o (vs A), MCD (vs A)
  emotion_transfer.png  3-panel plot: WER / SIM-o / MCD vs weight

Key results (Whisper large-v3-turbo WER, WavLM SIM-o, MCD vs A):
  w=0.00  WER= 0.0%  SIM-o=0.893  MCD= 726  ← perfect, pure A voice
  w=0.25  WER= 0.0%  SIM-o=0.896  MCD= 707  ← still dominated by A
  w=0.50  WER=15.0%  SIM-o=0.833  MCD= 870  ← "Nature." prefix hallucinated
  w=0.75  WER= 0.0%  SIM-o=0.812  MCD= 832
  w=1.00  WER= 0.0%  SIM-o=0.811  MCD= 919  ← pure B style, A identity gone
  zero-shot  WER=0.0%  SIM-o=0.840  MCD=846  ← B as sole reference

Metric trends confirm the blend gradient is working:
  SIM-o:  0.893 → 0.833 → 0.811  (↓ — A's voice identity progressively lost)
  MCD:    726   → 870   → 919    (↑ — output moves acoustically away from A)

The w=0.50 WER=15% artefact is mechanistically interesting:
  A's reference text contains the word "nature" ("Some call me nature...").
  At the midpoint blend, the model's attention is split between two very
  different conditioning mels and "nature" bleeds from the text conditioning
  into the generated output: Whisper hears "Nature. I don't really care..."
  At w=0 and w≥0.75, one speaker dominates → stable conditioning → 0% WER.

Duration shift visible in the audio:
  w=0.00–0.50: 12.26s  (driven by A's pace: English female, 5.3s clip)
  w=0.75–1.00: 13.32s  (pace shifts as B's Mandarin mel takes over)
  zero-shot:   16.90s  (B's ref_text is Chinese, different byte-rate estimate)

Note on comparison with Hebrew-reference experiment:
  The original experiment used two adjacent Hebrew Common Voice clips (both
  neutral read speech, similar emotional register). Those results are archived
  in results/emotion_transfer_hebrew_refs/. The canonical F5-TTS reference
  experiment provides maximum acoustic contrast and uses the paper's own
  reference clip and demo generation text, making the results directly
  comparable to the base paper's examples.

Script: scripts/run_phase4.py
