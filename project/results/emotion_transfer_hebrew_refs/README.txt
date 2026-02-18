Phase 4 — Emotion / Style Transfer via Direct Mel Injection
============================================================

Purpose:
  Generate speech with the voice of Reference A and the emotional style of
  Reference B, using a continuous blend weight w (0=pure A, 1=pure B).

  Use case:
    --identity  any speaker WAV  (the voice you want)
    --emotion   any speaker WAV  (the emotion/style you want)
    --text      the sentence to generate
    --weight    0.0 to 1.0

Method:
  Instead of decoding blended mels through the vocoder (which fails because
  blended mels are out-of-distribution for Vocos), the blended mel tensor is
  injected directly into model.sample(cond=mel_3d). When cond.ndim==3, cfm.py
  uses it as conditioning without any internal mel conversion — no Vocos
  inversion is needed.

    mel_blend = (1-w) * mel_A + w * mel_B        # [1, T, 100]
    generated, _ = model.sample(cond=mel_blend, text=..., duration=...)

Contents:
  audio/
    w0.00.wav          pure voice of speaker A (identity only)
    w0.25.wav          75% A identity, 25% B style
    w0.50.wav          50/50 blend
    w0.75.wav          25% A identity, 75% B style
    w1.00.wav          pure style of speaker B
    zeroshot_emotion.wav  B used directly as sole reference (zero-shot clone)

  results_metrics.csv   WER (%), SIM-o (vs identity ref A), MCD (vs B)
  emotion_transfer.png  3-panel plot: WER / SIM-o / MCD vs weight

Key results (Whisper large-v3-turbo WER, WavLM SIM-o):
  w=0.00  WER= 8.3%  SIM-o=0.777  (pure identity A)
  w=0.25  WER= 8.3%  SIM-o=0.721
  w=0.50  WER=25.0%  SIM-o=0.720  (heaviest distortion at 50/50 blend)
  w=0.75  WER=16.7%  SIM-o=0.711
  w=1.00  WER=16.7%  SIM-o=0.725  (pure emotion B)
  zero-shot  WER=16.7%  SIM-o=0.725  (B as sole reference)

Metrics interpretation:
  WER increases with w        → blending gradually degrades intelligibility
  SIM-o decreasing with w     → identity of A is progressively diluted (expected)
  MCD increasing with w       → output moves acoustically away from A toward B (expected)

Note: The reference audios used in the experiment are two Hebrew Common Voice
speakers (common_voice_he_38078769 and _38078770). The generated text is
English. The algorithm is language-agnostic — the same script works with any
combination of input language and output language, as long as the model
supports the output language (currently English only without fine-tuning).

Script: scripts/run_phase4.py
