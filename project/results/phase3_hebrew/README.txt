Phase 3 — Hebrew Sentences (Pretrained Model)
=============================================

Purpose:
  A deliberate demonstration that the pretrained F5-TTS model (trained on
  English + Mandarin Chinese) cannot generate intelligible Hebrew. These
  outputs are the "before fine-tuning" baseline for the Hebrew extension.
  They become meaningful once Phase 3.5 (Hebrew fine-tuning) is complete
  and can be compared against the fine-tuned model's output on the same text.

Contents:
  he_test01.wav  "שלום, מה שלומך היום?"       (Hello, how are you today?)
  he_test02.wav  "ירושלים היא בירת ישראל..."  (Jerusalem is the capital of Israel...)
  he_test03.wav  "המצב הפך לבלתי נסבל."       (The situation became unbearable.)

  All three should sound garbled / incoherent with the pretrained model.

Note: he_test03 uses the same sentence as the emotion transfer experiment
(Phase 4), making it easy to compare a garbled pretrained output against
the emotion-transferred output once fine-tuning is complete.

Script: scripts/run_all_phases.py --phases 3
