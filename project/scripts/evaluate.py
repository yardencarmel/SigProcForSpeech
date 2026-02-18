#!/usr/bin/env python3
"""
Phase 5 — Evaluation Metrics

Computes:
  --metric wer    Word Error Rate via Whisper large-v3 (supports Hebrew)
  --metric sim    Speaker Similarity via WavLM-based cosine distance
  --metric mcd    Mel Cepstral Distortion (for emotion transfer quality)
  --metric all    All of the above

Input: a CSV file with columns:
  generated_wav, reference_wav [, emotion_wav, transcript]

Output: metrics appended to the input CSV and a summary table printed to stdout.

Usage:
    # WER only (needs transcripts column)
    .venv/bin/python scripts/evaluate.py \\
        --csv results/sway_sampling/results.csv \\
        --metric wer

    # Speaker similarity (generated vs identity reference)
    .venv/bin/python scripts/evaluate.py \\
        --csv results/sway_sampling/results.csv \\
        --metric sim

    # All metrics
    .venv/bin/python scripts/evaluate.py \\
        --csv results/phase1/results.csv \\
        --metric all

    # Quick single-file evaluation
    .venv/bin/python scripts/evaluate.py \\
        --generated output.wav \\
        --reference ref_identity.wav \\
        --transcript "the text that was synthesized" \\
        --metric all
"""

import sys
import argparse
import csv
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Token-level word error rate.
      WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference word count.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    N = len(ref_words)
    if N == 0:
        return 0.0

    # Dynamic programming (Wagner-Fischer)
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / N


class WhisperASR:
    """Wrapper around openai/whisper-large-v3 for transcription."""

    def __init__(self, device: str = "cpu"):
        print("[INFO] Loading Whisper large-v3 (downloads on first run ~3 GB) ...")
        try:
            import whisper
            self.model = whisper.load_model("large-v3", device=device)
            self._backend = "whisper"
        except ImportError:
            try:
                from transformers import pipeline
                self.model = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3",
                    device=device,
                )
                self._backend = "transformers"
            except ImportError:
                raise ImportError(
                    "Install either 'openai-whisper' or 'transformers' for ASR.\n"
                    "  pip install openai-whisper\n"
                    "  # or\n"
                    "  pip install transformers"
                )
        print("[INFO] Whisper ready.")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        if self._backend == "whisper":
            opts = {}
            if language:
                opts["language"] = language
            result = self.model.transcribe(audio_path, **opts)
            return result["text"].strip()
        else:
            kwargs = {}
            if language:
                kwargs["generate_kwargs"] = {"language": language}
            result = self.model(audio_path, **kwargs)
            return result["text"].strip()


# ---------------------------------------------------------------------------
# Speaker Similarity (SIM-o)
# ---------------------------------------------------------------------------

class SpeakerSimilarity:
    """
    WavLM-large based speaker similarity following the paper's SIM-o metric.
    Extracts speaker embeddings and computes cosine similarity.
    """

    def __init__(self, device: str = "cpu"):
        print("[INFO] Loading WavLM for speaker similarity ...")
        try:
            from transformers import WavLMModel, AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                "microsoft/wavlm-large"
            )
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
            self.model.eval()
            self.device = device
            print("[INFO] WavLM ready.")
        except Exception as e:
            raise ImportError(
                f"Could not load WavLM: {e}\n"
                "  pip install transformers"
            )

    def _get_embedding(self, audio_path: str) -> np.ndarray:
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.squeeze().numpy()

        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pool over time
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def similarity(self, generated_path: str, reference_path: str) -> float:
        emb_gen = self._get_embedding(generated_path)
        emb_ref = self._get_embedding(reference_path)
        cos_sim = np.dot(emb_gen, emb_ref) / (
            np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref) + 1e-8
        )
        return float(cos_sim)


# ---------------------------------------------------------------------------
# Mel Cepstral Distortion
# ---------------------------------------------------------------------------

def compute_mcd(generated_path: str, reference_path: str, sr: int = 24000) -> float:
    """
    Mel Cepstral Distortion between two audio files.

    MCD measures spectral distance in the cepstral domain.
    Lower = more similar acoustic characteristics.

    Both signals are trimmed/padded to the same length before comparison.
    """
    try:
        from python_speech_features import mfcc
    except ImportError:
        # Fallback: compute MFCC with torchaudio + librosa
        pass

    def load_and_resample(path):
        audio, orig_sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            audio = torchaudio.functional.resample(audio, orig_sr, sr)
        return audio.squeeze().numpy()

    audio_gen = load_and_resample(generated_path)
    audio_ref = load_and_resample(reference_path)

    # MFCC via torchaudio (24 coefficients, skip C0)
    def get_mfcc(signal):
        t = torch.tensor(signal).unsqueeze(0)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=25,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 100},
        )
        mfcc = mfcc_transform(t).squeeze(0)   # (n_mfcc, T)
        return mfcc[1:].T.numpy()             # drop C0, shape (T, 24)

    mfcc_gen = get_mfcc(audio_gen)
    mfcc_ref = get_mfcc(audio_ref)

    # Align length (DTW would be better, but frame-wise is fast)
    T = min(len(mfcc_gen), len(mfcc_ref))
    mfcc_gen = mfcc_gen[:T]
    mfcc_ref = mfcc_ref[:T]

    # MCD = (10 / ln(10)) * mean_over_frames( sqrt(2 * sum_c (c_gen - c_ref)^2) )
    diff = mfcc_gen - mfcc_ref
    mcd = (10.0 / np.log(10.0)) * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    return float(mcd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluation: WER / SIM-o / MCD")
    p.add_argument("--metric", choices=["wer", "sim", "mcd", "all"], default="all")

    # Single-file mode
    p.add_argument("--generated", default=None, help="Generated WAV")
    p.add_argument("--reference", default=None, help="Identity reference WAV")
    p.add_argument("--emotion_ref", default=None, help="Emotion reference WAV (for MCD)")
    p.add_argument("--transcript", default=None, help="Ground-truth transcript")
    p.add_argument("--language", default=None,
                   help="ASR language hint (e.g. 'he' for Hebrew, 'en' for English)")

    # Batch mode
    p.add_argument("--csv", default=None,
                   help="CSV with columns: generated_wav,reference_wav,transcript[,emotion_wav]")
    p.add_argument("--outcsv", default=None,
                   help="Output CSV (defaults to input CSV with _metrics suffix)")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate_single(args, asr=None, sim_model=None):
    results = {}

    if args.metric in ("wer", "all") and args.transcript:
        if asr is None:
            asr = WhisperASR(args.device)
        hypothesis = asr.transcribe(args.generated, language=args.language)
        wer = compute_wer(hypothesis, args.transcript)
        results["transcript_hyp"] = hypothesis
        results["wer"] = round(wer, 4)
        print(f"  Hypothesis : {hypothesis}")
        print(f"  Reference  : {args.transcript}")
        print(f"  WER        : {wer:.4f}  ({wer*100:.1f}%)")

    if args.metric in ("sim", "all") and args.reference:
        if sim_model is None:
            sim_model = SpeakerSimilarity(args.device)
        sim = sim_model.similarity(args.generated, args.reference)
        results["sim_o"] = round(sim, 4)
        print(f"  SIM-o      : {sim:.4f}")

    if args.metric in ("mcd", "all") and args.emotion_ref:
        mcd = compute_mcd(args.generated, args.emotion_ref)
        results["mcd"] = round(mcd, 4)
        print(f"  MCD        : {mcd:.4f} dB")

    return results, asr, sim_model


def main():
    args = parse_args()

    if args.generated:
        # Single-file mode
        print(f"\nEvaluating: {args.generated}")
        print("-" * 50)
        results, _, _ = evaluate_single(args)
        print("\nSummary:", results)
        return 0

    if args.csv:
        # Batch mode
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}")
            return 1

        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        asr = None
        sim_model = None
        all_wers, all_sims, all_mcds = [], [], []

        for i, row in enumerate(rows, 1):
            print(f"\n[{i}/{len(rows)}] {row.get('generated_wav', '?')}")
            args.generated  = row.get("generated_wav")
            args.reference  = row.get("reference_wav")
            args.emotion_ref = row.get("emotion_wav")
            args.transcript = row.get("transcript")

            if not args.generated or not Path(args.generated).exists():
                print("  [SKIP] file not found")
                continue

            metrics, asr, sim_model = evaluate_single(args, asr, sim_model)
            row.update(metrics)
            if "wer" in metrics: all_wers.append(metrics["wer"])
            if "sim_o" in metrics: all_sims.append(metrics["sim_o"])
            if "mcd" in metrics: all_mcds.append(metrics["mcd"])

        # Write output CSV
        outcsv = args.outcsv or str(csv_path.with_suffix("")) + "_metrics.csv"
        fieldnames = list(rows[0].keys())
        for m in ("wer", "sim_o", "mcd", "transcript_hyp"):
            if m not in fieldnames:
                fieldnames.append(m)
        with open(outcsv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        if all_wers:
            print(f"  Mean WER  : {np.mean(all_wers):.4f}  ({np.mean(all_wers)*100:.1f}%)")
        if all_sims:
            print(f"  Mean SIM-o: {np.mean(all_sims):.4f}")
        if all_mcds:
            print(f"  Mean MCD  : {np.mean(all_mcds):.4f} dB")
        print(f"\n  Results saved to: {outcsv}")
        return 0

    print("[ERROR] Provide --generated (single mode) or --csv (batch mode)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
