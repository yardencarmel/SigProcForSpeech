#!/usr/bin/env python3
"""
F5-TTS Hebrew Inference with Emotion Transfer (Mel-Space Blending)

Two reference audios instead of one:
  Audio A (identity): A recording of the target speaker (e.g. you)
  Audio B (style/emotion): A recording with the desired emotion/style
                           (may be in a different language)

Goal: Generate speech that sounds like speaker A but with the style of B.

Implementation: blend happens in **mel-spectrogram space** (not waveforms).
Waveform blending produces phase-incoherent noise; mel blending is acoustically
coherent and matches how F5-TTS conditions generation internally.

Bug fixed vs previous version:
  The original code computed a blended waveform but then passed the original
  path to preprocess_ref_audio_text(), so the blend was silently discarded.
  This version inverts the blended mel back to a temp WAV so the full pipeline
  actually sees the blend.

Usage:
    # Basic Hebrew TTS — voice cloning only
    python run_emotion_transfer.py \\
        --text "שלום עולם" \\
        --ref_audio speaker.wav \\
        --output output.wav

    # With emotion/style transfer
    python run_emotion_transfer.py \\
        --text "שלום עולם" \\
        --ref_audio_identity neutral_speaker.wav \\
        --ref_audio_emotion angry_sample.wav \\
        --emotion_weight 0.4 \\
        --output emotional_output.wav

    # Cross-lingual: Hebrew identity + English scream
    python run_emotion_transfer.py \\
        --text "אני מאוד כועס!" \\
        --ref_audio_identity hebrew_speaker.wav \\
        --ref_audio_emotion english_scream.wav \\
        --emotion_weight 0.5 \\
        --output cross_lingual_output.wav
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).parent.parent
F5_TTS_DIR = PROJECT_DIR / "F5-TTS"
if F5_TTS_DIR.exists():
    sys.path.insert(0, str(F5_TTS_DIR / "src"))

try:
    from f5_tts.model import DiT
    from f5_tts.model.cfm import CFM
    from f5_tts.model.utils import get_tokenizer
    from f5_tts.infer.utils_infer import (
        load_vocoder,
        load_model,
        preprocess_ref_audio_text,
        infer_process,
    )
    F5_TTS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] F5-TTS not fully installed: {e}")
    F5_TTS_AVAILABLE = False

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from hebrew_utils import normalize_hebrew_text


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 24000) -> Tuple[torch.Tensor, int]:
    """Load and resample to mono at target_sr."""
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0), target_sr


def audio_to_mel(
    audio: torch.Tensor,
    sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 100,
) -> torch.Tensor:
    """
    Convert audio waveform to log-mel spectrogram.

    Uses the same mel parameters as F5-TTS / vocos:
      - amplitude spectrum (power=1)
      - slaney norm and mel scale
    Returns shape (n_mels, T).
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    )
    mel = mel_transform(audio.unsqueeze(0)).squeeze(0)   # (n_mels, T)
    mel = torch.log(torch.clamp(mel, min=1e-5))          # log-mel
    return mel


def mel_to_audio(mel: torch.Tensor, vocoder, device: str = "cpu") -> np.ndarray:
    """
    Invert a log-mel spectrogram back to a waveform using the vocos vocoder.

    mel: (n_mels, T)
    Returns numpy array of audio samples at 24 kHz.
    """
    mel_batch = mel.unsqueeze(0).to(device)   # (1, n_mels, T)
    with torch.no_grad():
        audio = vocoder.decode(mel_batch)     # (1, samples)
    return audio.squeeze().cpu().numpy()


def blend_mels(
    mel_identity: torch.Tensor,
    mel_emotion: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """
    Linear interpolation of two log-mel spectrograms.

    Both mels are trimmed to the shorter one along the time axis before
    blending so the resulting conditioning length is consistent.

    weight = 0.0  -> pure identity
    weight = 1.0  -> pure emotion
    """
    T = min(mel_identity.shape[-1], mel_emotion.shape[-1])
    m1 = mel_identity[..., :T]
    m2 = mel_emotion[..., :T]
    return (1.0 - weight) * m1 + weight * m2


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="F5-TTS: Hebrew TTS with mel-space emotion/style transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", "-t", required=True,
                        help="Text to synthesize (Hebrew or any language)")
    parser.add_argument("--ref_audio", "-r", default=None,
                        help="Reference audio for voice cloning (identity)")
    parser.add_argument("--ref_audio_identity", default=None,
                        help="Alias for --ref_audio")
    parser.add_argument("--ref_audio_emotion", default=None,
                        help="Reference audio for emotion/style (may be cross-lingual)")
    parser.add_argument("--emotion_weight", type=float, default=0.35,
                        help="Style blend weight: 0=pure identity, 1=pure emotion "
                             "(default 0.35)")
    parser.add_argument("--output", "-o", default="output.wav",
                        help="Output WAV file path")
    parser.add_argument("--model_path", default=None,
                        help="Fine-tuned checkpoint (uses pretrained F5TTS_v1_Base if omitted)")
    parser.add_argument("--vocab_path", default=None,
                        help="Custom vocab.txt for Hebrew; omit for default tokenizer")
    parser.add_argument("--nfe_steps", type=int, default=32,
                        help="Diffusion NFE steps (default 32)")
    parser.add_argument("--cfg_strength", type=float, default=2.0,
                        help="Classifier-free guidance strength (default 2.0)")
    parser.add_argument("--sway_coef", type=float, default=-1.0,
                        help="Sway Sampling coefficient s; 0=uniform, <0 biases toward "
                             "early steps (default -1.0, best per paper)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Quick preview: 8 NFE steps")
    parser.add_argument("--no_normalize_hebrew", action="store_true",
                        help="Skip Hebrew text normalization")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class F5TTSWithStyleTransfer:
    """
    F5-TTS inference with mel-space style/emotion transfer.

    When an emotion reference is provided the model is conditioned on a
    linear interpolation of the two log-mel spectrograms (identity + emotion).
    The text drives *what* is said; the blended mel drives *how* it sounds.

    Cross-lingual use: the mel spectrogram is language-agnostic (it is
    purely acoustic), so an English emotion reference can be blended with
    a Hebrew identity reference without issues.
    """

    TARGET_SR = 24000

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device

        if vocab_path and os.path.exists(vocab_path):
            print(f"[INFO] Loading custom vocabulary: {vocab_path}")
            self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_path, "custom")
        else:
            print("[INFO] Using default F5-TTS tokenizer")
            self.vocab_char_map, self.vocab_size = None, None

        print("[INFO] Loading F5-TTS model ...")
        self.model = self._load_model(model_path)

        print("[INFO] Loading vocoder ...")
        self.vocoder = load_vocoder(is_local=False, local_path=None)

        print("[INFO] Ready.")

    def _load_model(self, model_path: Optional[str]):
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4,
        )
        if model_path and os.path.exists(model_path):
            mel_kwargs = dict(
                n_fft=1024, hop_length=256, win_length=1024,
                n_mel_channels=100, target_sample_rate=self.TARGET_SR,
                mel_spec_type="vocos",
            )
            model = CFM(
                transformer=DiT(
                    **model_cfg,
                    text_num_embeds=self.vocab_size or 256,
                    mel_dim=100,
                ),
                mel_spec_kwargs=mel_kwargs,
                vocab_char_map=self.vocab_char_map,
            )
            ckpt = torch.load(model_path, map_location=self.device)
            model.load_state_dict(ckpt.get("model", ckpt))
            print(f"[INFO] Loaded fine-tuned weights: {model_path}")
        else:
            model = load_model("F5TTS_v1_Base", DiT, model_cfg)
            print("[INFO] Using pretrained F5TTS_v1_Base")
        return model.to(self.device).eval()

    def _build_blended_ref_wav(
        self,
        identity_path: str,
        emotion_path: str,
        weight: float,
    ) -> str:
        """
        Blend two audio references in mel space and write the result to a
        temporary WAV file.  Returns the temp file path (caller must delete).
        """
        audio_id, sr = load_audio(identity_path, self.TARGET_SR)
        audio_em, _  = load_audio(emotion_path,  self.TARGET_SR)

        mel_id = audio_to_mel(audio_id, sr)
        mel_em = audio_to_mel(audio_em, sr)

        mel_blended = blend_mels(mel_id, mel_em, weight)

        blended_np = mel_to_audio(mel_blended, self.vocoder, self.device)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, blended_np, self.TARGET_SR)
        return tmp.name

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_audio_emotion_path: Optional[str] = None,
        emotion_weight: float = 0.35,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_coef: float = -1.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
        normalize_hebrew: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech.  Returns (audio_np, sample_rate)."""
        if seed is not None:
            torch.manual_seed(seed)

        if normalize_hebrew:
            text = normalize_hebrew_text(text)
        print(f"[INFO] Text: {text}")

        # Choose effective reference (identity-only, emotion-only, or blend)
        tmp_path = None
        if ref_audio_emotion_path and 0.0 < emotion_weight < 1.0:
            print(
                f"[INFO] Mel-space blend: identity={os.path.basename(ref_audio_path)} "
                f"emotion={os.path.basename(ref_audio_emotion_path)} "
                f"weight={emotion_weight}"
            )
            tmp_path = self._build_blended_ref_wav(
                ref_audio_path, ref_audio_emotion_path, emotion_weight
            )
            effective_ref = tmp_path
        elif ref_audio_emotion_path and emotion_weight >= 1.0:
            effective_ref = ref_audio_emotion_path
        else:
            effective_ref = ref_audio_path

        # Preprocess (trim to ≤15 s, handle transcription)
        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
            effective_ref, "", device=self.device
        )

        # Inference
        audio, sr, _ = infer_process(
            ref_audio_proc,
            ref_text_proc,
            text,
            self.model,
            self.vocoder,
            mel_spec_type="vocos",
            speed=speed,
            nfe_step=nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_coef,
            device=self.device,
        )

        if tmp_path:
            os.unlink(tmp_path)

        return audio, sr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not F5_TTS_AVAILABLE:
        print("[ERROR] F5-TTS not installed. Run: pip install -e F5-TTS/")
        return 1

    ref_audio = args.ref_audio or args.ref_audio_identity
    if not ref_audio:
        print("[ERROR] Provide --ref_audio or --ref_audio_identity")
        return 1

    nfe = 8 if args.quick else args.nfe_steps

    print("=" * 62)
    print("F5-TTS  —  Mel-Space Emotion Transfer")
    print("=" * 62)
    print(f"  Text              : {args.text}")
    print(f"  Identity ref      : {ref_audio}")
    if args.ref_audio_emotion:
        print(f"  Emotion ref       : {args.ref_audio_emotion}")
        print(f"  Emotion weight    : {args.emotion_weight}  "
              "(0=identity only, 1=emotion only)")
    print(f"  NFE steps         : {nfe}")
    print(f"  CFG strength      : {args.cfg_strength}")
    print(f"  Sway coef (s)     : {args.sway_coef}")
    print(f"  Output            : {args.output}")
    print(f"  Device            : {args.device}")
    print("=" * 62)

    tts = F5TTSWithStyleTransfer(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        device=args.device,
    )

    audio, sr = tts.generate(
        text=args.text,
        ref_audio_path=ref_audio,
        ref_audio_emotion_path=args.ref_audio_emotion,
        emotion_weight=args.emotion_weight,
        nfe_steps=nfe,
        cfg_strength=args.cfg_strength,
        sway_coef=args.sway_coef,
        speed=args.speed,
        seed=args.seed,
        normalize_hebrew=not args.no_normalize_hebrew,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr)
    print(f"\n[OK] Saved → {out}  ({len(audio)/sr:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
