#!/usr/bin/env python3
"""
F5-TTS Hebrew Inference with Emotion Transfer

This script generates Hebrew speech using F5-TTS with:
1. Voice cloning from a reference audio (identity)
2. Optional emotion/style transfer from a second reference audio

The emotion transfer is EXPERIMENTAL and works by blending the acoustic
features of two reference audios during generation.

Usage:
    # Basic Hebrew TTS with voice cloning
    python run_emotion_transfer.py \
        --text "שלום עולם" \
        --ref_audio speaker.wav \
        --output output.wav
    
    # With emotion transfer (experimental)
    python run_emotion_transfer.py \
        --text "שלום עולם" \
        --ref_audio_identity neutral_speaker.wav \
        --ref_audio_emotion angry_sample.wav \
        --emotion_weight 0.3 \
        --output emotional_output.wav
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import soundfile as sf
import numpy as np

# Add F5-TTS to path
PROJECT_DIR = Path(__file__).parent.parent
F5_TTS_DIR = PROJECT_DIR / "F5-TTS"
if F5_TTS_DIR.exists():
    sys.path.insert(0, str(F5_TTS_DIR / "src"))

try:
    from f5_tts.model import DiT
    from f5_tts.model.cfm import CFM
    from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
    from f5_tts.infer.utils_infer import (
        load_vocoder,
        load_model,
        preprocess_ref_audio_text,
        infer_process,
    )
    F5_TTS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] F5-TTS not fully installed: {e}")
    print("Please run setup_project.py first.")
    F5_TTS_AVAILABLE = False


# Add hebrew_utils to path
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from hebrew_utils import normalize_hebrew_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Hebrew speech with optional emotion transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Hebrew text to synthesize"
    )
    
    parser.add_argument(
        "--ref_audio", "-r",
        type=str,
        default=None,
        help="Reference audio for voice cloning (identity)"
    )
    
    parser.add_argument(
        "--ref_audio_identity",
        type=str,
        default=None,
        help="Reference audio for voice identity (alternative to --ref_audio)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    
    # Emotion transfer arguments
    parser.add_argument(
        "--ref_audio_emotion",
        type=str,
        default=None,
        help="Reference audio for emotion/style (experimental)"
    )
    
    parser.add_argument(
        "--emotion_weight",
        type=float,
        default=0.3,
        help="Weight for emotion reference (0.0-1.0, default: 0.3)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model checkpoint (uses pretrained if not specified)"
    )
    
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to vocabulary file (default: data/Hebrew_Dataset/vocab.txt)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--nfe_steps",
        type=int,
        default=32,
        help="Number of function evaluations for diffusion (default: 32)"
    )
    
    parser.add_argument(
        "--cfg_strength",
        type=float,
        default=2.0,
        help="Classifier-free guidance strength (default: 2.0)"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer NFE steps (8 instead of 32)"
    )
    
    return parser.parse_args()


def load_audio(path: str, target_sr: int = 24000) -> Tuple[torch.Tensor, int]:
    """Load and resample audio file."""
    audio, sr = torchaudio.load(path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    return audio.squeeze(0), target_sr


def blend_audio_features(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    weight: float = 0.3
) -> torch.Tensor:
    """
    Blend acoustic features of two audio samples.
    
    This is a simple blending approach that mixes the mel spectrograms.
    More sophisticated approaches could blend at different model layers.
    
    Args:
        audio1: Primary audio (identity)
        audio2: Secondary audio (emotion)
        weight: Weight for secondary audio (0.0-1.0)
        
    Returns:
        Blended audio
    """
    # Ensure same length (truncate to shorter)
    min_len = min(audio1.shape[-1], audio2.shape[-1])
    audio1 = audio1[..., :min_len]
    audio2 = audio2[..., :min_len]
    
    # Simple linear blend
    blended = (1 - weight) * audio1 + weight * audio2
    
    return blended


class HebrewTTSInference:
    """Hebrew TTS inference with emotion transfer support."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = device
        self.target_sample_rate = 24000
        
        # Default paths
        if vocab_path is None:
            vocab_path = str(PROJECT_DIR / "data" / "Hebrew_Dataset" / "vocab.txt")
        
        # Load tokenizer
        print(f"Loading vocabulary from {vocab_path}")
        if os.path.exists(vocab_path):
            self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_path, "custom")
            print(f"Vocabulary size: {self.vocab_size}")
        else:
            print("[WARNING] Hebrew vocab not found, using default tokenizer")
            self.vocab_char_map, self.vocab_size = None, 256
        
        # Load model
        print("Loading F5-TTS model...")
        self.model = self._load_model(model_path)
        
        # Load vocoder
        print("Loading vocoder...")
        self.vocoder = load_vocoder(is_local=False, local_path=None)
        
        print("Model loaded successfully!")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load F5-TTS model."""
        # Model configuration
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        
        # Mel spectrogram configuration
        mel_spec_kwargs = dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=self.target_sample_rate,
            mel_spec_type="vocos",
        )
        
        # Create model
        model = CFM(
            transformer=DiT(
                **model_cfg,
                text_num_embeds=self.vocab_size,
                mel_dim=100
            ),
            mel_spec_kwargs=mel_spec_kwargs,
            vocab_char_map=self.vocab_char_map,
        )
        
        # Load weights
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned weights from {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
        else:
            # Use pretrained F5-TTS
            print("Using pretrained F5-TTS model (downloading if needed)...")
            model = load_model(
                "F5TTS_v1_Base",
                DiT,
                model_cfg,
                vocab_file=None if self.vocab_char_map is None else "custom"
            )
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_audio_emotion_path: Optional[str] = None,
        emotion_weight: float = 0.3,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        speed: float = 1.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Hebrew speech with optional emotion transfer.
        
        Args:
            text: Hebrew text to synthesize
            ref_audio_path: Reference audio for voice identity
            ref_audio_emotion_path: Optional reference for emotion/style
            emotion_weight: Blending weight for emotion reference
            nfe_steps: Diffusion steps
            cfg_strength: Classifier-free guidance strength
            speed: Speech speed multiplier
            seed: Random seed
            
        Returns:
            (audio_array, sample_rate)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Normalize Hebrew text
        text = normalize_hebrew_text(text)
        print(f"Normalized text: {text}")
        
        # Load reference audio
        ref_audio, sr = load_audio(ref_audio_path, self.target_sample_rate)
        ref_text = ""  # Empty for zero-shot cloning
        
        # Handle emotion blending if second reference provided
        if ref_audio_emotion_path:
            print(f"Blending emotion from {ref_audio_emotion_path} (weight: {emotion_weight})")
            emotion_audio, _ = load_audio(ref_audio_emotion_path, self.target_sample_rate)
            ref_audio = blend_audio_features(ref_audio, emotion_audio, emotion_weight)
        
        # Preprocess reference
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            ref_audio_path, 
            ref_text,
            device=self.device
        )
        
        # Generate speech
        print("Generating speech...")
        with torch.no_grad():
            audio, sr, _ = infer_process(
                ref_audio_processed,
                ref_text_processed,
                text,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                speed=speed,
                nfe_step=nfe_steps,
                cfg_strength=cfg_strength,
                device=self.device,
            )
        
        return audio, sr


def main():
    args = parse_args()
    
    if not F5_TTS_AVAILABLE:
        print("[ERROR] F5-TTS is not installed. Please run setup_project.py first.")
        return 1
    
    # Handle ref_audio argument aliases
    ref_audio = args.ref_audio or args.ref_audio_identity
    if ref_audio is None:
        print("[ERROR] Reference audio is required. Use --ref_audio or --ref_audio_identity")
        return 1
    
    # Quick mode
    nfe_steps = 8 if args.quick else args.nfe_steps
    
    print("="*60)
    print("F5-TTS Hebrew Inference with Emotion Transfer")
    print("="*60)
    print(f"Text: {args.text}")
    print(f"Reference (identity): {ref_audio}")
    if args.ref_audio_emotion:
        print(f"Reference (emotion): {args.ref_audio_emotion}")
        print(f"Emotion weight: {args.emotion_weight}")
    print(f"Output: {args.output}")
    print(f"NFE steps: {nfe_steps}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Initialize inference
    tts = HebrewTTSInference(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        device=args.device
    )
    
    # Generate
    audio, sr = tts.generate(
        text=args.text,
        ref_audio_path=ref_audio,
        ref_audio_emotion_path=args.ref_audio_emotion,
        emotion_weight=args.emotion_weight,
        nfe_steps=nfe_steps,
        cfg_strength=args.cfg_strength,
        speed=args.speed,
        seed=args.seed
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    
    print(f"\n[OK] Generated audio saved to: {output_path}")
    print(f"Duration: {len(audio) / sr:.2f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
