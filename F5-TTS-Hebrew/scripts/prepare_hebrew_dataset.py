#!/usr/bin/env python3
"""
Hebrew Dataset Preparation Script for F5-TTS

This script prepares Hebrew audio datasets for F5-TTS fine-tuning.
It supports:
1. Mozilla Common Voice Hebrew format
2. Custom folder with .wav files + metadata.csv

Output:
- data/Hebrew_Dataset/vocab.txt (vocabulary file)
- data/Hebrew_Dataset/metadata.csv (F5-TTS format: audio_file|text)

Usage:
    # For Mozilla Common Voice
    python prepare_hebrew_dataset.py --input path/to/cv-corpus-he --format commonvoice
    
    # For custom dataset
    python prepare_hebrew_dataset.py --input path/to/custom_data --format custom
    
    # Dry run (no output, just validation)
    python prepare_hebrew_dataset.py --input path/to/data --dry-run
"""

import os
import sys
import csv
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from hebrew_utils import (
    normalize_hebrew_text,
    build_hebrew_vocab,
    save_vocab,
    is_hebrew_letter
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARNING] pandas not installed. Some features may be limited.")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("[WARNING] torchaudio not installed. Audio validation disabled.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Hebrew dataset for F5-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mozilla Common Voice Hebrew
  python prepare_hebrew_dataset.py --input cv-corpus-17.0-2024-03-15/he --format commonvoice
  
  # Custom dataset (folder with audio + metadata.csv)
  python prepare_hebrew_dataset.py --input my_hebrew_data --format custom
  
  # With specific output directory
  python prepare_hebrew_dataset.py --input data --format custom --output data/Hebrew_Dataset
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input dataset (Common Voice folder or custom folder)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["commonvoice", "custom"],
        default="custom",
        help="Dataset format: 'commonvoice' for Mozilla CV, 'custom' for wav+metadata"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: data/Hebrew_Dataset in project folder)"
    )
    
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Target sample rate for audio (default: 24000)"
    )
    
    parser.add_argument(
        "--keep-niqqud",
        action="store_true",
        help="Keep niqqud (vowel marks) in text"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without creating output files"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    
    return parser.parse_args()


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get audio duration in seconds."""
    if not TORCHAUDIO_AVAILABLE:
        return None
    
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"[WARNING] Could not read {audio_path}: {e}")
        return None


def validate_audio(audio_path: str, min_duration: float, max_duration: float) -> Tuple[bool, str]:
    """
    Validate audio file.
    
    Returns:
        (is_valid, reason)
    """
    if not os.path.exists(audio_path):
        return False, "File not found"
    
    duration = get_audio_duration(audio_path)
    if duration is None:
        # Can't check duration, assume valid
        return True, "OK (duration unknown)"
    
    if duration < min_duration:
        return False, f"Too short ({duration:.2f}s < {min_duration}s)"
    
    if duration > max_duration:
        return False, f"Too long ({duration:.2f}s > {max_duration}s)"
    
    return True, f"OK ({duration:.2f}s)"


def validate_text(text: str) -> Tuple[bool, str]:
    """
    Validate Hebrew text.
    
    Returns:
        (is_valid, reason)
    """
    if not text or not text.strip():
        return False, "Empty text"
    
    # Check if there's at least one Hebrew letter
    hebrew_count = sum(1 for c in text if is_hebrew_letter(c))
    if hebrew_count == 0:
        return False, "No Hebrew characters"
    
    # Check text length (rough character limit for TTS)
    if len(text) > 500:
        return False, f"Text too long ({len(text)} chars)"
    
    return True, "OK"


def load_commonvoice_data(input_path: str, split: str = "validated") -> List[Tuple[str, str]]:
    """
    Load Mozilla Common Voice dataset.
    
    Common Voice structure:
    - cv-corpus/he/
        - clips/
            - common_voice_he_xxxxx.mp3
        - validated.tsv
        - train.tsv
        - test.tsv
        - dev.tsv
    
    Args:
        input_path: Path to language folder (e.g., cv-corpus/he)
        split: Which split to use ('validated', 'train', 'test', 'dev')
        
    Returns:
        List of (audio_path, text) tuples
    """
    input_path = Path(input_path)
    
    # Find the TSV file
    tsv_path = input_path / f"{split}.tsv"
    if not tsv_path.exists():
        # Try without split
        tsv_path = input_path / "validated.tsv"
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"Could not find TSV file in {input_path}")
    
    clips_dir = input_path / "clips"
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    
    print(f"Loading Common Voice data from {tsv_path}")
    
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            audio_file = clips_dir / row['path']
            text = row['sentence']
            data.append((str(audio_file), text))
    
    print(f"Found {len(data)} samples in Common Voice dataset")
    return data


def load_custom_data(input_path: str) -> List[Tuple[str, str]]:
    """
    Load custom dataset with metadata.csv.
    
    Expected structure:
    - input_path/
        - audio/
            - file1.wav
            - file2.wav
        - metadata.csv (columns: audio_file, text OR filename|text)
    
    Alternatively:
    - input_path/
        - wavs/
            - file1.wav
        - metadata.csv
    
    Or flat structure:
    - input_path/
        - file1.wav
        - file2.wav
        - metadata.csv
    """
    input_path = Path(input_path)
    
    # Find metadata file
    metadata_path = None
    for name in ['metadata.csv', 'metadata.txt', 'transcripts.csv', 'transcripts.txt']:
        p = input_path / name
        if p.exists():
            metadata_path = p
            break
    
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata file found in {input_path}")
    
    # Find audio directory
    audio_dir = input_path
    for name in ['audio', 'wavs', 'clips', 'wav']:
        d = input_path / name
        if d.exists() and d.is_dir():
            audio_dir = d
            break
    
    print(f"Loading custom data from {metadata_path}")
    print(f"Audio directory: {audio_dir}")
    
    data = []
    
    # Try to detect delimiter and format
    with open(metadata_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if '|' in first_line:
            delimiter = '|'
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        
        # Check if first row is header
        first_row = next(reader)
        has_header = 'audio' in first_row[0].lower() or 'file' in first_row[0].lower() or 'text' in first_row[-1].lower()
        
        if not has_header:
            # First row is data, process it
            audio_file, text = first_row[0], first_row[-1]
            audio_path = audio_dir / audio_file
            if not audio_path.suffix:
                audio_path = audio_path.with_suffix('.wav')
            data.append((str(audio_path), text))
        
        for row in reader:
            if len(row) < 2:
                continue
            audio_file, text = row[0], row[-1]
            
            # Handle full path vs filename
            if os.path.isabs(audio_file):
                audio_path = Path(audio_file)
            else:
                audio_path = audio_dir / audio_file
            
            # Add extension if missing
            if not audio_path.suffix:
                for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    test_path = audio_path.with_suffix(ext)
                    if test_path.exists():
                        audio_path = test_path
                        break
            
            data.append((str(audio_path), text))
    
    print(f"Found {len(data)} samples in custom dataset")
    return data


def process_dataset(
    data: List[Tuple[str, str]],
    output_dir: Path,
    min_duration: float,
    max_duration: float,
    sample_rate: int,
    keep_niqqud: bool,
    dry_run: bool,
    limit: Optional[int] = None
) -> Tuple[int, int, List[str]]:
    """
    Process and validate dataset.
    
    Returns:
        (valid_count, invalid_count, all_texts)
    """
    if limit:
        data = data[:limit]
    
    valid_samples = []
    invalid_samples = []
    all_texts = []
    
    stats = defaultdict(int)
    
    print(f"\nProcessing {len(data)} samples...")
    
    for i, (audio_path, text) in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)}...")
        
        # Normalize text
        normalized_text = normalize_hebrew_text(text, keep_niqqud=keep_niqqud)
        
        # Validate text
        text_valid, text_reason = validate_text(normalized_text)
        if not text_valid:
            invalid_samples.append((audio_path, text, f"Text: {text_reason}"))
            stats[f"text_{text_reason}"] += 1
            continue
        
        # Validate audio
        audio_valid, audio_reason = validate_audio(audio_path, min_duration, max_duration)
        if not audio_valid:
            invalid_samples.append((audio_path, text, f"Audio: {audio_reason}"))
            stats[f"audio_{audio_reason}"] += 1
            continue
        
        valid_samples.append((audio_path, normalized_text))
        all_texts.append(normalized_text)
        stats["valid"] += 1
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Valid samples: {len(valid_samples)}")
    print(f"  Invalid samples: {len(invalid_samples)}")
    print(f"\nBreakdown:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")
    
    if dry_run:
        print("\n[DRY RUN] No files written.")
        return len(valid_samples), len(invalid_samples), all_texts
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build and save vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_hebrew_vocab(all_texts)
    vocab_path = output_dir / "vocab.txt"
    save_vocab(vocab, str(vocab_path))
    
    # Write metadata file in F5-TTS format
    metadata_path = output_dir / "metadata.csv"
    print(f"Writing metadata to {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['audio_file', 'text'])  # Header
        for audio_path, text in valid_samples:
            # F5-TTS expects absolute paths
            abs_path = os.path.abspath(audio_path)
            writer.writerow([abs_path, text])
    
    print(f"Wrote {len(valid_samples)} samples to {metadata_path}")
    
    # Optionally write invalid samples for review
    if invalid_samples:
        invalid_path = output_dir / "invalid_samples.csv"
        with open(invalid_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_file', 'text', 'reason'])
            for row in invalid_samples:
                writer.writerow(row)
        print(f"Wrote {len(invalid_samples)} invalid samples to {invalid_path}")
    
    return len(valid_samples), len(invalid_samples), all_texts


def main():
    args = parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: data/Hebrew_Dataset in project folder
        project_dir = Path(__file__).parent.parent
        output_dir = project_dir / "data" / "Hebrew_Dataset"
    
    print("="*60)
    print("F5-TTS Hebrew Dataset Preparation")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Format: {args.format}")
    print(f"Output: {output_dir}")
    print(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Keep niqqud: {args.keep_niqqud}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print("="*60)
    
    # Load data based on format
    if args.format == "commonvoice":
        data = load_commonvoice_data(args.input)
    else:
        data = load_custom_data(args.input)
    
    # Process dataset
    valid, invalid, texts = process_dataset(
        data=data,
        output_dir=output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        sample_rate=args.sample_rate,
        keep_niqqud=args.keep_niqqud,
        dry_run=args.dry_run,
        limit=args.limit
    )
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    
    if not args.dry_run:
        print(f"""
Next steps:
1. Review the vocabulary file: {output_dir / 'vocab.txt'}
2. Check invalid samples (if any): {output_dir / 'invalid_samples.csv'}
3. Run fine-tuning:
   python F5-TTS/src/f5_tts/train/finetune_cli.py \\
       --finetune \\
       --dataset_name Hebrew_Dataset \\
       --tokenizer custom \\
       --tokenizer_path {output_dir / 'vocab.txt'} \\
       --learning_rate 1e-5
        """)
    
    return 0 if valid > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
