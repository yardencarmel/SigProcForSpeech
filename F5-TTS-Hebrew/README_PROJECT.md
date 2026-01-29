# F5-TTS Hebrew Extension

**Academic Project: Hebrew Language Adaptation with Cross-Lingual Emotion Transfer**

This project extends the [F5-TTS](https://github.com/SWivid/F5-TTS) text-to-speech system to support Hebrew language synthesis with experimental emotion transfer capabilities.

## Features

- 🇮🇱 **Hebrew TTS**: Generate natural Hebrew speech using F5-TTS
- 🎤 **Voice Cloning**: Zero-shot voice cloning from a reference audio
- 🎭 **Emotion Transfer** (Experimental): Guide generation style using a second reference
- 📊 **Data Preparation**: Scripts for Common Voice Hebrew and custom datasets

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python setup_project.py

# Or manually:
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements_hebrew.txt
```

### 2. Download Hebrew Dataset

The fastest way to get started is using the automated download script:

```bash
# 1. Get your API key from Mozilla Data Collective
#    - Go to https://datacollective.mozillafoundation.org
#    - Create account, login, and generate an API key
#    - IMPORTANT: Agree to the Hebrew dataset's Terms of Use

# 2. Add your API key to .env file
#    Open .env and replace 'your_api_key_here' with your actual key

# 3. Download the dataset
python scripts/download_dataset.py --extract
```

**Alternative**: If you already have Common Voice downloaded:
```bash
python scripts/prepare_hebrew_dataset.py \
    --input path/to/cv-corpus-24.0-he \
    --format commonvoice
```

**Custom dataset format:**
```
my_data/
├── audio/
│   ├── file1.wav
│   └── file2.wav
└── metadata.csv  # Format: audio_file|text
```

### 3. Fine-tune the Model

```bash
# Using the CLI script
python F5-TTS/src/f5_tts/train/finetune_cli.py \
    --finetune \
    --dataset_name Hebrew_Dataset \
    --tokenizer custom \
    --tokenizer_path data/Hebrew_Dataset/vocab.txt \
    --learning_rate 1e-5 \
    --epochs 50 \
    --batch_size_per_gpu 3200 \
    --save_per_updates 10000 \
    --logger tensorboard
```

### 4. Generate Hebrew Speech

```bash
# Basic generation
python scripts/run_emotion_transfer.py \
    --text "שלום עולם, מה שלומך היום?" \
    --ref_audio samples/speaker.wav \
    --output output.wav

# With emotion transfer (experimental)
python scripts/run_emotion_transfer.py \
    --text "אני כל כך שמח לראות אותך!" \
    --ref_audio_identity samples/neutral_speaker.wav \
    --ref_audio_emotion samples/happy_voice.wav \
    --emotion_weight 0.3 \
    --output happy_output.wav
```

## Project Structure

```
F5-TTS-Hebrew/
├── .env                       # API keys (create from template, don't commit!)
├── .gitignore                 # Git ignore patterns
├── configs/
│   └── train_hebrew.yaml      # Training configuration
├── data/
│   ├── raw/                   # Downloaded dataset archives
│   └── Hebrew_Dataset/        # Prepared dataset (after running prep)
│       ├── vocab.txt          # Hebrew vocabulary
│       └── metadata.csv       # Audio paths and transcripts
├── scripts/
│   ├── download_dataset.py    # Download Common Voice Hebrew via API
│   ├── hebrew_utils.py        # Hebrew text processing utilities
│   ├── prepare_hebrew_dataset.py  # Dataset preparation
│   └── run_emotion_transfer.py    # Inference with emotion
├── F5-TTS/                    # Cloned F5-TTS repository
├── venv/                      # Virtual environment (created by setup)
├── requirements_hebrew.txt    # Python dependencies
├── setup_project.py           # Automated setup
└── README_PROJECT.md          # This file
```

## Technical Details

### Tokenizer Approach

F5-TTS uses character-based tokenization. For Hebrew:
- We use the `custom` tokenizer mode with a `vocab.txt` file
- Hebrew characters (U+05D0 to U+05EA) are included as individual tokens
- Niqqud (vowel marks) are removed during text normalization

### Emotion Transfer Implementation

The emotion transfer is experimental and works by:
1. Loading two reference audios: **identity** (voice) and **emotion** (style)
2. Blending their acoustic features with a configurable weight
3. Using the blended features to condition generation

**Limitations:**
- F5-TTS couples voice identity and emotional expression
- True cross-lingual transfer (English emotion → Hebrew output) may be limited
- Best results with emotional Hebrew references

### Model Architecture

No changes to the core F5-TTS architecture. The extension consists of:
- Custom vocabulary for Hebrew characters
- Text normalization pipeline for Hebrew
- Wrapper scripts for inference with multiple references

## Training Tips

1. **Start with fine-tuning**, not training from scratch
2. Use **learning rate 1e-5** (10x lower than pretraining)
3. Short **warmup period** (1000 steps vs 20000)
4. Minimum **10-15 hours** of Hebrew audio recommended
5. Set `use_ema=False` for early checkpoints (see F5-TTS docs)

## Troubleshooting

### "Hebrew vocab not found"
Run `prepare_hebrew_dataset.py` first to generate `data/Hebrew_Dataset/vocab.txt`

### CUDA out of memory
- Reduce `batch_size_per_gpu` in config (try 1600 or 800)
- Enable `checkpoint_activations: True`
- Use `bnb_optimizer: True` for 8-bit Adam

### Audio sounds robotic/distorted
- Ensure reference audio is clean (no background noise)
- Try increasing `nfe_steps` (32 or 64)
- Adjust `cfg_strength` (try 1.5-3.0)

## Resources

- [F5-TTS Paper](https://arxiv.org/abs/2410.06885)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [Mozilla Common Voice Hebrew](https://commonvoice.mozilla.org/he)

## Citation

If you use this work, please cite:

```bibtex
@article{chen2024f5tts,
  title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author={Chen, Yushen and others},
  journal={arXiv preprint arXiv:2410.06885},
  year={2024}
}
```

## License

This project follows the F5-TTS license (CC-BY-NC-4.0 for model weights).
