# F5-TTS: Hebrew Extension with Cross-Lingual Emotion Transfer

**Signal Processing & Learning Methods for Speech — Final Project**

This project extends [F5-TTS](https://github.com/SWivid/F5-TTS) (Chen et al., ACL 2025) in two directions motivated by limitations the paper itself acknowledges:

1. **Improvement 1 — Emotion / Style Transfer**: Separate speaker identity from vocal style using two reference audios. Audio A sets the voice; Audio B sets the emotion. The implementation operates in mel-spectrogram space and supports cross-lingual transfer (e.g. English scream → Hebrew output).

2. **Improvement 2 — Hebrew Language Support**: Fine-tune F5-TTS on Mozilla Common Voice Hebrew, with a custom character-level tokenizer and text normalisation pipeline.

See [`PLAN.txt`](PLAN.txt) for the full task checklist and current status.

---

## Background

F5-TTS is a fully non-autoregressive TTS system based on flow matching with a Diffusion Transformer (DiT). It achieves zero-shot speaker cloning by conditioning generation on a 3–10 second reference audio mel spectrogram, with no explicit phoneme alignment, duration predictor, or text encoder. The paper explicitly lists emotion control as a future-work limitation:

> *"F5-TTS lacks fine-grained control of paralinguistic details, e.g., emotion, which is of great research and practical application value."*

Both improvements address this gap directly.

---

## Current Status

| Component | Status |
|---|---|
| F5-TTS cloned & installed | Done — `pip install -e F5-TTS/` |
| Hebrew text normalisation (`hebrew_utils.py`) | Done |
| Dataset preparation pipeline (`prepare_hebrew_dataset.py`) | Done — untested on real data |
| Common Voice Hebrew download script | Done |
| Fine-tuning config (`configs/train_hebrew.yaml`) | Done |
| Hebrew fine-tuning (actual training run) | **Pending** — needs dataset |
| Emotion transfer — mel-space blending (`run_emotion_transfer.py`) | Done |
| English demo / baseline (`demo_english.py`) | Done — needs GPU run |
| Hebrew pretrained baseline (`demo_hebrew_pretrained.py`) | Done — needs GPU run |
| Sway Sampling ablation (`ablation_sway_sampling.py`) | Done — needs GPU run |
| CFG strength ablation (`ablation_cfg.py`) | Done — needs GPU run |
| Emotion weight ablation (`ablation_emotion_weight.py`) | Done — needs GPU run |
| Evaluation metrics — WER / SIM-o / MCD (`evaluate.py`) | Done — needs audio outputs |
| Results plots (`plot_results.py`) | Done — needs metrics CSV |
| Quantitative results table | **Pending** |

---

## Setup

The `.venv` is already populated with the required packages. Activate it and install the F5-TTS source package:

```bash
source .venv/bin/activate          # Linux / Mac

git clone https://github.com/SWivid/F5-TTS.git
pip install -e F5-TTS/
```

Verify:
```bash
python -c "from f5_tts.infer.utils_infer import load_model; print('OK')"
```

---

## Usage

### English zero-shot baseline

```bash
python scripts/demo_english.py \
    --ref_audio path/to/reference.wav \
    --outdir results/phase1/english
```

Generates 5 English samples to confirm the pipeline works before any Hebrew work.

### Hebrew with pretrained model (expected to fail)

```bash
python scripts/demo_hebrew_pretrained.py \
    --ref_audio path/to/reference.wav \
    --outdir results/phase1/hebrew_pretrained
```

Demonstrates that the pretrained (English + Chinese) model cannot handle Hebrew script, motivating fine-tuning.

### Voice cloning (single reference)

```bash
python scripts/run_emotion_transfer.py \
    --text "שלום, מה שלומך?" \
    --ref_audio samples/speaker.wav \
    --output output.wav
```

### Emotion / style transfer (dual reference)

```bash
python scripts/run_emotion_transfer.py \
    --text "שלום, מה שלומך?" \
    --ref_audio_identity samples/neutral_speaker.wav \
    --ref_audio_emotion  samples/angry_voice.wav \
    --emotion_weight 0.4 \
    --output angry_output.wav
```

`emotion_weight` is a continuous knob: `0.0` = pure identity, `1.0` = pure emotion. Values around `0.3–0.5` preserve recognisable identity while injecting style.

### Cross-lingual emotion transfer (English emotion → Hebrew output)

```bash
python scripts/run_emotion_transfer.py \
    --text "המצב הפך לבלתי נסבל לחלוטין." \
    --ref_audio_identity samples/hebrew_speaker.wav \
    --ref_audio_emotion  samples/english_scream.wav \
    --emotion_weight 0.5 \
    --output crosslingual_output.wav
```

This works because the mel spectrogram is purely acoustic — it carries no linguistic information — so an English emotion reference conditions generation the same way as a Hebrew one would.

---

## Hebrew Fine-tuning

### 1. Download the dataset

```bash
# Option A — automated via Mozilla Data Collective API
#   Register at https://commonvoice.mozilla.org/he, generate API key, add to .env
python scripts/download_dataset.py --extract

# Option B — manual download
#   Download cv-corpus-XX.0-2024-XX-XX-he.tar.gz from Common Voice
#   Extract to data/raw/
```

### 2. Prepare the dataset

```bash
python scripts/prepare_hebrew_dataset.py \
    --input data/raw/cv-corpus-*/he \
    --format commonvoice \
    --output data/Hebrew_Dataset
```

Outputs `data/Hebrew_Dataset/vocab.txt` and `data/Hebrew_Dataset/metadata.csv`.

### 3. Fine-tune

```bash
python F5-TTS/src/f5_tts/train/finetune_cli.py \
    --finetune \
    --dataset_name Hebrew_Dataset \
    --tokenizer custom \
    --tokenizer_path data/Hebrew_Dataset/vocab.txt \
    --learning_rate 1e-5 \
    --epochs 50 \
    --batch_size_per_gpu 3200 \
    --logger tensorboard
```

See [`configs/train_hebrew.yaml`](configs/train_hebrew.yaml) for all hyperparameters.
Minimum recommended data: ~10 hours of Hebrew audio. Common Voice Hebrew has ~70 hours.

### 4. Inference with fine-tuned model

```bash
python scripts/run_emotion_transfer.py \
    --text "ירושלים היא עיר עתיקה ויפה." \
    --ref_audio samples/hebrew_speaker.wav \
    --model_path ckpts/F5TTS_v1_Base_hebrew/model_XXXXX.pt \
    --vocab_path data/Hebrew_Dataset/vocab.txt \
    --output output_hebrew.wav
```

---

## Ablations

### Sway Sampling (reproduces Figure 3 of the paper)

```bash
python scripts/ablation_sway_sampling.py \
    --ref_audio samples/ref_english.wav \
    --outdir results/sway_sampling

python scripts/evaluate.py \
    --csv results/sway_sampling/results.csv \
    --metric all --language en

python scripts/plot_results.py \
    --csv results/sway_sampling/results_metrics.csv \
    --ablation sway --outdir results/plots
```

Sweeps `s ∈ {0.4, 0.0, -0.4, -0.8, -1.0}` × `NFE ∈ {8, 16, 32}`.

### CFG strength

```bash
python scripts/ablation_cfg.py \
    --ref_audio samples/ref_english.wav \
    --outdir results/cfg_strength

python scripts/evaluate.py --csv results/cfg_strength/results.csv --metric all
```

### Emotion weight

```bash
python scripts/ablation_emotion_weight.py \
    --ref_identity samples/neutral_speaker.wav \
    --ref_emotion  samples/angry_voice.wav \
    --text "The situation has become completely unacceptable." \
    --outdir results/emotion_transfer

python scripts/evaluate.py \
    --csv results/emotion_transfer/results.csv \
    --metric all --language en

python scripts/plot_results.py \
    --csv results/emotion_transfer/results_metrics.csv \
    --ablation emotion --outdir results/plots
```

---

## Evaluation Metrics

| Metric | Tool | What it measures |
|---|---|---|
| WER | Whisper large-v3 | Faithfulness to text (lower = better) |
| SIM-o | WavLM-large cosine similarity | Speaker identity preservation (higher = better) |
| MCD | MFCC-based cepstral distortion | Acoustic closeness to emotion reference (lower = more style transferred) |

```bash
# Single file
python scripts/evaluate.py \
    --generated output.wav \
    --reference ref_identity.wav \
    --transcript "the text that was synthesised" \
    --metric all --language he

# Batch (from ablation CSV)
python scripts/evaluate.py --csv results/emotion_transfer/results.csv --metric all
```

---

## Results

*To be filled in after running experiments.*

### Sway Sampling ablation

| s | NFE | WER (%) | SIM-o |
|---|---|---|---|
| +0.4 | 32 | | |
| 0.0 | 32 | | |
| −0.4 | 32 | | |
| −0.8 | 32 | | |
| −1.0 | 32 | | |
| −1.0 | 16 | | |
| −1.0 | 8 | | |

### Emotion transfer ablation

| weight | SIM-o | MCD (dB) | WER (%) |
|---|---|---|---|
| 0.0 (identity only) | | | |
| 0.2 | | | |
| 0.3 | | | |
| 0.5 | | | |
| 0.7 | | | |
| 1.0 (emotion only) | | | |

### Hebrew TTS

| Model | WER (%) | SIM-o | Notes |
|---|---|---|---|
| Pretrained (EN+ZH) | | | Expected: very high WER |
| Fine-tuned (Hebrew CV) | | | |

---

## Project Structure

```
project/
├── PLAN.txt                       # Task checklist with completion status
├── README_PROJECT.md              # This file
├── requirements_hebrew.txt        # Python dependencies
├── setup_project.py               # One-time environment setup helper
├── configs/
│   └── train_hebrew.yaml          # Fine-tuning hyperparameters
├── scripts/
│   ├── hebrew_utils.py            # Text normalisation, vocab building
│   ├── download_dataset.py        # Mozilla Common Voice Hebrew download
│   ├── prepare_hebrew_dataset.py  # CV / custom dataset → F5-TTS format
│   ├── run_emotion_transfer.py    # Inference: voice cloning + emotion transfer
│   ├── demo_english.py            # Phase 1 baseline: English zero-shot
│   ├── demo_hebrew_pretrained.py  # Phase 1 baseline: Hebrew with pretrained model
│   ├── ablation_sway_sampling.py  # Phase 2: Sway Sampling sweep
│   ├── ablation_cfg.py            # Phase 2: CFG strength sweep
│   ├── ablation_emotion_weight.py # Phase 4: emotion_weight sweep
│   ├── evaluate.py                # WER / SIM-o / MCD computation
│   └── plot_results.py            # Publication-quality result plots
├── F5-TTS/                        # Cloned F5-TTS repo (gitignored)
├── .venv/                         # Virtual environment (gitignored)
├── data/                          # Dataset files (gitignored)
├── ckpts/                         # Model checkpoints (gitignored)
└── results/                       # Generated audio + metrics (gitignored)
```

---

## Technical Details

### Emotion Transfer: Mel-Space Blending

Standard F5-TTS conditions generation on a single audio prompt via its mel spectrogram. This project extends that to two prompts:

1. Load `audio_identity` and `audio_emotion`
2. Compute log-mel spectrograms for both using F5-TTS's own mel parameters (100 mel channels, 24 kHz, hop 256, amplitude spectrum, Slaney norm)
3. Interpolate: `mel_cond = (1 − w) × mel_identity + w × mel_emotion`
4. Invert back to a waveform via the vocos vocoder → pass to the standard F5-TTS preprocessing pipeline

This is acoustically coherent. The earlier approach of blending raw waveforms before mel extraction was both incorrect (phase noise) and had a bug where the blend was silently discarded because the original file path was still passed to the model.

Cross-lingual transfer works because the mel spectrogram carries acoustic style but no script-level information.

### Hebrew Tokeniser

F5-TTS uses character-level tokenisation. For Hebrew:
- Characters U+05D0–U+05EA (alef to tav) are each a single token
- Niqqud (diacritical vowel marks, U+0591–U+05C6) are stripped during normalisation — they are redundant for a TTS model that learns pronunciation from audio
- Final-form letters (ך ם ן ף ץ) are kept as distinct tokens
- Punctuation and digits are included for robustness

### Fine-tuning Strategy

- Base model: `F5TTS_v1_Base` (335M parameters, pretrained on English + Chinese)
- Learning rate: 1e-5 (10× lower than pretraining's 7.5e-5)
- Warmup: 1000 steps (vs 20 000 for pretraining)
- The pretrained weights provide strong acoustic priors; only the character embeddings for new Hebrew tokens need to be learned from scratch

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'f5_tts'`**
Run `pip install -e F5-TTS/` with the venv activated.

**CUDA out of memory during fine-tuning**
Reduce `batch_size_per_gpu` in `configs/train_hebrew.yaml` (try 1600 or 800), or enable `checkpoint_activations: True` and `bnb_optimizer: True`.

**Audio sounds robotic / repetitive**
- Use a cleaner reference audio (no background noise, 4–10 s)
- Increase `--nfe_steps` to 32 or 64
- Try `--sway_coef -1.0` (paper's best setting)

**Hebrew vocab not found**
Run `prepare_hebrew_dataset.py` first to generate `data/Hebrew_Dataset/vocab.txt`.

---

## References

- Chen et al. (2025). *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.* ACL 2025. [arXiv:2410.06885](https://arxiv.org/abs/2410.06885)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [Mozilla Common Voice Hebrew](https://commonvoice.mozilla.org/he)
