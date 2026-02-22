# F5-TTS: Acoustic Style Transfer via Mel Injection & Noise-Biased Generation

**Signal Processing & Learning Methods for Speech — Final Project**

This project extends [F5-TTS](https://github.com/SWivid/F5-TTS) (Chen et al., ACL 2025) in two directions motivated by limitations the paper itself acknowledges:

1. **Improvement 1 — Acoustic Style Transfer via Direct Mel Injection**: Blend two reference speaker mel spectrograms at inference time and inject the blend directly into the model's conditioning pathway. No retraining, no extra parameters.

2. **Improvement 2 — Noise-Injection Style Transfer (SDEdit on Flow Matching)**: Bias the ODE starting point by mixing Gaussian noise with a style reference mel, while keeping the identity conditioning pure. Inspired by SDEdit's noise-level inversion principle.

---

## Quick Start (fresh clone)

> Tested on Linux with Python 3.10+. A GPU is strongly recommended for inference.

### 1. Clone the repo

```bash
git clone https://github.com/yardencarmel/SigProcForSpeech.git
cd SigProcForSpeech/project
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch — pick the right CUDA version for your GPU from https://pytorch.org
# Example for CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

> **Linux system dependency for `phonemizer`** (used by some TTS utilities):
> ```bash
> sudo apt-get install espeak-ng
> ```

### 3. Clone and install F5-TTS

```bash
git clone https://github.com/SWivid/F5-TTS.git
pip install -e F5-TTS/
```

### 4. Verify the installation

```bash
python -c "
from f5_tts.infer.utils_infer import load_model
from f5_tts.model import DiT
import torch
print('f5_tts: OK')
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
"
```

### 5. Get a reference audio

You need a short (4–10 s), clean WAV recording of a single speaker for voice cloning. The F5-TTS canonical examples (`F5-TTS/src/f5_tts/infer/examples/basic/`) work well and are used as defaults in all scripts.

### 6. Run the English baseline

```bash
python scripts/run_all_phases.py \
    --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
    --phases 1
```

This generates 5 English sentences and saves them to `results/phase1/english/`.

### 7. (Optional) Install evaluation dependencies

```bash
# Whisper for WER — downloads ~3 GB model on first run
pip install openai-whisper

# WavLM for speaker similarity — downloads ~1.3 GB on first run
# (installed automatically via transformers, which is already in requirements.txt)
```

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
| English baseline (Phase 1) | Done |
| Sway Sampling ablation (Phase 2) | Done |
| CFG strength ablation (Phase 2) | Done |
| Style transfer — mel injection (`run_phase4.py`) | Done |
| Noise injection sweep (`run_noise_inject.py`) | Done |
| Metrics — WER / SIM-A / SIM-B / MCD | Done |
| Plots | Done |

---

## Usage

### English zero-shot baseline

```bash
python scripts/run_all_phases.py \
    --ref_audio path/to/reference.wav \
    --phases 1
```

Generates 5 English samples to confirm the pipeline works.

### Sway Sampling + CFG ablations (reproduce paper results)

```bash
python scripts/run_all_phases.py \
    --ref_audio path/to/reference.wav \
    --phases 2
```

### Style transfer — direct mel injection

```bash
python scripts/run_phase4.py \
    --identity path/to/identity_speaker.wav \
    --emotion  path/to/style_speaker.wav \
    --text "The situation has become completely unacceptable."
```

`--identity` sets the voice; `--emotion` sets the style. The `--weight` parameter (0–1) controls the blend.

### Noise-injection style transfer

```bash
python scripts/run_noise_inject.py \
    --identity path/to/identity_speaker.wav \
    --emotion  path/to/style_speaker.wav \
    --text "The situation has become completely unacceptable." \
    --noise_levels 0.0 0.1 0.2 0.3 0.5 0.7 \
    --sways -1.0 -0.4 0.0
```

Sweeps over noise levels (alpha) and sway coefficients. Results go to `results/noise_inject/`.

---

## Ablations

### Sway Sampling (reproduces Figure 3 of the paper)

```bash
python scripts/run_all_phases.py \
    --ref_audio path/to/ref.wav \
    --phases 2 5
```

Sweeps `s ∈ {0.4, 0.0, -0.4, -0.8, -1.0}` × `NFE ∈ {8, 16, 32}`.

### CFG strength

Runs as part of Phase 2 (above). Sweeps `cfg ∈ {1.0, 1.5, 2.0, 2.5, 3.0}`.

### Style transfer weight (direct mel injection)

```bash
python scripts/run_phase4.py   # sweeps w in {0.0, 0.25, 0.50, 0.75, 1.00}
```

### Noise injection sweep

```bash
python scripts/run_noise_inject.py   # sweeps alpha × sway (18 total runs)
```

---

## Evaluation Metrics

| Metric | Tool | What it measures |
|---|---|---|
| WER | Whisper large-v3-turbo | Faithfulness to text (lower = better) |
| SIM-o / SIM-A | WavLM-base-plus cosine similarity | Speaker identity preservation vs reference A (higher = better) |
| SIM-B | WavLM-base-plus cosine similarity | Style acquisition vs reference B (higher = more style transferred) |
| MCD | MFCC-based cepstral distortion | Acoustic distance from reference (lower = closer) |

---

## Results

### Sway Sampling ablation

| s | NFE | WER (%) | SIM-o |
|---|---|---|---|
| +0.4 | 32 | 51.9 | 0.826 |
| 0.0 | 32 | 3.7 | 0.836 |
| −0.4 | 32 | 7.9 | 0.826 |
| −0.8 | 32 | 0.0 | 0.784 |
| −1.0 | 32 | 7.4 | 0.802 |

### Style transfer — direct mel injection

| weight | WER (%) | SIM-o | MCD |
|---|---|---|---|
| 0.00 | 0.0 | 0.893 | 726 |
| 0.25 | 0.0 | 0.896 | 707 |
| 0.50 | 15.0 | 0.833 | 870 |
| 0.75 | 0.0 | 0.812 | 832 |
| 1.00 | 0.0 | 0.811 | 919 |

### Noise injection sweep

See `results/noise_inject/results_metrics.csv` and `results/noise_inject/noise_inject_sweep.png`.

---

## Project Structure

```
project/
├── README_PROJECT.md              # This file
├── requirements.txt               # Python dependencies
├── setup_project.py               # One-time environment setup helper (legacy)
├── scripts/
│   ├── run_all_phases.py          # Phases 1+2+5: baseline, ablations, eval
│   ├── run_phase4.py              # Style transfer: direct mel injection
│   ├── run_noise_inject.py        # Noise-injection style transfer sweep
│   ├── compute_noise_inject_metrics.py  # Metrics for pre-generated audio
│   └── finalize_noise_inject.py   # Parse log → CSV + plot
├── F5-TTS/                        # Cloned F5-TTS repo (gitignored)
├── .venv/                         # Virtual environment (gitignored)
├── data/                          # Dataset files (gitignored)
├── ckpts/                         # Model checkpoints (gitignored)
└── results/                       # Generated audio + metrics (gitignored)
    ├── phase1/english/            # 5 English zero-shot WAVs
    ├── sway_sampling/             # 45 WAVs + results_metrics.csv
    ├── cfg_strength/              # 15 WAVs + results_metrics.csv
    ├── emotion_transfer/          # 6 WAVs + results_metrics.csv + plot
    ├── noise_inject/              # 18 WAVs + results_metrics.csv + plot
    └── plots/                     # sway_sampling, cfg_strength, emotion_weight, sway_pdf
```

---

## Technical Details

### Improvement 1 — Direct Mel Injection

Reading `cfm.py` reveals a branch: if the conditioning tensor is 3D `[B, T_mel, C]`, it is used directly without going through the internal mel-spec conversion. This allows injecting an arbitrary (blended) mel tensor as the conditioning signal:

```python
mel_blend = (1 - w) * mel_A + w * mel_B   # [1, T, 100]
generated, _ = model.sample(cond=mel_blend, ...)   # 3D → used directly
```

Using `model.mel_spec` to compute the mels ensures the conditioning tensor is in exactly the representation the model was trained on (same normalisation, hop length, etc.), keeping the blend in-distribution for the Transformer.

### Improvement 2 — Noise-Injection Style Transfer (SDEdit)

Instead of blending the conditioning signal, we bias the ODE *starting point*:

```
x_0 = (1 - alpha) * randn_like(mel) + alpha * mel_B
```

The ODE then integrates from `t_start = alpha` to `t = 1`, conditioned throughout on the identity mel A. This is analogous to SDEdit's "noise-level inversion": higher alpha means more of B's structure survives into the output, trading off identity preservation for style injection. Unlike conditioning injection, the identity reference (mel A) is never blended — it drives the model's attention entirely. The interplay with sway sampling (which concentrates ODE steps near `t = 0`) is a key experimental variable.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'f5_tts'`**
Run `pip install -e F5-TTS/` with the venv activated.

**CUDA out of memory during inference**
Reduce `--nfe` steps or use `--device cpu`.

**Audio sounds robotic / repetitive**
- Use a cleaner reference audio (no background noise, 4–10 s)
- Increase `--nfe` to 32 or 64
- Try `--sway -1.0` (paper's best setting)

---

## References

- Chen et al. (2025). *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.* ACL 2025. [arXiv:2410.06885](https://arxiv.org/abs/2410.06885)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
