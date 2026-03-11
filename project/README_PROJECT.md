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
| Method A — SDEdit noise injection (`run_noise_inject.py`) | Done |
| Method B — Style Guidance 2-pass ODE (`run_style_guidance.py`) | Ready to run |
| Method C — Scheduled Conditioning (`run_scheduled_cond.py`) | Ready to run |
| Method D — Noise Stats Transfer (`run_noise_stats.py`) | Ready to run |
| Cross-method comparison (`compare_methods.py`) | Ready (run after A–D) |
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

### Method A — SDEdit noise injection

```bash
python scripts/run_noise_inject.py \
    --identity path/to/identity_speaker.wav \
    --emotion  path/to/style_speaker.wav \
    --text "The situation has become completely unacceptable." \
    --noise_levels 0.0 0.1 0.2 0.3 0.5 0.7 \
    --sways -1.0 -0.4 0.0
```

Biases the ODE starting point: `x_0 = (1-α)*randn + α*mel_B`, then integrates
conditioned on mel_A. Results go to `results/noise_inject/`.

### Method B — Style Guidance (2-pass ODE extrapolation)

```bash
python scripts/run_style_guidance.py \
    --identity path/to/identity_speaker.wav \
    --emotion  path/to/style_speaker.wav \
    --guidance_scales 0.0 0.5 1.0 1.5 2.0 \
    --sways -1.0 0.0
```

Runs the transformer twice per ODE step (identity + style) and extrapolates:
`vf = vf_A + guidance_scale * (vf_B - vf_A)`. Results go to `results/style_guidance/`.

### Method C — Scheduled Conditioning Blend

```bash
python scripts/run_scheduled_cond.py \
    --identity path/to/identity_speaker.wav \
    --emotion  path/to/style_speaker.wav \
    --switch_points 0.0 0.25 0.5 0.75 1.0 \
    --sways -1.0 0.0
```

Switches conditioning from mel_B to mel_A at a programmable ODE time `t*`:
early steps (t < t*) use style B, late steps use identity A. Results go to
`results/scheduled_cond/`.

### Method D — Noise Statistics Transfer

```bash
python scripts/run_noise_stats.py \
    --identity path/to/identity_speaker.wav \
    --emotion  path/to/style_speaker.wav \
    --noise_levels 0.0 0.1 0.2 0.3 0.5 0.7 \
    --sways -1.0 0.0
```

Rescales starting noise to match mel_B's global statistics (mean, std) without
copying specific frames. Results go to `results/noise_stats/`.

### Cross-method comparison (run after all methods)

```bash
python scripts/compare_methods.py --methods A B C D
```

Reads `results_metrics.csv` from each method, generates comparison plots
(trends, scatter, WER-vs-SIM-B), a combined CSV, and a mathematical chapter.
Results go to `results/comparison/`.

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

### Noise injection sweep (Method A)

```bash
python scripts/run_noise_inject.py   # sweeps alpha × sway (18 total runs)
```

### Extended sweeps (Methods B, C, D)

```bash
python scripts/run_style_guidance.py   # guidance_scale × sway (10 runs)
python scripts/run_scheduled_cond.py   # switch_point × sway (10 runs)
python scripts/run_noise_stats.py      # alpha × sway (12 runs)
python scripts/compare_methods.py      # aggregates all results
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
│   ├── run_noise_inject.py        # Method A — SDEdit noise injection sweep
│   ├── run_style_guidance.py      # Method B — 2-pass ODE style guidance
│   ├── run_scheduled_cond.py      # Method C — Scheduled conditioning blend
│   ├── run_noise_stats.py         # Method D — Noise statistics transfer
│   ├── compare_methods.py         # Cross-method comparison plots + CSV
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
    ├── noise_inject/              # 18 WAVs + results_metrics.csv + plot  (Method A)
    ├── style_guidance/            # 10 WAVs + results_metrics.csv + plot  (Method B)
    ├── scheduled_cond/            # 10 WAVs + results_metrics.csv + plot  (Method C)
    ├── noise_stats/               # 12 WAVs + results_metrics.csv + plot  (Method D)
    ├── comparison/                # combined_metrics.csv + comparison plots
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

### Method A — SDEdit Noise Injection

Instead of blending the conditioning signal, we bias the ODE *starting point*:

```
x_0 = (1 - alpha) * randn_like(mel) + alpha * mel_B
```

The ODE then integrates from `t_start = alpha` to `t = 1`, conditioned throughout on mel_A.

### Method B — Style Guidance (2-Pass ODE Extrapolation)

At each ODE step the transformer is evaluated twice — once under mel_A, once under mel_B — and the results are linearly extrapolated:

```
vf = vf_A + guidance_scale * (vf_B - vf_A)
```

guidance_scale=0 reproduces the identity baseline; guidance_scale>1 extrapolates beyond the style reference. Structurally identical to classifier-free guidance but along the identity→style axis.

### Method C — Scheduled Conditioning Blend

A step function switches the conditioning from mel_B (style) to mel_A (identity) at ODE time `t*`:

```
cond(t) = mel_B  if t < t*   # style drives coarse spectral structure
cond(t) = mel_A  if t ≥ t*   # identity refines fine details
```

Single-pass (same cost as baseline). switch_point=0 → pure identity; switch_point=1 → pure style.

### Method D — Noise Statistics Transfer

A statistics-only variant of Method A: instead of blending mel_B frames into x_0, only the global mean and standard deviation of mel_B are transferred:

```
x_0 = (randn / randn.std()) * target_std + alpha * mel_B.mean()
target_std = alpha * mel_B.std() + (1 - alpha) * randn.std()
```

The temporal structure of mel_B is never copied — only its amplitude envelope.

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
