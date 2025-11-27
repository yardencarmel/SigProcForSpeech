# HW1 – Signal Processing for Speech

This assignment inspects a reference LibriSpeech utterance (Q1) and compares gender classification pipelines built from MFCC means and YIN-based pitch statistics (Q2). All logic lives in `hw1.py`.

## Environment
- Python 3.12 (matches the provided `hw1venv` virtual environment)
- Core dependencies: `librosa`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`

### Quick setup
```bash
cd path-to/HW1/hw1.py
python3 -m venv hw1venv              # skip if already created
source hw1venv/bin/activate
pip install -r requirements.txt     # or pip install librosa matplotlib scikit-learn scipy
```

## Audio assets
`hw1.py` expects the LibriSpeech subsets under `HW1/assets`:
```
HW1/
 ├─ hw1.py
 └─ assets/
     ├─ dev-clean/LibriSpeech/dev-clean/<speaker>/<chapter>/<utterance>.flac
     └─ test-clean/LibriSpeech/test-clean/<speaker>/<chapter>/<utterance>.flac
```
- Place the official `dev-clean` and `test-clean` folders exactly as shown.
- Keep the accompanying `SPEAKERS.TXT` alongside them under `assets` so `parse_speakers_txt` can resolve genders.
- If you need a different data root, edit the `ROOT` constant in `hw1.py`.

## Running the scripts
```bash
cd path-to/HW1/hw1.py
source hw1venv/bin/activate
python hw1.py
```
- Q1 loads `84-121123-0000.flac`, prints summary stats, and opens waveform/STFT/MFCC figures.
- Q2 builds train/test splits (64/16 speakers), extracts MFCC-mean and pitch-stat features, trains RBF-SVMs, and prints accuracy + confusion matrices.

### Tips
- Figures pop up via Matplotlib; use a backend that supports interactive windows .
- Ensure the terminal inherits the `hw1.py` directory so relative `Path` calls resolve correctly.

## Verification checklist
- ✅ `SPEAKERS.TXT` reachable at `HW1/assets/SPEAKERS.TXT`
- ✅ LibriSpeech audio under `assets/dev-clean/LibriSpeech/dev-clean` and `assets/test-clean/LibriSpeech/test-clean`
- ✅ Virtual environment (or system Python) has the listed dependencies
- ✅ `python hw1.py` executes without missing-file errors

