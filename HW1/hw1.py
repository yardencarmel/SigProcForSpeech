#from librosa import stft, load, amplitude_to_db, yin
# feature.mfcc, display.waveshow, display.specshow,

import librosa
import matplotlib.pyplot as plt
import numpy as np


def Q1():
    # Part A
    print("loading audio file")
    audio_path = r"HW1/assets/dev-clean/84/121123/84-121123-0000.flac"
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Loaded audio: {audio_path}")
    print(f"Samples: {len(y)}")
    print(f"Sample rate: {sr}")
    print(f"length of the audio file (samples/sample rate): {len(y)/sr} seconds")
    print(f"max amp.: {y.max():.6f} [FS]")
    print(f"min amp.: {y.min():.6f} [FS]")
    
    # Part B
    plt.figure(figsize=(12, 3))
    #librosa.display.waveshow(y, sr=sr, x_axis="time")
    plt.title("Waveform")
    plt.tight_layout()
    plt.xlabel("time [s]")
    plt.ylabel("amp. [normalized dimensionless]")
    plt.show()

    # Part C
    plt.figure(figsize=(12, 3))
    win_length = int(0.050 * sr)
    hop_length = int(0.003 * sr)
    n_fft = 1 << (win_length - 1).bit_length()   # smallest power-of-two >=e win_length
    y_stft = librosa.stft(y,win_length=win_length,hop_length=hop_length, n_fft=n_fft)
    y_stft_abs = np.abs(y_stft)
    y_db = librosa.amplitude_to_db(y_stft_abs, ref=np.max)
    img = librosa.display.specshow(y_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
    plt.colorbar(img, format="%+2.0f dB", label="dB")
    plt.show()

    # Part D
    plt.figure(figsize=(12, 3))
    mfcc = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=13,
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=40, fmin=0, fmax=sr/2)

    img = librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time")
    plt.xlabel("time [s]")
    plt.ylabel("MFCC index (C0...C12)")
    plt.yticks(np.arange(13), [f"C{i}" for i in range(13)])
    plt.title("MFCCs (13 coefficients)")
    plt.colorbar(img, label="Coefficient value")
    plt.show()

def Q2():
    pass

import os, glob, random, numpy as np
import librosa
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------- paths (edit ROOT if needed) ----------
ROOT = "HW1/assets"  # folder that contains dev-clean, dev-test and SPEAKERS.TXT
SUBSETS = ["dev-clean", "dev-test"]
SPEAKERS_TXT = os.path.join(ROOT, "SPEAKERS.TXT")
RNG_SEED = 42
random.seed(RNG_SEED); np.random.seed(RNG_SEED)

# --------- helpers ----------
def parse_speakers_txt(path):
    """Return dict: {speaker_id(int): 'M' or 'F'}"""
    gender = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # LibriSpeech lines are often pipe-separated: "ID | SEX | SUBSET | ..."
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                try:
                    spk = int(parts[0]); sex = parts[1]
                except Exception:
                    continue
            else:
                parts = line.split()
                if len(parts) < 2: continue
                try:
                    spk = int(parts[0]); sex = parts[1]
                except Exception:
                    continue
            if sex in ("M", "F"):
                gender[spk] = sex
    return gender

def find_speakers_in_subsets(root, subsets):
    """Return sorted list of speaker IDs present under the given subsets."""
    speakers = set()
    for sub in subsets:
        subdir = os.path.join(root, sub)
        if not os.path.isdir(subdir): continue
        for name in os.listdir(subdir):
            if name.isdigit() and os.path.isdir(os.path.join(subdir, name)):
                speakers.add(int(name))
    return sorted(speakers)

def list_utterances_for_speaker(root, subsets, spk_id):
    """All .flac utterance paths for this speaker across subsets."""
    files = []
    for sub in subsets:
        pattern = os.path.join(root, sub, str(spk_id), "*", "*.flac")
        files.extend(glob.glob(pattern))
    return sorted(files)

# --------- feature extractors ----------
def mfcc_mean(y, sr, n_mfcc=13, win_ms=50, hop_ms=3):
    win_length = max(1, int(sr * win_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    n_fft = 1 << (win_length - 1).bit_length()  # next power of two ≥ win_length
    M = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=40, fmin=0, fmax=sr/2
    )
    return M.mean(axis=1)  # (n_mfcc,)

def pitch_stats(y, sr, win_ms=50, hop_ms=3, fmin=50.0, fmax=400.0):
    frame_length = max(2, int(sr * win_ms / 1000.0))
    hop_length   = max(1, int(sr * hop_ms / 1000.0))
    f0 = librosa.yin(
        y=y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )  # Hz, shape (frames,)
    # ignore NaNs (unvoiced); if all NaN, return zeros
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        return np.zeros(5, dtype=float)
    return np.array([
        float(np.min(f0)),
        float(np.max(f0)),
        float(np.mean(f0)),
        float(np.median(f0)),
        float(np.var(f0)),
    ], dtype=float)

def extract_feature_for_file(path, kind):
    y, sr = librosa.load(path, sr=None)  # keep native sr
    if kind == "mfcc":
        return mfcc_mean(y, sr)
    elif kind == "pitch":
        return pitch_stats(y, sr)
    else:
        raise ValueError("unknown feature kind")

def build_dataset_for_speakers(speaker_ids, gender_map, kind, max_utts_per_speaker=20):
    X, y, meta = [], [], []
    rng = np.random.default_rng(RNG_SEED)
    for spk in speaker_ids:
        files = list_utterances_for_speaker(ROOT, SUBSETS, spk)
        if len(files) == 0:
            continue
        if len(files) > max_utts_per_speaker:
            files = rng.choice(files, size=max_utts_per_speaker, replace=False).tolist()
        for wav in files:
            feat = extract_feature_for_file(wav, kind)
            X.append(feat)
            y.append(gender_map[spk])
            meta.append((spk, wav))
    return np.vstack(X), np.array(y), meta  # X: (N, d)

# --------- main ----------
def main():
    gender_map = parse_speakers_txt(SPEAKERS_TXT)
    speakers = [s for s in find_speakers_in_subsets(ROOT, SUBSETS) if s in gender_map]

    # There should be 80 speakers (40+40); proceed with those we found:
    genders = np.array([gender_map[s] for s in speakers])

    # Stratified speaker split: 66 train / 16 test (≈20%)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=16/80, random_state=RNG_SEED)
    train_idx, test_idx = next(splitter.split(np.array(speakers), genders))
    spk_train = [speakers[i] for i in train_idx]
    spk_test  = [speakers[i] for i in test_idx]

    print(f"Speakers: total={len(speakers)} | train={len(spk_train)} | test={len(spk_test)}")
    print("Train gender counts:", {g: list(np.array([gender_map[s] for s in spk_train])).count(g) for g in ("M","F")})
    print("Test  gender counts:", {g: list(np.array([gender_map[s] for s in spk_test ])).count(g) for g in ("M","F")})

    # ---- Build datasets (utterance-level samples; labels are speaker gender) ----
    print("\nExtracting MFCC-mean features…")
    Xtr_mfcc, ytr, _ = build_dataset_for_speakers(spk_train, gender_map, "mfcc", max_utts_per_speaker=20)
    Xte_mfcc, yte, _ = build_dataset_for_speakers(spk_test , gender_map, "mfcc", max_utts_per_speaker=20)

    print("Extracting YIN pitch-stat features…")
    Xtr_pitch, _, _ = build_dataset_for_speakers(spk_train, gender_map, "pitch", max_utts_per_speaker=20)
    Xte_pitch, _, _ = build_dataset_for_speakers(spk_test , gender_map, "pitch", max_utts_per_speaker=20)

    # ---- Train & evaluate: MFCC ----
    clf_mfcc = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))
    ])
    clf_mfcc.fit(Xtr_mfcc, ytr)
    yhat_mfcc = clf_mfcc.predict(Xte_mfcc)
    print("\n=== MFCC-mean SVM ===")
    print("Accuracy:", accuracy_score(yte, yhat_mfcc))
    print(confusion_matrix(yte, yhat_mfcc, labels=["M","F"]))
    print(classification_report(yte, yhat_mfcc, digits=3))

    # ---- Train & evaluate: Pitch stats ----
    clf_pitch = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))
    ])
    clf_pitch.fit(Xtr_pitch, ytr)
    yhat_pitch = clf_pitch.predict(Xte_pitch)
    print("\n=== Pitch-stats SVM ===")
    print("Accuracy:", accuracy_score(yte, yhat_pitch))
    print(confusion_matrix(yte, yhat_pitch, labels=["M","F"]))
    print(classification_report(yte, yhat_pitch, digits=3))

    # (Optional) combined features for curiosity:
    # Xtr_combo = np.hstack([Xtr_mfcc, Xtr_pitch])
    # Xte_combo = np.hstack([Xte_mfcc, Xte_pitch])
    # clf_combo = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))])
    # clf_combo.fit(Xtr_combo, ytr)
    # print("\n=== Combined (MFCC+Pitch) SVM ===")
    # print("Accuracy:", accuracy_score(yte, clf_combo.predict(Xte_combo)))

if __name__ == "__main__":
    main()


