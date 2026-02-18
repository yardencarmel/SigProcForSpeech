"""
run_finetune.py — Hebrew F5-TTS fine-tuning launcher.

What this script does:
  1. Monkey-patches torchaudio.load with soundfile (torchaudio 2.10 needs torchcodec DLLs
     which require a shared-library FFmpeg build; we have a static-exe build only).
  2. Converts data/Hebrew_Dataset/metadata.csv -> Arrow dataset that finetune_cli.py expects.
  3. Launches F5-TTS fine-tuning on the Hebrew Common Voice data.

Expected files:
  data/Hebrew_Dataset/metadata.csv   -- pipe-delimited: audio_file|text
  data/Hebrew_Dataset/vocab.txt      -- one token per line
  data/cv-clips-wav/*.wav            -- 24 kHz WAV clips (converted from MP3 earlier)

Output:
  F5-TTS/data/Hebrew_Dataset_custom/ -- Arrow dataset consumed by finetune_cli.py
  ckpts/Hebrew_Dataset/              -- training checkpoints
  results/finetune_log.txt           -- stdout/stderr log
"""

import os
import sys

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Add FFmpeg static exe to PATH so subprocesses can find it
_FFMPEG_BIN = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    r"Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin",
)
if os.path.isdir(_FFMPEG_BIN) and _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

# ---------------------------------------------------------------------------
# 1. Monkey-patch torchaudio.load -> soundfile (avoids torchcodec DLL issue)
# ---------------------------------------------------------------------------
import soundfile as _sf
import torch as _torch
import torchaudio as _torchaudio


def _sf_load(
    path,
    frame_offset=0,
    num_frames=-1,
    normalize=True,
    channels_first=True,
    format=None,
    backend=None,
    buffer_size=4096,
):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    tensor = _torch.from_numpy(data.T.copy())
    if frame_offset:
        tensor = tensor[:, frame_offset:]
    if num_frames > 0:
        tensor = tensor[:, :num_frames]
    return tensor, sr


_torchaudio.load = _sf_load

# ---------------------------------------------------------------------------
# 2. Prepare Arrow dataset
# ---------------------------------------------------------------------------
import csv
import json
from importlib.resources import files as pkg_files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def prepare_dataset():
    metadata_csv = Path(PROJECT_ROOT) / "data" / "Hebrew_Dataset" / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    # Output dir: where finetune_cli.py's load_dataset will look
    f5_data_root = Path(str(pkg_files("f5_tts").joinpath("../../data")))
    out_dir = f5_data_root / "Hebrew_Dataset_custom"
    raw_arrow_path = out_dir / "raw.arrow"
    dur_json_path = out_dir / "duration.json"

    if raw_arrow_path.exists() and dur_json_path.exists():
        print(f"[dataset] Arrow dataset already exists at {out_dir}, skipping preparation.")
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] Preparing Hebrew dataset -> {out_dir}")

    rows = []
    with open(metadata_csv, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        header = next(reader)  # audio_file|text
        for line in reader:
            if len(line) < 2:
                continue
            rows.append((line[0].strip(), line[1].strip()))

    print(f"[dataset] Read {len(rows)} rows from metadata.csv")

    results = []
    durations = []
    skipped = 0

    for audio_path, text in tqdm(rows, desc="Computing durations"):
        if not Path(audio_path).exists():
            skipped += 1
            continue
        try:
            info = sf.info(audio_path)
            dur = info.duration
        except Exception:
            skipped += 1
            continue
        if dur < 0.3 or dur > 30.0:
            skipped += 1
            continue
        results.append({"audio_path": audio_path, "text": text, "duration": dur})
        durations.append(dur)

    print(f"[dataset] Valid samples: {len(results)}  |  Skipped: {skipped}")
    total_hrs = sum(durations) / 3600
    print(f"[dataset] Total audio: {total_hrs:.2f} h")

    # Write Arrow
    with ArrowWriter(path=str(raw_arrow_path)) as writer:
        for rec in tqdm(results, desc="Writing raw.arrow"):
            writer.write(rec)
        writer.finalize()

    # Write duration.json
    with open(str(dur_json_path), "w", encoding="utf-8") as fh:
        json.dump({"duration": durations}, fh, ensure_ascii=False)

    print(f"[dataset] Dataset saved to {out_dir}")
    return str(out_dir)


# ---------------------------------------------------------------------------
# 3. Launch fine-tuning
# ---------------------------------------------------------------------------
def run_finetune():
    data_dir = prepare_dataset()

    vocab_path = str(Path(PROJECT_ROOT) / "data" / "Hebrew_Dataset" / "vocab.txt")
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"vocab.txt not found: {vocab_path}")

    # Inject args for finetune_cli argparse
    sys.argv = [
        "finetune_cli.py",
        "--finetune",
        "--exp_name", "F5TTS_v1_Base",
        "--dataset_name", "Hebrew_Dataset",
        "--tokenizer", "custom",
        "--tokenizer_path", vocab_path,
        "--learning_rate", "1e-5",
        "--epochs", "50",
        "--batch_size_per_gpu", "3200",
        "--batch_size_type", "frame",
        "--max_samples", "32",
        "--num_warmup_updates", "1000",
        "--save_per_updates", "10000",
        "--last_per_updates", "100",   # save model_last.pt every 100 updates for safe resume
        "--keep_last_n_checkpoints", "3",
        "--max_grad_norm", "1.0",
        # no --logger to avoid wandb/tensorboard deps during test
    ]

    print("\n[finetune] Starting F5-TTS Hebrew fine-tuning...")
    print(f"[finetune] vocab: {vocab_path}")
    print(f"[finetune] data:  {data_dir}")
    print("[finetune] This will take many hours on a single GPU.\n")

    from f5_tts.train.finetune_cli import main as finetune_main
    finetune_main()


if __name__ == "__main__":
    run_finetune()
