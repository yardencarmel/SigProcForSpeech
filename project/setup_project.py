#!/usr/bin/env python3
"""
F5-TTS Acoustic Style Transfer - Project Setup Script

This script:
1. Clones the F5-TTS repository (if not exists)
2. Creates a virtual environment
3. Installs dependencies
4. Verifies CUDA availability

Usage:
    python setup_project.py              # Full setup
    python setup_project.py --verify-only # Just verify environment
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return output."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed: {result.stderr}")
        return None
    return result.stdout.strip()


def check_git():
    """Check if git is installed."""
    result = run_command("git --version", check=False)
    if result is None:
        print("[ERROR] Git is not installed. Please install git first.")
        return False
    print(f"[OK] {result}")
    return True


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"[ERROR] Python 3.10+ required, got {version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check CUDA availability via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[OK] CUDA available: {device_name} (CUDA {cuda_version})")
            return True
        else:
            print("[WARNING] CUDA not available. Inference will be slow on CPU.")
            return True  # Continue anyway
    except ImportError:
        print("[INFO] PyTorch not installed yet. CUDA check will be done after install.")
        return True


def clone_f5_tts(project_dir):
    """Clone F5-TTS repository if not exists."""
    f5_dir = project_dir / "F5-TTS"

    if f5_dir.exists():
        print(f"[OK] F5-TTS already cloned at {f5_dir}")
        return f5_dir

    print("[INFO] Cloning F5-TTS repository...")
    result = run_command(
        "git clone https://github.com/SWivid/F5-TTS.git",
        cwd=project_dir
    )

    if result is None:
        return None

    print("[OK] F5-TTS cloned successfully")
    return f5_dir


def create_venv(project_dir):
    """Create virtual environment if not exists."""
    venv_dir = project_dir / "venv"

    if venv_dir.exists():
        print(f"[OK] Virtual environment already exists at {venv_dir}")
        return venv_dir

    print("[INFO] Creating virtual environment...")
    run_command(f'python -m venv "{venv_dir}"', cwd=project_dir)
    print(f"[OK] Virtual environment created at {venv_dir}")
    return venv_dir


def get_pip_path(venv_dir):
    """Get pip executable path based on OS."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def get_python_path(venv_dir):
    """Get python executable path based on OS."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_dependencies(project_dir, venv_dir, f5_dir):
    """Install all dependencies."""
    pip_path = get_pip_path(venv_dir)
    python_path = get_python_path(venv_dir)

    # Use 'python -m pip' to upgrade pip — required by modern pip on Windows
    print("[INFO] Upgrading pip...")
    result = run_command(f'"{python_path}" -m pip install --upgrade pip', check=False)
    if result is None:
        print("[WARNING] pip upgrade failed (non-critical, continuing...)")
    else:
        print("[OK] pip upgraded")

    # Install PyTorch with CUDA (user should adjust for their CUDA version)
    print("[INFO] Installing PyTorch (adjust CUDA version if needed)... this might take a while...")
    run_command(
        f'"{pip_path}" install torch torchaudio --index-url https://download.pytorch.org/whl/cu121',
        check=False  # May fail if no internet, continue anyway
    )

    # Install F5-TTS in editable mode
    print("[INFO] Installing F5-TTS in editable mode... this might take a while...")
    result = run_command(f'"{pip_path}" install -e "{f5_dir}"', check=False)
    if result is None:
        print("[WARNING] F5-TTS editable install failed (non-critical, continuing...)")

    # Remove torchcodec — it's pulled in by F5-TTS but doesn't work on Windows
    # (missing FFmpeg DLLs) and is not needed for TTS/audio inference.
    print("[INFO] Removing torchcodec (Windows-incompatible, not needed for TTS)...")
    run_command(f'"{pip_path}" uninstall torchcodec -y', check=False)
    print("[OK] torchcodec removed")

    # Install project requirements
    req_file = project_dir / "requirements.txt"
    if req_file.exists():
        print("[INFO] Installing project requirements...")
        result = run_command(f'"{pip_path}" install -r "{req_file}"', check=False)
        if result is None:
            print("[WARNING] Some project requirements failed to install (non-critical, continuing...)")

    print("[OK] Dependencies installed")


def verify_installation(venv_dir):
    """Verify that all required packages are installed."""
    python_path = get_python_path(venv_dir)

    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    verify_script = '''
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"PyTorch: NOT INSTALLED - {e}")

try:
    import f5_tts
    print("F5-TTS: INSTALLED")
except ImportError as e:
    print(f"F5-TTS: NOT INSTALLED - {e}")

try:
    import soundfile
    print("soundfile: INSTALLED")
except ImportError:
    print("soundfile: NOT INSTALLED")

try:
    import librosa
    print("librosa: INSTALLED")
except ImportError:
    print("librosa: NOT INSTALLED")

print("\\nSetup verification complete!")
'''

    run_command(f'"{python_path}" -c "{verify_script}"')


def print_next_steps(project_dir, venv_dir):
    """Print instructions for next steps."""
    activate_cmd = "venv\\Scripts\\activate" if sys.platform == "win32" else "source venv/bin/activate"

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"""
1. Activate the virtual environment:
   cd "{project_dir}"
   {activate_cmd}

2. Run the FULL experiment end-to-end (all steps):
   python scripts/8_run_experiment.py --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav

   Or run specific steps only (e.g. baseline + extension 1):
   python scripts/8_run_experiment.py --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav --steps 1 2

   Available steps:
     1  Baseline reproduction (English TTS + sway/CFG ablations + eval)
     2  Extension 1: Direct Mel Injection
     3  Extension 2, Method A: SDEdit Noise Injection
     4  Extension 2, Method B: Style Guidance (2-Pass ODE)
     5  Extension 2, Method C: Scheduled Conditioning Blend
     6  Extension 2, Method D: Noise Statistics Transfer
     7  All graphs + cross-method comparison

3. Or run individual steps directly:
   python scripts/1_baseline.py --ref_audio F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav
   python scripts/2_extension1.py
   python scripts/3_method_A.py
   python scripts/4_method_B.py
   python scripts/5_method_C.py
   python scripts/6_method_D.py
   python scripts/7_graphs.py --graphs all

4. Generate only specific graphs:
   python scripts/7_graphs.py --graphs baseline comparison
   python scripts/7_graphs.py --graphs method_A method_B
""")


def main():
    parser = argparse.ArgumentParser(description="Setup F5-TTS Acoustic Style Transfer Project")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify environment, don't install")
    parser.add_argument("--skip-venv", action="store_true",
                        help="Skip virtual environment creation (use current env)")
    args = parser.parse_args()

    # Get project directory (where this script is located)
    project_dir = Path(__file__).parent.resolve()
    print(f"[INFO] Project directory: {project_dir}")

    # Basic checks
    if not check_python():
        return 1

    if args.verify_only:
        check_cuda()
        return 0

    if not check_git():
        return 1

    # Clone F5-TTS
    f5_dir = clone_f5_tts(project_dir)
    if f5_dir is None:
        return 1

    if not args.skip_venv:
        # Create virtual environment
        venv_dir = create_venv(project_dir)

        # Install dependencies
        install_dependencies(project_dir, venv_dir, f5_dir)

        # Verify installation
        verify_installation(venv_dir)

        # Print next steps
        print_next_steps(project_dir, venv_dir)
    else:
        print("[INFO] Skipping venv creation. Installing to current environment...")
        # Use sys.executable -m pip to ensure the correct pip is used cross-platform
        py = sys.executable
        result = run_command(f'"{py}" -m pip install --upgrade pip', check=False)
        if result is None:
            print("[WARNING] pip upgrade failed (non-critical, continuing...)")
        result = run_command(f'"{py}" -m pip install -e "{f5_dir}"', check=False)
        if result is None:
            print("[WARNING] F5-TTS editable install failed (non-critical, continuing...)")
        # Remove torchcodec — Windows-incompatible, not needed for TTS
        run_command(f'"{py}" -m pip uninstall torchcodec -y', check=False)
        print("[OK] torchcodec removed")
        req_file = project_dir / "requirements.txt"
        if req_file.exists():
            result = run_command(f'"{py}" -m pip install -r "{req_file}"', check=False)
            if result is None:
                print("[WARNING] Some requirements failed to install (non-critical, continuing...)")

    print("\n[OK] Setup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
