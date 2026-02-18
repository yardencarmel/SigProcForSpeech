#!/usr/bin/env python3
"""
F5-TTS Hebrew Extension - Project Setup Script

This script:
1. Clones the F5-TTS repository (if not exists)
2. Creates a 'proj' branch for the Hebrew extension
3. Creates a virtual environment
4. Installs dependencies
5. Verifies CUDA availability

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
            print("[WARNING] CUDA not available. Training will be slow on CPU.")
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


def create_proj_branch(f5_dir):
    """Create and checkout 'proj' branch for Hebrew extension."""
    # Check if branch exists
    result = run_command("git branch --list proj", cwd=f5_dir, check=False)
    
    if result and "proj" in result:
        print("[OK] 'proj' branch already exists")
        run_command("git checkout proj", cwd=f5_dir)
    else:
        print("[INFO] Creating 'proj' branch...")
        run_command("git checkout -b proj", cwd=f5_dir)
        print("[OK] Created and switched to 'proj' branch")


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
    
    print("[INFO] Upgrading pip...")
    run_command(f'"{pip_path}" install --upgrade pip')
    
    # Install PyTorch with CUDA (user should adjust for their CUDA version)
    print("[INFO] Installing PyTorch (adjust CUDA version if needed)...")
    run_command(
        f'"{pip_path}" install torch torchaudio --index-url https://download.pytorch.org/whl/cu121',
        check=False  # May fail if no internet, continue anyway
    )
    
    # Install F5-TTS in editable mode
    print("[INFO] Installing F5-TTS in editable mode...")
    run_command(f'"{pip_path}" install -e "{f5_dir}"')
    
    # Install Hebrew-specific requirements
    req_file = project_dir / "requirements_hebrew.txt"
    if req_file.exists():
        print("[INFO] Installing Hebrew-specific requirements...")
        run_command(f'"{pip_path}" install -r "{req_file}"')
    
    print("[OK] Dependencies installed")


def verify_installation(venv_dir):
    """Verify that all required packages are installed."""
    python_path = get_python_path(venv_dir)
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check PyTorch and CUDA
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
    import hebrew_tokenizer
    print("hebrew_tokenizer: INSTALLED")
except ImportError:
    print("hebrew_tokenizer: NOT INSTALLED (optional)")

try:
    from phonemizer import phonemize
    print("phonemizer: INSTALLED")
except ImportError:
    print("phonemizer: NOT INSTALLED (optional)")

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

2. Prepare your Hebrew dataset:
   python scripts/prepare_hebrew_dataset.py --help

3. Fine-tune the model:
   python F5-TTS/src/f5_tts/train/finetune_cli.py \\
       --finetune \\
       --dataset_name Hebrew_Dataset \\
       --tokenizer custom \\
       --tokenizer_path data/Hebrew_Dataset/vocab.txt \\
       --learning_rate 1e-5 \\
       --epochs 50

4. Run inference with emotion transfer:
   python scripts/run_emotion_transfer.py --help
""")


def main():
    parser = argparse.ArgumentParser(description="Setup F5-TTS Hebrew Extension Project")
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
    
    # Create proj branch
    create_proj_branch(f5_dir)
    
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
        # Assume current pip is available
        run_command("pip install --upgrade pip")
        run_command(f'pip install -e "{f5_dir}"')
        req_file = project_dir / "requirements_hebrew.txt"
        if req_file.exists():
            run_command(f'pip install -r "{req_file}"')
    
    print("\n[OK] Setup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
