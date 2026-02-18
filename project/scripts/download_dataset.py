#!/usr/bin/env python3
"""
Download Mozilla Common Voice Hebrew Dataset

This script downloads the Common Voice Hebrew dataset from Mozilla Data Collective
using their API. Requires an API key stored in .env file.

Prerequisites:
1. Create an account at https://datacollective.mozillafoundation.org
2. Agree to the Hebrew dataset's Terms of Use on the website
3. Generate an API key from your profile settings
4. Add the key to .env file as MOZILLA_DC_API_KEY=your_key_here

Usage:
    python download_dataset.py                    # Download Hebrew CV 24.0
    python download_dataset.py --output ./data    # Specify output directory
    python download_dataset.py --extract          # Also extract the archive
"""

import os
import sys
import argparse
import tarfile
from pathlib import Path
from typing import Optional

# Try to load environment variables
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

# Mozilla Data Collective API
API_BASE_URL = "https://datacollective.mozillafoundation.org/api"

# Common Voice Hebrew 24.0 dataset ID
# From: https://datacollective.mozillafoundation.org/datasets/cmj8u3p7800atnxxbzrtfidju
DEFAULT_DATASET_ID = "cmj8u3p7800atnxxbzrtfidju"

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent


# ============================================================================
# Helper Functions
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory (works on any OS)."""
    return PROJECT_DIR


def load_env_file() -> bool:
    """Load environment variables from .env file."""
    env_path = get_project_root() / ".env"
    
    if not env_path.exists():
        print(f"[ERROR] .env file not found at: {env_path}")
        print("\nPlease create .env file with your Mozilla Data Collective API key:")
        print(f"  1. Copy .env.example to .env (or create new .env)")
        print(f"  2. Add: MOZILLA_DC_API_KEY=your_api_key_here")
        return False
    
    if DOTENV_AVAILABLE:
        load_dotenv(env_path)
        return True
    else:
        # Manual parsing if python-dotenv not installed
        print("[INFO] python-dotenv not installed, parsing .env manually")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        return True


def get_api_key() -> Optional[str]:
    """Get the API key from environment."""
    key = os.environ.get("MOZILLA_DC_API_KEY")
    
    if not key or key == "your_api_key_here":
        return None
    
    return key


def validate_dependencies() -> bool:
    """Check that required dependencies are installed."""
    if not REQUESTS_AVAILABLE:
        print("[ERROR] 'requests' package is required. Install with:")
        print("  pip install requests")
        return False
    return True


# ============================================================================
# API Functions
# ============================================================================

def get_dataset_info(api_key: str, dataset_id: str) -> Optional[dict]:
    """
    Get dataset information from Mozilla Data Collective API.
    
    Args:
        api_key: Bearer token for authentication
        dataset_id: Dataset ID to query
        
    Returns:
        Dataset info dict or None if failed
    """
    url = f"{API_BASE_URL}/datasets/{dataset_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 401:
            print("[ERROR] Authentication failed. Your API key may be invalid.")
            print("Please check your MOZILLA_DC_API_KEY in .env file.")
            return None
        elif response.status_code == 403:
            print("[ERROR] Access forbidden. You may need to:")
            print("  1. Agree to the dataset's Terms of Use on the website")
            print("  2. Check that your API key has the correct permissions")
            return None
        elif response.status_code == 404:
            print(f"[ERROR] Dataset not found: {dataset_id}")
            return None
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to get dataset info: {e}")
        return None


def get_download_url(api_key: str, dataset_id: str) -> Optional[str]:
    """
    Get presigned download URL from Mozilla Data Collective API.
    
    Args:
        api_key: Bearer token for authentication
        dataset_id: Dataset ID to download
        
    Returns:
        Presigned download URL or None if failed
    """
    url = f"{API_BASE_URL}/datasets/{dataset_id}/download"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, timeout=30)
        
        if response.status_code == 401:
            print("\n" + "="*60)
            print("[ERROR] AUTHENTICATION FAILED")
            print("="*60)
            print("Your API key was not accepted by the server.")
            print("\nPossible causes:")
            print("  1. The API key is invalid or expired")
            print("  2. The key was not copied correctly")
            print("\nTo fix:")
            print("  1. Go to https://datacollective.mozillafoundation.org")
            print("  2. Log in and go to Profile Settings")
            print("  3. Generate a new API key")
            print("  4. Update MOZILLA_DC_API_KEY in your .env file")
            print("="*60)
            return None
            
        elif response.status_code == 403:
            print("\n" + "="*60)
            print("[ERROR] ACCESS DENIED")
            print("="*60)
            print("Your API key is valid but you don't have access to this dataset.")
            print("\nYou MUST agree to the dataset's Terms of Use first:")
            print(f"  1. Go to https://datacollective.mozillafoundation.org/datasets/{dataset_id}")
            print("  2. Click on 'Download' or 'Access'")
            print("  3. Read and accept the Terms of Use")
            print("  4. Then try running this script again")
            print("="*60)
            return None
            
        elif response.status_code == 404:
            print(f"[ERROR] Dataset not found: {dataset_id}")
            print("The dataset ID may have changed. Check the Mozilla Data Collective website.")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # API returns presigned URL
        download_url = data.get("url") or data.get("downloadUrl") or data.get("presignedUrl")
        if not download_url:
            print(f"[ERROR] Unexpected API response format: {data}")
            return None
        
        return download_url
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to get download URL: {e}")
        return None


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        chunk_size: Download chunk size
        
    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading to: {output_path}")
        if total_size:
            print(f"File size: {total_size / (1024**3):.2f} GB")
        
        # Download with progress
        with open(output_path, 'wb') as f:
            if TQDM_AVAILABLE and total_size:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end="", flush=True)
                print()
        
        print(f"[OK] Download complete: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract a tar.gz archive.
    
    Args:
        archive_path: Path to the archive
        extract_to: Directory to extract to
        
    Returns:
        True if successful
    """
    try:
        print(f"Extracting {archive_path.name}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        
        print(f"[OK] Extracted to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Mozilla Common Voice Hebrew Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Before running this script:
  1. Go to https://datacollective.mozillafoundation.org
  2. Create an account and log in
  3. Navigate to the Hebrew dataset and AGREE to Terms of Use
  4. Go to Profile Settings and generate an API key
  5. Add to .env: MOZILLA_DC_API_KEY=your_key_here
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: data/raw in project folder)"
    )
    
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help=f"Dataset ID to download (default: {DEFAULT_DATASET_ID})"
    )
    
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive after downloading"
    )
    
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show dataset info, don't download"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Mozilla Common Voice Hebrew Dataset Downloader")
    print("="*60)
    
    # Check dependencies
    if not validate_dependencies():
        return 1
    
    # Load .env file
    if not load_env_file():
        return 1
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("\n" + "="*60)
        print("[ERROR] API KEY NOT CONFIGURED")
        print("="*60)
        print("The Mozilla Data Collective API key is missing or invalid.")
        print("\nTo fix this:")
        print("  1. Go to https://datacollective.mozillafoundation.org")
        print("  2. Create an account and log in")
        print("  3. Go to Profile Settings → API Keys")
        print("  4. Generate a new API key")
        print("  5. Open .env file in this project")
        print("  6. Replace 'your_api_key_here' with your actual key")
        print("="*60)
        return 1
    
    # Dataset configuration
    dataset_id = args.dataset_id or os.environ.get("MOZILLA_DC_DATASET_ID", DEFAULT_DATASET_ID)
    
    print(f"Dataset ID: {dataset_id}")
    print(f"API URL: {API_BASE_URL}")
    
    # Get dataset info
    print("\nFetching dataset information...")
    info = get_dataset_info(api_key, dataset_id)
    
    if info:
        print(f"Dataset: {info.get('name', 'Unknown')}")
        print(f"Description: {info.get('description', 'N/A')[:100]}...")
    
    if args.info_only:
        if info:
            import json
            print("\nFull dataset info:")
            print(json.dumps(info, indent=2))
        return 0 if info else 1
    
    # Get download URL
    print("\nRequesting download URL...")
    download_url = get_download_url(api_key, dataset_id)
    
    if not download_url:
        return 1
    
    print("[OK] Got presigned download URL (valid for 12 hours)")
    
    # Determine output path
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = get_project_root() / "data" / "raw"
    
    output_file = output_dir / f"cv-corpus-24.0-he.tar.gz"
    
    # Download
    if not download_file(download_url, output_file):
        return 1
    
    # Extract if requested
    if args.extract:
        extract_dir = get_project_root() / "data" / "cv-corpus-24.0-he"
        if not extract_archive(output_file, extract_dir):
            return 1
        
        print(f"\nDataset extracted. Next step:")
        print(f"  python scripts/prepare_hebrew_dataset.py \\")
        print(f"      --input {extract_dir} \\")
        print(f"      --format commonvoice")
    else:
        print(f"\nDownload complete. To extract and prepare:")
        print(f"  python scripts/download_dataset.py --extract")
        print(f"  # or manually extract and run:")
        print(f"  python scripts/prepare_hebrew_dataset.py --input data/cv-corpus-24.0-he --format commonvoice")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
