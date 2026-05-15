"""
Kaggle/Colab Setup Script.
Run this first in a Kaggle notebook to install dependencies and set up the environment.

Usage in Kaggle notebook:
    !pip install -q -r requirements.txt
    %run setup_kaggle.py
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the environment for Kaggle/Colab."""
    
    # Add project root to path
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set HuggingFace token
    hf_token = os.environ.get("HF_TOKEN", "")
    
    # Try to get from Kaggle secrets
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            hf_token = secrets.get_secret("HF_TOKEN")
            os.environ["HF_TOKEN"] = hf_token
            print("✅ HF_TOKEN loaded from Kaggle secrets")
        except Exception:
            pass
    
    # Try to get from .env file
    if not hf_token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                print("✅ HF_TOKEN loaded from .env file")
        except ImportError:
            pass
    
    if not hf_token:
        print("⚠️ HF_TOKEN not found. Set it via Kaggle secrets or .env file.")
    
    # Login to HuggingFace
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Logged in to HuggingFace")
        except Exception as e:
            print(f"⚠️ HuggingFace login failed: {e}")
    
    # Create directories
    dirs = ["data/raw", "data/processed", "data/rendered_reports",
            "results/report_generation", "results/qa_results", "results/comparison"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            if torch.cuda.device_count() > 1:
                print(f"   {torch.cuda.device_count()} GPUs available")
        else:
            print("⚠️ No GPU available. Models will be very slow on CPU.")
    except ImportError:
        print("⚠️ PyTorch not installed.")
    
    print("\n✅ Environment setup complete!")
    return hf_token


if __name__ == "__main__":
    setup_environment()
