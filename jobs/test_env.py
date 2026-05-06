# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.5.0",
#   "huggingface_hub>=0.24",
# ]
# ///
"""Smoke test: verify GPU, dataset mount, HF_TOKEN."""
import os
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

print("=" * 60)
print("VoxCPM Job Environment Smoke Test")
print("=" * 60)

print("\n[1] Python / OS")
print(f"  python: {sys.version}")
print(f"  uname:  {os.uname()}")

print("\n[2] PyTorch / CUDA")
print(f"  torch.__version__: {torch.__version__}")
print(f"  cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device count: {torch.cuda.device_count()}")
    print(f"  device 0:     {torch.cuda.get_device_name(0)}")
    print(f"  cuda version: {torch.version.cuda}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  vram free:    {free/1e9:.1f}GB / {total/1e9:.1f}GB")

print("\n[3] Dataset mount at /dataset_in")
data_dir = Path("/dataset_in")
if data_dir.exists():
    print(f"  /data exists")
    for p in data_dir.rglob("*"):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"    {p}  ({size_mb:.1f}MB)")
else:
    print(f"  /data does NOT exist")

print("\n[4] HF_TOKEN auth")
try:
    api = HfApi()
    user = api.whoami()
    print(f"  whoami: {user.get('name')}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n[5] nvidia-smi")
try:
    out = subprocess.check_output(["nvidia-smi"], text=True)
    print(out[:500])
except Exception as e:
    print(f"  no nvidia-smi: {e}")

print("\nDONE")
