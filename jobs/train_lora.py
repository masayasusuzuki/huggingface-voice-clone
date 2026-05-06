# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.5.1",
#   "torchaudio==2.5.1",
#   "voxcpm>=2.0.2",
#   "huggingface_hub>=0.24",
#   "argbind",
#   "tensorboardX",
#   "pyyaml",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12>=9.1.0",
#   "nvidia-cuda-nvrtc-cu12",
#   "nvidia-cuda-runtime-cu12",
# ]
# ///
"""
VoxCPM LoRA training job (HF Jobs version).

- Downloads prepared dataset (segments + manifest.jsonl) from HF dataset.
- Downloads VoxCPM2 base model from HF model hub.
- Runs LoRA fine-tuning via the official train script (from cloned GH repo).
- Uploads LoRA checkpoints to a HF model repo.

Args (env-driven so a single script handles test + full runs):
  RUN_NAME       (default: ryuken-lora-test) — sub-directory name in checkpoints/
  NUM_ITERS      (default: 500)             — total training iterations
  SAVE_INTERVAL  (default: 250)             — checkpoint frequency
  BATCH_SIZE     (default: 2)
  GRAD_ACCUM     (default: 8)
  LEARNING_RATE  (default: 1e-4)
  LORA_R         (default: 32)
  LORA_ALPHA     (default: 32)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DATASET_REPO = "masayasu/ryuken-voice"
BASE_MODEL_REPO = "openbmb/VoxCPM2"
LORA_OUTPUT_REPO = "masayasu/ryuken-voice-lora"

WORK = Path("/tmp/voxcpm_train")
DATA_DIR = WORK / "dataset"
MODEL_DIR = WORK / "base_model"
CKPT_DIR = WORK / "checkpoints"
REPO_DIR = WORK / "VoxCPM"

RUN_NAME = os.environ.get("RUN_NAME", "ryuken-lora-test")
NUM_ITERS = int(os.environ.get("NUM_ITERS", "500"))
SAVE_INTERVAL = int(os.environ.get("SAVE_INTERVAL", "250"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR = float(os.environ.get("LEARNING_RATE", "1e-4"))
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Cloning VoxCPM repo (for train script)...", flush=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "https://github.com/OpenBMB/VoxCPM.git", str(REPO_DIR)])

    print(f"[2/6] Downloading base model {BASE_MODEL_REPO}...", flush=True)
    snapshot_download(repo_id=BASE_MODEL_REPO, local_dir=str(MODEL_DIR), repo_type="model")

    print(f"[3/6] Downloading dataset {DATASET_REPO}...", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        local_dir=str(DATA_DIR),
        repo_type="dataset",
        allow_patterns=["manifest.jsonl", "segments/*.wav"],
    )

    manifest_local = DATA_DIR / "manifest.jsonl"
    if not manifest_local.exists():
        print(f"ERROR: manifest.jsonl not found at {manifest_local}", file=sys.stderr)
        return 1

    print(f"[3.5/6] Rewriting manifest with absolute audio paths...", flush=True)
    import json as _json

    rewritten = WORK / "manifest_abs.jsonl"
    with open(manifest_local, "r", encoding="utf-8") as fin, open(
        rewritten, "w", encoding="utf-8"
    ) as fout:
        n = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = _json.loads(line)
            rel = entry["audio"]
            absolute = DATA_DIR / rel
            if not absolute.exists():
                print(f"  WARNING: missing {absolute}", file=sys.stderr)
                continue
            entry["audio"] = str(absolute)
            fout.write(_json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    print(f"  rewrote {n} entries -> {rewritten}", flush=True)
    manifest_local = rewritten

    print(f"[4/6] Building training config...", flush=True)
    save_path = CKPT_DIR / RUN_NAME
    save_path.mkdir(parents=True, exist_ok=True)
    config = {
        "pretrained_path": str(MODEL_DIR),
        "train_manifest": str(manifest_local),
        "val_manifest": "",
        "sample_rate": 16000,
        "out_sample_rate": 48000,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM,
        "num_workers": 4,
        "num_iters": NUM_ITERS,
        "log_interval": 25,
        "valid_interval": SAVE_INTERVAL,
        "save_interval": SAVE_INTERVAL,
        "learning_rate": LR,
        "weight_decay": 0.01,
        "warmup_steps": min(100, NUM_ITERS // 10),
        "max_steps": NUM_ITERS,
        "max_batch_tokens": 8192,
        "max_grad_norm": 1.0,
        "save_path": str(save_path),
        "tensorboard": str(save_path / "logs"),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
        },
        "hf_model_id": BASE_MODEL_REPO,
        "distribute": True,
    }
    config_path = WORK / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"  config: {config_path}", flush=True)
    print(yaml.safe_dump(config, sort_keys=False), flush=True)

    print(f"[5/6] Running training (num_iters={NUM_ITERS})...", flush=True)
    cmd = [
        sys.executable,
        str(REPO_DIR / "scripts" / "train_voxcpm_finetune.py"),
        "--config_path",
        str(config_path),
    ]
    run(cmd)

    print(f"[6/6] Uploading LoRA checkpoints to {LORA_OUTPUT_REPO}...", flush=True)
    api = HfApi()
    api.create_repo(repo_id=LORA_OUTPUT_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(save_path),
        repo_id=LORA_OUTPUT_REPO,
        repo_type="model",
        path_in_repo=RUN_NAME,
        commit_message=f"Upload LoRA run: {RUN_NAME} ({NUM_ITERS} iters)",
        ignore_patterns=["logs/*", "*.log"],
    )
    print(f"DONE. LoRA at: https://huggingface.co/{LORA_OUTPUT_REPO}/tree/main/{RUN_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
