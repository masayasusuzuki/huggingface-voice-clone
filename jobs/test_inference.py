# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.5.1",
#   "torchaudio==2.5.1",
#   "voxcpm>=2.0.2",
#   "huggingface_hub>=0.24",
#   "soundfile",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12>=9.1.0",
#   "nvidia-cuda-nvrtc-cu12",
#   "nvidia-cuda-runtime-cu12",
# ]
# ///
"""
Quick inference test: load LoRA + base model, generate 1 sample, upload to dataset.
"""
import json
import os
import sys
from pathlib import Path

import soundfile as sf
from huggingface_hub import HfApi, snapshot_download

LORA_REPO = "masayasu/ryuken-voice-lora"
LORA_RUN = os.environ.get("RUN_NAME", "ryuken-lora-test")
LORA_STEP = os.environ.get("LORA_STEP", "latest")
TARGET_TEXT = os.environ.get(
    "TARGET_TEXT",
    "今回はトークが劇的に上手くなる方法論についてお話ししていこうと思います。",
)
OUTPUT_TAG = os.environ.get("OUTPUT_TAG", "")  # extra tag for output filename
MAX_LEN = int(os.environ.get("MAX_LEN", "600"))
CFG_VALUE = float(os.environ.get("CFG_VALUE", "2.0"))
INFERENCE_TIMESTEPS = int(os.environ.get("INFERENCE_TIMESTEPS", "20"))
SEED = int(os.environ.get("SEED", "0"))  # 0 = random, otherwise fixed
OUTPUT_DATASET = "masayasu/ryuken-voice"
_suffix = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
OUTPUT_PATH = f"inference_tests/{LORA_RUN}_{LORA_STEP}{_suffix}.wav"


def main() -> int:
    print(f"[1/3] Downloading LoRA checkpoint {LORA_REPO}/{LORA_RUN}/{LORA_STEP}...", flush=True)
    lora_dir = Path("/tmp/lora")
    snapshot_download(
        repo_id=LORA_REPO,
        local_dir=str(lora_dir),
        repo_type="model",
        allow_patterns=[f"{LORA_RUN}/{LORA_STEP}/*"],
    )
    ckpt_dir = lora_dir / LORA_RUN / LORA_STEP
    print(f"  ckpt: {ckpt_dir}", flush=True)
    print(f"  files: {sorted(p.name for p in ckpt_dir.iterdir())}", flush=True)

    print(f"[2/3] Loading model + LoRA, generating audio...", flush=True)
    cfg_path = ckpt_dir / "lora_config.json"
    with open(cfg_path) as f:
        lora_info = json.load(f)
    base_model = lora_info["base_model"]
    print(f"  base model: {base_model}", flush=True)

    from voxcpm.core import VoxCPM
    from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
    from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1

    lora_cfg_dict = lora_info.get("lora_config", {})
    LoRAConfigCls = LoRAConfigV2 if "voxcpm2" in base_model.lower() else LoRAConfigV1
    lora_cfg = LoRAConfigCls(**lora_cfg_dict) if lora_cfg_dict else None

    model = VoxCPM.from_pretrained(
        hf_model_id=base_model,
        load_denoiser=False,
        optimize=False,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt_dir),
    )

    if SEED > 0:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        import random as _random
        import numpy as _np
        _random.seed(SEED)
        _np.random.seed(SEED)
        print(f"  fixed seed: {SEED}", flush=True)

    print(f"  generating ({len(TARGET_TEXT)} chars, max_len={MAX_LEN}, cfg={CFG_VALUE}, steps={INFERENCE_TIMESTEPS}, seed={SEED or 'random'}): {TARGET_TEXT!r}", flush=True)
    audio = model.generate(
        text=TARGET_TEXT,
        cfg_value=CFG_VALUE,
        inference_timesteps=INFERENCE_TIMESTEPS,
        max_len=MAX_LEN,
        normalize=True,
        denoise=False,
    )
    out_local = Path("/tmp/output.wav")
    sf.write(str(out_local), audio, model.tts_model.sample_rate)
    print(f"  saved local: {out_local}, dur={len(audio)/model.tts_model.sample_rate:.2f}s", flush=True)

    print(f"[3/3] Uploading to {OUTPUT_DATASET}/{OUTPUT_PATH}...", flush=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_local),
        path_in_repo=OUTPUT_PATH,
        repo_id=OUTPUT_DATASET,
        repo_type="dataset",
        commit_message=f"Inference test: {LORA_RUN}/{LORA_STEP}",
    )
    print(f"DONE. Listen at: https://huggingface.co/datasets/{OUTPUT_DATASET}/blob/main/{OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
