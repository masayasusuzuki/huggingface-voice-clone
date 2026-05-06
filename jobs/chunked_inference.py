# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.5.1",
#   "torchaudio==2.5.1",
#   "voxcpm>=2.0.2",
#   "huggingface_hub>=0.24",
#   "soundfile",
#   "numpy",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12>=9.1.0",
#   "nvidia-cuda-nvrtc-cu12",
#   "nvidia-cuda-runtime-cu12",
# ]
# ///
"""
Chunked inference: split text on 「。」, generate each chunk with controlled silence between.

Produces:
  - Combined wav: inference_tests/{RUN_NAME}_{LORA_STEP}_{OUTPUT_TAG}.wav
  - Per-chunk wavs: inference_tests/chunks/{RUN_NAME}_{OUTPUT_TAG}_chunk_{i}.wav
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, snapshot_download

LORA_REPO = "masayasu/ryuken-voice-lora"
LORA_RUN = os.environ.get("RUN_NAME", "ryuken-lora-full-5k")
LORA_STEP = os.environ.get("LORA_STEP", "latest")
TARGET_TEXT = os.environ.get(
    "TARGET_TEXT",
    "成功する人の特徴ってですね、即決即断ができることなんですよ。考えすぎる人は結果を出せない。動ける人だけが勝つ。シンプルな話です。",
)
OUTPUT_TAG = os.environ.get("OUTPUT_TAG", "chunked")
MAX_LEN = int(os.environ.get("MAX_LEN", "1500"))
CFG_VALUE = float(os.environ.get("CFG_VALUE", "2.0"))
INFERENCE_TIMESTEPS = int(os.environ.get("INFERENCE_TIMESTEPS", "50"))
SEED = int(os.environ.get("SEED", "0"))
PAUSE_PERIOD_SEC = float(os.environ.get("PAUSE_PERIOD_SEC", "0.4"))   # 「。」の後の無音
PAUSE_COMMA_SEC = float(os.environ.get("PAUSE_COMMA_SEC", "0.0"))     # 「、」の後 (0 = モデル任せ)
PAUSE_QUESTION_SEC = float(os.environ.get("PAUSE_QUESTION_SEC", "0.4")) # 「？」「?」
PAUSE_EXCLAIM_SEC = float(os.environ.get("PAUSE_EXCLAIM_SEC", "0.4"))  # 「！」「!」
SAVE_CHUNKS = os.environ.get("SAVE_CHUNKS", "1") == "1"

OUTPUT_DATASET = "masayasu/ryuken-voice"
OUTPUT_PATH = f"inference_tests/{LORA_RUN}_{LORA_STEP}_{OUTPUT_TAG}.wav"


def split_into_chunks(text: str) -> list[tuple[str, float]]:
    """Split text on terminal punctuation, return list of (chunk_text, post_silence_sec)."""
    # split keeping the punctuation. supports 。.！!？?
    pattern = r"([。．\.！!？?])"
    parts = re.split(pattern, text)
    chunks = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if p in ("。", "．", "."):
            if buf.strip():
                chunks.append((buf.strip() + p, PAUSE_PERIOD_SEC))
                buf = ""
        elif p in ("！", "!"):
            if buf.strip():
                chunks.append((buf.strip() + p, PAUSE_EXCLAIM_SEC))
                buf = ""
        elif p in ("？", "?"):
            if buf.strip():
                chunks.append((buf.strip() + p, PAUSE_QUESTION_SEC))
                buf = ""
        else:
            buf += p
    if buf.strip():
        chunks.append((buf.strip(), 0.0))
    return chunks


def main() -> int:
    print(f"[1/4] Downloading LoRA {LORA_REPO}/{LORA_RUN}/{LORA_STEP}...", flush=True)
    lora_root = Path("/tmp/lora")
    snapshot_download(
        repo_id=LORA_REPO,
        local_dir=str(lora_root),
        repo_type="model",
        allow_patterns=[f"{LORA_RUN}/{LORA_STEP}/*"],
    )
    ckpt_dir = lora_root / LORA_RUN / LORA_STEP
    print(f"  ckpt: {ckpt_dir}", flush=True)

    print(f"[2/4] Loading model + LoRA...", flush=True)
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
    sample_rate = model.tts_model.sample_rate
    print(f"  sample_rate: {sample_rate}", flush=True)

    print(f"[3/4] Splitting text + generating chunks...", flush=True)
    chunks = split_into_chunks(TARGET_TEXT)
    print(f"  {len(chunks)} chunks (pause: 。={PAUSE_PERIOD_SEC}s 、={PAUSE_COMMA_SEC}s ？={PAUSE_QUESTION_SEC}s ！={PAUSE_EXCLAIM_SEC}s)", flush=True)
    for i, (c, p) in enumerate(chunks):
        print(f"  [{i}] ({len(c)}c, +{p}s pause): {c!r}", flush=True)

    if SEED > 0:
        import torch
        print(f"  Using fixed seed: {SEED}", flush=True)

    audio_segments = []
    chunk_dir = Path("/tmp/chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for i, (chunk_text, post_silence) in enumerate(chunks):
        print(f"  [chunk {i+1}/{len(chunks)}] generating ({len(chunk_text)} chars)...", flush=True)
        if SEED > 0:
            import torch
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            import random as _random
            _random.seed(SEED)
            np.random.seed(SEED)

        audio = model.generate(
            text=chunk_text,
            cfg_value=CFG_VALUE,
            inference_timesteps=INFERENCE_TIMESTEPS,
            max_len=MAX_LEN,
            normalize=True,
            denoise=False,
        )
        audio = np.asarray(audio, dtype=np.float32)
        dur = len(audio) / sample_rate
        print(f"           generated: dur={dur:.2f}s", flush=True)

        if SAVE_CHUNKS:
            chunk_path = chunk_dir / f"{LORA_RUN}_{OUTPUT_TAG}_chunk_{i:02d}.wav"
            sf.write(str(chunk_path), audio, sample_rate)

        audio_segments.append(audio)
        if post_silence > 0:
            silence = np.zeros(int(post_silence * sample_rate), dtype=np.float32)
            audio_segments.append(silence)

    combined = np.concatenate(audio_segments)
    out_local = Path("/tmp/output.wav")
    sf.write(str(out_local), combined, sample_rate)
    total_dur = len(combined) / sample_rate
    print(f"  combined wav: {out_local}, total dur={total_dur:.2f}s", flush=True)

    print(f"[4/4] Uploading to {OUTPUT_DATASET}/{OUTPUT_PATH}...", flush=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_local),
        path_in_repo=OUTPUT_PATH,
        repo_id=OUTPUT_DATASET,
        repo_type="dataset",
        commit_message=f"Chunked inference: {LORA_RUN}/{LORA_STEP} {OUTPUT_TAG}",
    )

    if SAVE_CHUNKS:
        print(f"  uploading per-chunk wavs...", flush=True)
        api.upload_folder(
            folder_path=str(chunk_dir),
            repo_id=OUTPUT_DATASET,
            repo_type="dataset",
            path_in_repo=f"inference_tests/chunks/{LORA_RUN}_{OUTPUT_TAG}",
            commit_message=f"Chunks: {LORA_RUN}/{LORA_STEP} {OUTPUT_TAG}",
        )

    print(f"DONE. Combined: https://huggingface.co/datasets/{OUTPUT_DATASET}/blob/main/{OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
