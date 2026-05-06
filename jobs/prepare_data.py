# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.5.0",
#   "torchaudio>=2.5.0",
#   "faster-whisper>=1.0.3",
#   "ctranslate2>=4.4.0",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12>=9.1.0",
#   "huggingface_hub>=0.24",
#   "soundfile>=0.12",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
VoxCPM LoRA data preparation job (HF Jobs version).

Downloads raw FLAC from HF dataset directly via hf_hub_download (no volume mount).
Uses faster-whisper large-v3 on GPU for VAD-segmented Japanese transcription.
Pushes wavs + manifest.jsonl back to the same dataset.
"""
import json
import os
import sys
from pathlib import Path


def _setup_nvidia_libs():
    """Add nvidia-cublas-cu12 and nvidia-cudnn-cu12 lib dirs to LD_LIBRARY_PATH
    so CTranslate2 (used by faster-whisper) can find them at GPU init."""
    import importlib.util

    lib_dirs = []
    for pkg in ["nvidia.cublas", "nvidia.cudnn"]:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        for base in spec.submodule_search_locations:
            cand = Path(base) / "lib"
            if cand.is_dir():
                lib_dirs.append(str(cand))
    if lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
        print(f"  LD_LIBRARY_PATH prepended: {lib_dirs}", flush=True)


_setup_nvidia_libs()

import soundfile as sf
import torch
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

DATASET_REPO = "masayasu/ryuken-voice"
INPUT_PATH_IN_REPO = "raw/sample_16k_mono.flac"

WORK_DIR = Path("/tmp/voxcpm_prep")
SEGMENTS_DIR = WORK_DIR / "segments"
MANIFEST_PATH = WORK_DIR / "manifest.jsonl"

MIN_SEG_SEC = 2.5
MAX_SEG_SEC = 25.0


def main() -> int:
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Downloading {INPUT_PATH_IN_REPO} from {DATASET_REPO}...", flush=True)
    flac_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=INPUT_PATH_IN_REPO,
        repo_type="dataset",
    )
    print(f"  saved to: {flac_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[2/5] Loading faster-whisper large-v3 ({device}, {compute_type})...", flush=True)
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)

    print(f"[3/5] Transcribing (Japanese, VAD on)...", flush=True)
    segments_iter, info = model.transcribe(
        flac_path,
        language="ja",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        word_timestamps=False,
    )
    print(f"  duration: {info.duration:.1f}s ({info.duration/60:.1f}min)", flush=True)

    full_audio, sr = sf.read(flac_path)
    print(f"  loaded full audio: {len(full_audio)/sr:.1f}s @ {sr}Hz", flush=True)

    print(f"[4/5] Slicing segments (min={MIN_SEG_SEC}s max={MAX_SEG_SEC}s)...", flush=True)
    manifest_lines = []
    skipped_short = 0
    skipped_long = 0
    skipped_empty = 0

    pbar = tqdm(desc="segments", unit="seg")
    for seg in segments_iter:
        pbar.update(1)
        start = float(seg.start)
        end = float(seg.end)
        text = (seg.text or "").strip()
        duration = end - start

        if not text:
            skipped_empty += 1
            continue
        if duration < MIN_SEG_SEC:
            skipped_short += 1
            continue
        if duration > MAX_SEG_SEC:
            skipped_long += 1
            continue

        idx = len(manifest_lines)
        s = max(0, int(start * sr))
        e = min(len(full_audio), int(end * sr))
        chunk = full_audio[s:e]

        out_name = f"seg_{idx:05d}.wav"
        out_path = SEGMENTS_DIR / out_name
        sf.write(out_path, chunk, sr, subtype="PCM_16")

        manifest_lines.append(
            json.dumps(
                {
                    "audio": f"segments/{out_name}",
                    "text": text,
                    "duration": round(duration, 3),
                },
                ensure_ascii=False,
            )
        )
    pbar.close()

    total_dur = sum(json.loads(l)["duration"] for l in manifest_lines)
    print(
        f"  kept {len(manifest_lines)} segments "
        f"(total {total_dur:.1f}s = {total_dur/60:.1f}min). "
        f"skipped: empty={skipped_empty} short={skipped_short} long={skipped_long}",
        flush=True,
    )

    if not manifest_lines:
        print("ERROR: no segments produced", file=sys.stderr)
        return 2

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"  wrote manifest: {MANIFEST_PATH}", flush=True)

    print(f"[5/5] Uploading {len(manifest_lines)} wavs + manifest to {DATASET_REPO}...", flush=True)
    api = HfApi()
    api.upload_folder(
        folder_path=str(WORK_DIR),
        repo_id=DATASET_REPO,
        repo_type="dataset",
        path_in_repo=".",
        commit_message=f"Add prepared dataset: {len(manifest_lines)} segments",
    )
    print("DONE.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
