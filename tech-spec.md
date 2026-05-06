# Voice Clone 技術仕様書

最終更新：2026-05-02
ステータス：**MVP検証完了 / 運用・スケール段階に向けたリファレンス**

サービス全体の要件は `voice-clone-service-spec.md` を参照。本書は **技術スタック・パラメータ・ノウハウ** を実装目線でまとめたもの。

---

## 1. パイプライン全体像

```
┌──────────┐  ┌─────────────┐  ┌────────────┐  ┌───────────┐
│ 元音声    │→│ データ準備   │→│ LoRA学習   │→│ 推論      │
│ wav/mp3   │  │ (HF Job)    │  │ (HF Job)   │  │ (HF Job/  │
│ 3時間〜    │  │ 分割+書起   │  │ L4 GPU     │  │  Space)   │
└──────────┘  └─────────────┘  └────────────┘  └───────────┘
                     ↓                ↓                ↓
                ryuken-voice    ryuken-voice-     生成wav
                (Dataset)       lora (Model)      （試聴 or 配信）
```

---

## 2. ストレージ構成（HuggingFace）

| リポジトリ | 種類 | 用途 |
|---|---|---|
| `masayasu/ryuken-voice` | Dataset (Private) | 元音声 + 学習用セグメント + manifest + ジョブスクリプト + 推論結果 |
| `masayasu/ryuken-voice-lora` | Model (Private) | LoRA学習済み重み（runごと/stepごと） |
| `masayasu/jobs-artifacts` | Bucket (Private, 自動生成) | HF Jobs内部用 |
| `masayasu/voxcpm-personal` | Space (Private) | 推論UI（Gradio） |

### 2.1 ryuken-voice の構造

```
masayasu/ryuken-voice/
├── raw/
│   └── sample_16k_mono.flac      # 元音声（16kHz mono FLAC, 228MB）
├── segments/                      # 学習用セグメント
│   ├── seg_00000.wav             # 2.5〜25秒の音声片（2008個）
│   └── ...
├── manifest.jsonl                 # 全セグメントの (path, text, duration)
├── jobs/                          # HFジョブで使うスクリプト
│   ├── prepare_data.py
│   ├── train_lora.py
│   └── test_inference.py
└── inference_tests/               # 試聴用の生成結果
    └── *.wav
```

### 2.2 ryuken-voice-lora の構造

```
masayasu/ryuken-voice-lora/
├── ryuken-lora-test/             # 500iterテスト
│   ├── step_0000000/
│   ├── step_0000250/
│   ├── step_0000500/
│   └── latest/                   # 最終stepのコピー
└── ryuken-lora-full-5k/          # 5000iter本番
    ├── step_0001000/
    ├── step_0002000/
    ├── step_0003000/
    ├── step_0004000/
    ├── step_0005000/
    └── latest/
```

各stepディレクトリに `lora_weights.safetensors`（72.4MB）と `lora_config.json`、`optimizer.pth`等。

---

## 3. データ準備仕様

### 3.1 元音声の前処理

```bash
ffmpeg -y -i sample_full.wav \
  -ac 1 -ar 16000 \
  -c:a flac -compression_level 8 \
  sample_16k_mono.flac
```

- 入力：48kHz stereo PCM 2.5GB（3時間49分）
- 出力：16kHz mono FLAC 228MB（VoxCPMが学習に使う形式）
- 圧縮率：約11倍

### 3.2 セグメント分割 + 書き起こし

`jobs/prepare_data.py` で実装。

| 項目 | 値 |
|---|---|
| ASRモデル | faster-whisper large-v3 (fp16, GPU) |
| VAD | silero VAD（faster-whisperに内蔵） |
| min_silence_duration_ms | 400 |
| 言語 | ja |
| beam_size | 5 |
| MIN_SEG_SEC | 2.5 |
| MAX_SEG_SEC | 25.0 |

### 3.3 出力manifest形式

```json
{"audio": "segments/seg_00000.wav", "text": "山本竜剣の人生救済チャンネル どうもこんにちは 山本竜剣です", "duration": 3.34}
```

### 3.4 実績値（3時間49分音声 → セグメント化）

| 項目 | 値 |
|---|---|
| 元音声時間 | 13,723秒（228.7分） |
| 採用セグメント数 | 2,008個 |
| 採用音声合計 | 8,158秒（136分） |
| Skipped: empty | 0 |
| Skipped: short (<2.5s) | 3,285 |
| Skipped: long (>25s) | 7 |
| データ準備所要時間 | 約50分 |
| L4 GPUコスト | 約 $0.65 |

---

## 4. LoRA学習仕様

### 4.1 ベースモデル

- `openbmb/VoxCPM2`（Apache-2.0、商用可、2Bパラメータ、HuggingFace Hub）

### 4.2 LoRA設定

```yaml
lora:
  enable_lm: true     # Language model に LoRA適用
  enable_dit: true    # Diffusion Transformer に LoRA適用
  enable_proj: false  # 出力Projectionには適用しない
  r: 32               # LoRA rank
  alpha: 32           # LoRA scaling
  dropout: 0.0
```

### 4.3 学習ハイパーパラメータ

| 項目 | テスト | **本番** |
|---|---|---|
| num_iters | 500 | **5000** |
| save_interval | 250 | 1000 |
| batch_size | 2 | 2 |
| grad_accum_steps | 8 | 8 |
| effective_batch_size | 16 | 16 |
| learning_rate | 1e-4 | 1e-4 |
| weight_decay | 0.01 | 0.01 |
| warmup_steps | 50 | 100 |
| max_grad_norm | 1.0 | 1.0 |
| max_batch_tokens | 8,192 | 8,192 |
| sample_rate | 16,000 | 16,000 |
| out_sample_rate | 48,000 | 48,000 |

### 4.4 学習結果（5000iter）

- 学習時間：約3時間（1step ≈ 2.1秒、L4）
- L4 GPUコスト：約 $2.40
- Loss推移：1.02（step 0）→ 0.80（step 4900、最低値）
- 1エポック ≈ 125 iter（2008サンプル / batch 16）
- **5000iter = 約40エポック**
- 出力LoRA重み：72.4MB（safetensors）

### 4.5 推奨LoRA configレンジ（実験案）

| 用途 | rank | alpha | iter | 想定コスト |
|---|---|---|---|---|
| 軽量・高速学習 | 16 | 16 | 3000 | $1.5 |
| **標準（今回採用）** | **32** | **32** | **5000** | **$2.4** |
| 高表現力 | 64 | 64 | 8000 | $4.5 |
| 究極 | 128 | 128 | 10000 | $7〜 |

---

## 5. 推論仕様

### 5.1 主要パラメータと体感

実機テストで確認済みの効果：

| パラメータ | 値域 | 効果 | 推奨 |
|---|---|---|---|
| **CFG (cfg_value)** | 1.0〜3.0 | 学習データへの忠実度 | **2.0**（個性最大、品質高） |
| **inference_timesteps** | 10〜100 | 拡散ステップ数 = 滑らかさ | **50〜100** |
| **max_len** | 600〜2000 | 最大生成step数 | 文字数に応じて、180字なら2000 |
| **normalize** | bool | テキスト正規化（数字・略語） | **True**（日本語で必須） |
| **denoise** | bool | リファレンス音声のZipEnhancer | False（LoRAなら不要） |

### 5.2 CFG実験結果（5000iter LoRA）

| CFG | 体感 | 評価 |
|---|---|---|
| 1.2 | 似てるが弱い、特徴薄い | △ |
| 1.5 | 自然だが個性弱い | ○ |
| **2.0** | **本人らしさ最大、若干誇張** | **◎ 採用** |
| 2.5+ | 不自然（過剰演技） | × |

### 5.3 timesteps実験結果（CFG=2.0固定）

| timesteps | 生成時間（180字） | 品質 |
|---|---|---|
| 20 | 約12秒 | やや粗い |
| 50 | 約30秒 | 滑らか・推奨 |
| **100** | 約60秒 | **最高品質・本番採用** |

### 5.4 推奨デフォルトプリセット

```python
generate(
    text=...,
    cfg_value=2.0,
    inference_timesteps=100,
    max_len=2000,
    normalize=True,
    denoise=False,
)
```

---

## 6. HF Jobs 運用コマンド

### 6.1 認証

```bash
# 一回だけ
hf auth login --token hf_xxxxxxxxx
```

### 6.2 データ準備ジョブ

```bash
hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --timeout 2h \
  -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/prepare_data.py"
```

### 6.3 LoRA学習ジョブ

```bash
hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --env RUN_NAME=ryuken-lora-full-5k \
  --env NUM_ITERS=5000 \
  --env SAVE_INTERVAL=1000 \
  --timeout 5h \
  -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/train_lora.py"
```

### 6.4 推論ジョブ

```bash
hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --env RUN_NAME=ryuken-lora-full-5k \
  --env LORA_STEP=latest \
  --env OUTPUT_TAG=demo \
  --env MAX_LEN=2000 \
  --env CFG_VALUE=2.0 \
  --env INFERENCE_TIMESTEPS=100 \
  --env "TARGET_TEXT=...生成したい日本語テキスト..." \
  --timeout 30m \
  -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/test_inference.py"
```

### 6.5 ジョブ状態確認

```bash
hf jobs ps -a              # 全ジョブ一覧
hf jobs inspect <JOB_ID>   # 詳細
hf jobs logs <JOB_ID>      # ログ
hf jobs cancel <JOB_ID>    # キャンセル
```

---

## 7. ハードウェア仕様（HuggingFace）

| Flavor | GPU | VRAM | 単価 | 用途 |
|---|---|---|---|---|
| `l4x1` | 1x Nvidia L4 | 24GB | $0.80/h | **本タスク採用** |
| `a10g-large` | 1x Nvidia A10G | 24GB | $1.50/h | 学習やや高速 |
| `a100-large` | 1x A100 | 80GB | $2.50/h | 大規模LoRA・全パラ学習向け |
| `t4-small` | 1x T4 | 16GB | $0.40/h | **不可（FlashAttention非対応）** |

L4はAda Lovelace世代でFlashAttention対応、コスパ最良。

---

## 8. コスト実績（プロトタイプ完走時点）

| フェーズ | 内訳 | 金額 |
|---|---|---|
| データ準備 | L4 50分 | $0.65 |
| テスト学習（500iter） | L4 30分 | $0.40 |
| 本番学習（5000iter） | L4 3時間 | $2.40 |
| 推論テスト（短文） | L4 5分×2回 | $0.13 |
| 推論テスト（長文） | L4 8分×6回 | $0.83 |
| ボツジョブ・リトライ | 計 約20分 | $0.27 |
| **合計** | **約 5時間** | **約 $4.7** |

予算：$10チャージ → **残$5程度**で2回目の本番学習可能。

---

## 9. 既知の問題と回避策

### 9.1 HF Jobs ボリュームマウント失敗

**症状**：`Volume mount failed: Volume mount failed`が間欠的に発生
**原因**：jobs-artifactsバケットへのスクリプト書き込みが不安定
**回避策**：ローカルファイル指定ではなく、**スクリプトをHF Datasetに先にuploadしてURLで指定**

```bash
# NG: ローカルパス指定（バケットマウント経由）
hf jobs uv run ./jobs/script.py

# OK: URL指定（マウント不要）
hf jobs uv run "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/script.py"
```

### 9.2 cuBLAS / cuDNN / nvrtc not found

**症状**：`libcublas.so.12 is not found` / `libnvrtc-builtins.so.13.0` 等
**原因**：uv環境にCUDAランタイムライブラリが入っていない
**回避策**：スクリプトのdependenciesに以下を明示

```python
# /// script
# dependencies = [
#   "torch==2.5.1",
#   "torchaudio==2.5.1",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12>=9.1.0",
#   "nvidia-cuda-nvrtc-cu12",
#   "nvidia-cuda-runtime-cu12",
# ]
# ///
```

実行前に LD_LIBRARY_PATH の設定も推奨：

```python
def _setup_nvidia_libs():
    import importlib.util
    lib_dirs = []
    for pkg in ["nvidia.cublas", "nvidia.cudnn"]:
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            for base in spec.submodule_search_locations:
                cand = Path(base) / "lib"
                if cand.is_dir():
                    lib_dirs.append(str(cand))
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])

_setup_nvidia_libs()
```

### 9.3 manifest内の相対パスで音声が読めない

**症状**：`FileNotFoundError: 'segments/seg_xxxxx.wav'`
**原因**：`load_dataset("json")`が相対パスを解決できない
**回避策**：学習スクリプト内で **manifestを絶対パスに書き換えてから渡す**

### 9.4 train_voxcpm_finetune.py の引数

**症状**：`TypeError: train() missing 2 required positional arguments`
**原因**：`--args.load`は無効、正しくは`--config_path`
**回避策**：

```bash
python scripts/train_voxcpm_finetune.py --config_path config.yaml
```

### 9.5 HF API レートリミット（whoami-v2）

**症状**：`You've hit the rate limit for the /whoami-v2 endpoint`
**原因**：短時間に多数のジョブ操作・キャンセル等
**回避策**：60〜180秒待ってから再試行。長時間放置で完全クリア。

---

## 10. テキスト正規化（normalize=True）の効果

VoxCPMは内部で `wetext` ライブラリを使用。

| 入力 | 出力（読み） |
|---|---|
| 2026年5月2日 | にせんにじゅうろくねん ごがつ ふつか |
| 100% | ひゃくぱーせんと |
| Apple Inc. | アップル・インク |
| 〜って感じで | 〜って感じで（変換なし） |

**OFFにすると数字・記号が機械読みになるので必ずON推奨**。

---

## 11. 推論レイテンシ実測（L4）

| テキスト | timesteps | 生成時間 |
|---|---|---|
| 短文（40字、5秒音声） | 20 | 約8秒 |
| 短文（40字） | 50 | 約20秒 |
| **長文（170字、約60秒音声）** | **100** | **約60秒** |
| 短文 | 100 | 約12秒 |

体感：**生成時間 ≈ 出力音声時間 × 1.0〜1.2倍**（timesteps=100時、L4）

---

## 12. スケール時の論点（将来用）

### 12.1 マルチテナント化
- ユーザーごとにLoRA重みを分離保管：`<service>/lora-<user_id>`
- 同時学習ジョブ数の上限管理（HF Jobs同時実行数）
- ストレージ分離：`<service>/dataset-<user_id>`

### 12.2 推論レイテンシ短縮
- HF Inference Endpoints で常時起動（Cold start解消）
- Modal serverless GPU（spot pricing $0.65/hour程度）
- 自前GPUサーバ（A10G $0.30〜0.40/hour、3rd party）

### 12.3 コスト最適化
- LoRA重みは10MB台に圧縮可能（rank=8, alpha=16）→ 軽量プラン用
- 推論バッチ化（複数ユーザーリクエストを1GPUで束ねる）
- スポット価格GPU（HF Jobs価格の50%以下）

### 12.4 品質向上ロードマップ
1. データ追加（3h → 6h → 10h）
2. iter延長（5000 → 10000）
3. LoRA rank↑（32 → 64 → 128）
4. 手動データクリーニング
5. 多様性確保（複数話速・トーン）

---

## 13. ファイル所在（ローカル）

```
/Users/suzukimotoyasumain/Desktop/MASAYASU/
├── voice-clone-service-spec.md      # サービス要件定義
├── voice-clone-tech-spec.md         # 本書（技術仕様）
└── products/VoxCPM/                  # VoxCPMソースコード（git clone）

/Users/suzukimotoyasumain/Desktop/samplevoice/
├── sample_full.wav                   # 元音声（2.5GB）
├── processed/
│   └── sample_16k_mono.flac          # 変換済み（228MB）
└── jobs/
    ├── prepare_data.py
    ├── train_lora.py
    └── test_inference.py
```

---

## 14. 参考リンク

- VoxCPM公式: https://github.com/OpenBMB/VoxCPM
- VoxCPM Hugging Face: https://huggingface.co/openbmb/VoxCPM2
- HF Jobs CLI Doc: https://huggingface.co/docs/huggingface_hub/en/guides/cli
- HF Storage Buckets: https://huggingface.co/docs/hub/storage-buckets
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
