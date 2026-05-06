# Voice Clone 実験ログ

ステータス：**進行中（5000iter LoRA + 推論パラメータ最適化フェーズ）**

開始：2026-05-02
最終更新：2026-05-02

VoxCPM2 + LoRA fine-tuning + HF Jobs を使った日本語音声クローンの一連の実験を時系列で記録。

---

## 0. 前提

- 元音声：3時間49分の本人発話（48kHz stereo WAV、2.5GB）
- ターゲット：日本語のテキスト読み上げで本人の声を再現
- 制約：プライベート運用、HFアカウント `masayasu` 内で完結

---

## 1. 環境構築フェーズ

### 1.1 HF CLI セットアップ
- `pip install huggingface_hub[cli]` で `hf` v1.13.0 を導入
- 旧 `huggingface-cli` は deprecated → 新 `hf` を使用
- PATH：`/Users/suzukimotoyasumain/Library/Python/3.11/bin/hf`

### 1.2 認証
- `https://huggingface.co/settings/tokens` で **Write** 権限のトークン作成（name: `cli-claude`）
- `hf auth login --token hf_xxx` で認証
- トークンは `~/.cache/huggingface/token` に保存

### 1.3 元音声の前処理（ローカル）
- `ffmpeg -ac 1 -ar 16000 -c:a flac sample_16k_mono.flac` で変換
- 2.5GB stereo 48kHz → 228MB mono 16kHz FLAC
- 圧縮率：約11倍

### 1.4 HF Datasets作成＆アップロード
- `hf repo create masayasu/ryuken-voice --repo-type dataset --private`
- `hf upload masayasu/ryuken-voice processed/sample_16k_mono.flac raw/sample_16k_mono.flac --repo-type dataset`

---

## 2. データ準備フェーズ（HF Job）

### 2.1 トラブル：ボリュームマウント失敗
- `hf jobs uv run` でローカルスクリプト指定 → "Volume mount failed" エラー間欠的発生
- 原因：jobs-artifacts バケットへの書き込みが不安定
- 解決：**スクリプトをDatasetに先uploadして URL指定で実行**

### 2.2 トラブル：cuBLAS/cuDNN/nvrtc not found
- faster-whisper の GPU 初期化で `libcublas.so.12 not found`
- 解決：deps に `nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cuda-nvrtc-cu12` を明示
- `LD_LIBRARY_PATH` を実行時に設定する `_setup_nvidia_libs()` を追加

### 2.3 データ準備実績
- faster-whisper large-v3 + silero VAD（min_silence=400ms）で書き起こし＆分割
- 元音声 13,723秒 → **2,008セグメント / 8,158秒 (136分)**
- Skipped: short(<2.5s)=3,285、long(>25s)=7、empty=0
- 所要時間：約50分、コスト：約 $0.65（L4 GPU）

---

## 3. LoRA学習フェーズ（HF Job）

### 3.1 トラブル：argbindのCLIフラグ
- `--args.load config.yaml` でエラー → `train() missing required arguments`
- 解決：正しくは **`--config_path config.yaml`**

### 3.2 トラブル：manifest内の相対パス解決失敗
- `FileNotFoundError: 'segments/seg_xxxxx.wav'`
- 解決：学習ジョブ内で **manifest.jsonl を絶対パスに書き換え** てから渡す

### 3.3 テスト学習（500 iter）
- batch_size=2, grad_accum=8, lr=1e-4, LoRA r=32 alpha=32
- Loss: 1.02 → 0.83 で下降確認
- 所要：約30分、コスト：$0.40
- LoRA重み（72.4MB）保存：`masayasu/ryuken-voice-lora/ryuken-lora-test`

### 3.4 本番学習（5000 iter）
- 同設定、5000ステップ＝約40エポック
- Loss: 1.02 (step 0) → 0.80 (step 4900最低値)
- 所要：約3時間、コスト：$2.40
- 1step ≈ 2.1秒（L4）
- LoRA重み保存：`masayasu/ryuken-voice-lora/ryuken-lora-full-5k`

---

## 4. 推論パラメータ実験

### 4.1 短文（40字、起業話）
- 第一弾：CFG=2.0, timesteps=20 → 「めっちゃいい」
- 5000iterで本人らしさ最大、テスト学習(500iter)はベタっと薄い

### 4.2 長文（170字、起業話）
- CFG=2.0, timesteps=20 → 「クオリティ高いけどイントネーションが強すぎる」

### 4.3 CFG探索（長文・5000iter）
| CFG | 評価 |
|---|---|
| 1.2 | 弱い、特徴薄い |
| 1.5 | 自然だが個性弱い |
| **2.0** | **本人らしさ最大、若干誇張** ◎ |
| 2.5+ | 不自然・過剰 |

→ **CFG=2.0 が採用基準**

### 4.4 timesteps探索（CFG=2.0固定・長文）
| timesteps | 評価 |
|---|---|
| 20 | やや粗い |
| **50** | **滑らか・最適** ◎ |
| 100 | 平坦化、逆に微妙 |

→ **timesteps=50 が採用基準**

### 4.5 「。」での停止制御（チャンク分割実験）
- `chunked_inference.py` を実装：「。」で分割→個別生成→無音0.4s挟んで結合
- 結果：**境界の抑揚が断絶し非常に不自然**
- 結論：**チャンク分割は廃止**、1回で全文生成方式へ戻す

### 4.6 シード制御の追加
- 推論ごとに結果が異なる現象（拡散モデルの確率的ノイズ）
- 対策：`SEED` 環境変数で torch / numpy / random を全部固定
- 検証：seed=5 を3回連続生成 → ほぼ完全同一を確認（GPU微小揺らぎのみ）

---

## 5. シードガチャ実験

### 5.1 テキストごとに最適シードが違うことを確認

| テキスト | 試したseed | ベスト |
|---|---|---|
| 副業マインド系（170字） | 1,2,3,4,5 → 10,25,42,67,90 | seed=5（最初） / seed=42, 90（次） |
| 行動量系（122字, ですます調） | 5,42,90 → 9,15,20 → 3,5,7 → 5×3 | seed=5（記録） |
| 行動量系（広域分散） | 5, 100, 1000, 10000, 100000 | （比較データとして取得） |
| ビジネス＝時間資産系（154字） | 5, 25, 100, 1000, 10000 | （実行中／評価中） |
| 起業失敗系（170字） | 5, 100, 1000 | （実行中／評価中） |

### 5.2 シードに関する知見

- 同じseed・同じテキスト = ほぼ同一音声（決定論的）
- 違うseed = 違うノイズ起点 = 違う音声
- 当たりseedはテキストごとに変わる（汎用的に使えるseedはない）
- 5本中1〜2本が「当たり」体感
- 案：将来的にWhisper(WER) + 話者類似度で**自動評価＋自動選別**を実装すれば運用化可能

---

## 6. 廃止された方針

- `chunked_inference.py`：「。」分割→結合方式（境界が不自然のため）

---

## 7. 確定した運用パラメータ（2026-05-02時点）

```python
# Inference best practice
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 50
MAX_LEN = 1500〜2000  # テキスト長に応じて
SEED = テキストごとにガチャで決定
normalize = True
denoise = False

# LoRA Training
RUN_NAME = "ryuken-lora-full-5k"  # 5000iter版
LORA_STEP = "latest"  # = step_5000
```

---

## 8. 累計コスト

| フェーズ | 内訳 | 金額 |
|---|---|---|
| データ準備 | L4 50min | $0.65 |
| テスト学習(500iter) | L4 30min | $0.40 |
| 本番学習(5000iter) | L4 3h | $2.40 |
| 推論実験（多数回） | L4 計約2h | $1.60〜 |
| ボツジョブ | 計20min | $0.27 |
| **累計** | | **約 $5〜6** |

残予算：$10チャージ → **約$4〜5残**（さらに本番学習1回 + 推論多数回が可能）

---

## 9. 未決事項・継続検討

### 品質改善ロードマップ（候補）
1. データ追加：3時間 → 6時間（最大効果）
2. iter延長：5000 → 8000〜10000
3. LoRA rank拡大：32 → 64
4. 自動評価＋自動選別の実装（WER + 話者類似度 + MOS）
5. 手動データクリーニング（ノイズセグメント除外）

### 運用フェーズ移行
- voxcpm-personal Spaceへの LoRA 組み込み
- ユーザー受け入れフロー（一般公開向け）の設計
- 法的・倫理的同意プロセスの実装
- 課金システム連携

---

## 10. 関連ファイル

```
products/huggingface-voice-clone/
├── service-spec.md         # サービス要件定義（再作成予定）
├── tech-spec.md            # 技術仕様詳細
├── experiment-log.md       # 本ファイル
└── jobs/
    ├── prepare_data.py     # データ準備ジョブ
    ├── train_lora.py       # LoRA学習ジョブ
    ├── test_inference.py   # 推論ジョブ（seed対応）
    ├── chunked_inference.py # チャンク分割ジョブ（廃止）
    └── test_env.py         # 環境チェック（初期検証用）
```

---

## 11. 参考リンク

- Dataset: https://huggingface.co/datasets/masayasu/ryuken-voice
- LoRA Model: https://huggingface.co/masayasu/ryuken-voice-lora
- 推論Space: https://huggingface.co/spaces/masayasu/voxcpm-personal
- VoxCPM公式: https://github.com/OpenBMB/VoxCPM
