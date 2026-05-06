# Voice Clone — 引き継ぎドキュメント

## はじめに（りゅうけんさんへ）

このドキュメントは、やまもとりゅうけん氏の音声クローンシステムをまるごとお渡しするための引き継ぎ書です。

もともとりゅうけんさんの声をクローンする必要があり、VoxCPM2（中国・OpenBMBの音声合成モデル）と HuggingFace の GPU ジョブ環境を組み合わせてプロトタイプを構築しました。りゅうけんさんの音声約3時間49分を学習させ、テキストから本人ボイスを生成できる状態まで検証済みです（累計コスト約$4.70）。

本来であればりゅうけんさんの環境に一からセットアップするところですが、HuggingFace のアカウントごとお渡しするのが最もシンプルで工数もかからないと判断しました。学習済みモデル・スクリプト・動作環境がすべてこのアカウントに載っています。

このプロジェクトは私（鈴木）から完全に手放します。今後の運用・改良はご自由にどうぞ。

なお、Pro プラン（$9/月）は **6月1日まで有効** で、**約$14のクレジットが残っています**。クレジットカードの登録なしでもしばらく検証・試用いただける状態です。本格運用に入る前に支払い方法の登録をお願いします。

## 引き継ぎアカウント情報

| 項目 | 値 |
|---|---|
| サービス | HuggingFace |
| アカウント名 | `masayasu` |
| 登録メールアドレス | `suzukimasayasu.mailadress@gmail.com` |
| ログインパスワード | （別途お伝えします） |
| プラン | Pro（$9/月、6月1日まで有効） |
| クレジット残高 | 約$14 |
| クレジットカード | 削除済み（引き継ぎ後に登録してください） |

## 引き継ぎ後の初回作業

1. パスワード変更: https://huggingface.co/settings
2. メールアドレス変更（任意）: 同上 Settings ページから
3. 支払い方法の登録: https://huggingface.co/settings/billing
   - Pro プラン（$9/月）+ GPU 従量課金（L4 $0.80/h）が必要です
   - クレジットカードを登録してください
4. API トークン発行: https://huggingface.co/settings/tokens
   - **Write 権限必須**。名前は任意（例: `cli`）

## アカウントにあるもの

| リポジトリ | 種類 | 中身 |
|---|---|---|
| `masayasu/ryuken-voice` | Dataset (Private) | 学習用音声セグメント、ジョブスクリプト、推論結果 |
| `masayasu/ryuken-voice-lora` | Model (Private) | 学習済みLoRA重み（5000iter） |
| `masayasu/voxcpm-personal` | Space (Private) | 推論UI（現在停止中） |

学習済みの声は **やまもとりゅうけん氏** のものです。別の声をクローンしたい場合は新しく学習データを用意して学習し直す必要があります。

## 必要なもの

- Python 3.11+
- HuggingFace CLI: `pip install huggingface_hub[cli]`
- 認証: `hf auth login --token hf_あなたのトークン`
- 元音声ファイル（新しく学習する場合のみ）

## 推論の実行（既存のりゅうけんボイスでテキスト読み上げ）

```bash
hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --env RUN_NAME=ryuken-lora-full-5k \
  --env LORA_STEP=latest \
  --env OUTPUT_TAG=my_test \
  --env MAX_LEN=2000 \
  --env CFG_VALUE=2.0 \
  --env INFERENCE_TIMESTEPS=50 \
  --env SEED=42 \
  --env "TARGET_TEXT=ここに生成したい日本語テキストを入れる" \
  --timeout 30m -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/test_inference.py"
```

生成された音声は `masayasu/ryuken-voice` の `inference_tests/` に保存されます。

### パラメータの意味

| パラメータ | 推奨値 | 説明 |
|---|---|---|
| CFG_VALUE | 2.0 | 高いほど本人らしさ強い。2.5超えると不自然 |
| INFERENCE_TIMESTEPS | 50 | 拡散ステップ数。高いほど滑らかだが時間かかる |
| MAX_LEN | 1500-2000 | テキスト長に応じて調整 |
| SEED | 任意 | テキストごとに変えると結果が変わる。ガチャ推奨 |
| normalize | True | 数字・略語の読み上げ正規化。日本語では必須 |

### シードガチャのやり方

同じテキストでも seed を変えると違う音声になります。
`SEED=42` の部分を変えて複数回実行し、一番良いものを選んでください。

## 新しい声を学習させる場合

3ステップです。元音声は **3時間以上のクリアな発話** を推奨。

### Step 1: 元音声の前処理（ローカル）

```bash
ffmpeg -y -i 元音声.wav -ac 1 -ar 16000 -c:a flac -compression_level 8 sample_16k_mono.flac
```

### Step 2: データ準備（HF Job）

前処理した FLAC を Dataset にアップロードしてから実行:

```bash
hf upload masayasu/ryuken-voice sample_16k_mono.flac raw/sample_16k_mono.flac --repo-type dataset

hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --timeout 2h -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/prepare_data.py"
```

所要: 約50分 / コスト: 約$0.65

### Step 3: LoRA学習（HF Job）

```bash
hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --env RUN_NAME=my-new-voice \
  --env NUM_ITERS=5000 \
  --env SAVE_INTERVAL=1000 \
  --timeout 5h -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/train_lora.py"
```

所要: 約3時間 / コスト: 約$2.40

学習後は `masayasu/ryuken-voice-lora/my-new-voice/` に LoRA 重みが保存されます。
推論時は `RUN_NAME=my-new-voice` に変更してください。

## ジョブ管理コマンド

```bash
hf jobs ps -a              # 全ジョブ一覧
hf jobs inspect <JOB_ID>   # ジョブ詳細
hf jobs logs <JOB_ID>      # ログ確認
hf jobs cancel <JOB_ID>    # キャンセル
```

## コスト目安

| 作業 | 時間 | コスト |
|---|---|---|
| データ準備 | 50分 | $0.65 |
| LoRA学習（5000iter） | 3時間 | $2.40 |
| 推論（1テキスト） | 3〜5分 | $0.05 |
| Proプラン（月額） | - | $9.00 |

## 技術スタック

- ベースモデル: [OpenBMB/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)（Apache-2.0、商用可）
- 学習手法: LoRA (Low-Rank Adaptation)
- ASR: faster-whisper large-v3 + silero VAD
- GPU: Nvidia L4（24GB VRAM）

## 注意点

- ジョブスクリプトを変更したら、必ず Dataset に再アップロードしてから実行してください
- T4 GPU は非対応です（FlashAttention が使えない）。必ず L4 以上を指定してください
- 短時間にジョブを大量投入するとレートリミットに引っかかります
- 学習データの権利・同意は自己責任で確認してください

## 参考リンク

- HF Billing: https://huggingface.co/settings/billing
- Dataset: https://huggingface.co/datasets/masayasu/ryuken-voice
- Model: https://huggingface.co/masayasu/ryuken-voice-lora
- Space: https://huggingface.co/spaces/masayasu/voxcpm-personal
- VoxCPM GitHub: https://github.com/OpenBMB/VoxCPM
- HF Jobs CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cli
