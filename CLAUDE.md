# huggingface-voice-clone — 作業ファイル説明

## 必須ルール（チャクラム＝Claude向け）

**作業のたびに記録を更新すること。**

何か実行・実験・決定したら、その都度：

1. **`experiment-log.md` に追記**：実行内容・結果・気づき・採用/廃止判断
   - 失敗実験も書く（同じ轍を踏まないため）
   - 設定値・コスト・所要時間を含める
2. **`CLAUDE.md`（本ファイル）を更新**：
   - 「確定パラメータ」セクションが古くなったら書き換え
   - 「次にやるべきこと」セクションを進捗に合わせて差し替え
   - 新しい既知問題があれば追記
3. **`tech-spec.md` を更新**：技術仕様の変更（新しいパラメータ・スクリプト）があれば反映

**会話の流れで決まったことを保存しないと、次セッションで全部失われる**。義務として徹底すること。

セッション終了時は最低限以下の状態を保証：
- 今日試したことが experiment-log.md に書かれている
- 今日採用に変えたパラメータが CLAUDE.md に反映されている
- 廃止した方針があれば明示されている

---

## 次回起動方法

### 1. ターミナルでClaude Code起動
```bash
cd /Users/suzukimotoyasumain/Desktop/MASAYASU/products/huggingface-voice-clone
claude
```
そして冒頭で以下と打つ：
```
このディレクトリのCLAUDE.mdとexperiment-log.mdを読んで、現状を把握して
```

### 2. 認証チェック（必要時のみ）
```bash
/Users/suzukimotoyasumain/Library/Python/3.11/bin/hf auth whoami
```
→ `user=masayasu` が出ればOK。出ない／期限切れの場合：
```bash
/Users/suzukimotoyasumain/Library/Python/3.11/bin/hf auth login --token hf_xxx
```
トークンは `https://huggingface.co/settings/tokens` で発行。**Write権限必須**。

### 3. 状態確認URL
- 残高：https://huggingface.co/settings/billing
- 学習済みLoRA：https://huggingface.co/masayasu/ryuken-voice-lora
- 過去サンプル：https://huggingface.co/datasets/masayasu/ryuken-voice/tree/main/inference_tests
- 推論Space（停止中）：https://huggingface.co/spaces/masayasu/voxcpm-personal

### 4. よく使う依頼パターン

| やりたいこと | 開始の合言葉 |
|---|---|
| 新テキストで音声生成 | 「[テキスト]をseed [N]で生成して」 |
| seedガチャで当たり探し | 「seed [N1,N2,N3]で生成」 |
| 品質向上の追加学習 | 「追加データで再学習したい」 |
| Spaceに組み込み | 「voxcpm-personalにLoRA組み込んで」 |
| 自動評価実装 | 「seed自動選別を実装して」 |

### 5. 作業終了時の保守ルール

- 結果や決定事項は `experiment-log.md` に追記
- 方針変更があれば `CLAUDE.md` の「確定パラメータ」「次にやるべきこと」を更新
- 終わったらSpaceは必ずPause（Spaceを使った場合）

---

## このディレクトリは何か

**HuggingFace Spaces + HF Jobs + VoxCPM2 をベースにした、日本語音声クローンサービスのプロダクト開発用ディレクトリ**。

最終目的：個人クリエイター・企業向けに「自分の声を3時間アップロードしたら、テキストから本人ボイスを生成できる」Webサービスを一般公開すること。

現在は **MVP検証フェーズ**。プロトタイプとして1名分（やまもとりゅうけん氏）の音声を学習させ、品質と運用パラメータを検証している。

---

## ファイル構成

```
huggingface-voice-clone/
├── CLAUDE.md           # 本ファイル（次のClaudeセッション用の入口）
├── service-spec.md     # サービス要件定義（誰に何を売るか・料金プラン・法的配慮）
├── tech-spec.md        # 技術仕様（パイプライン・パラメータ・運用コマンド・既知問題）
├── cost.md             # コスト分析（100字あたり推論単価・バッチ化効果・サブスク粗利）
├── experiment-log.md   # 実験ログ（時系列の試行錯誤記録）
└── jobs/               # HF Jobs用のPythonスクリプト
    ├── prepare_data.py       # 元音声 → セグメント分割 + ASR書き起こし + manifest作成
    ├── train_lora.py         # LoRA学習（VoxCPM2ベース、5000iterで$2.40）
    ├── test_inference.py     # 単発推論（CFG/timesteps/SEEDをenv指定）
    ├── chunked_inference.py  # 「。」分割→結合版（廃止、不自然な断絶のため）
    └── test_env.py           # 初期環境検証用（GPU/Mount/Auth確認）
```

---

## 重要な前提（次セッション開始時に必ず確認）

### HFアカウント
- ユーザー：`masayasu`
- プラン：HF Pro（$9/月）
- CLI：`/Users/suzukimotoyasumain/Library/Python/3.11/bin/hf`（PATHに無いのでフルパス推奨）
- 認証済み（`~/.cache/huggingface/token`）

### HF上のリソース
| 種別 | リポジトリ | 用途 |
|---|---|---|
| Dataset | `masayasu/ryuken-voice` | 学習データ + ジョブスクリプト + 推論結果 |
| Model | `masayasu/ryuken-voice-lora` | 学習済みLoRA重み |
| Space | `masayasu/voxcpm-personal` | 推論UI（現在Pause中） |
| Bucket | `masayasu/jobs-artifacts` | HF Jobs内部用（自動生成） |

### 確定した運用パラメータ
```
CFG_VALUE: 2.0         # 高い=本人らしさ強い、超えると不自然
INFERENCE_TIMESTEPS: 50  # 50がベスト、100以上は逆に平坦化
MAX_LEN: 1500-2000     # テキスト長に応じて
SEED: テキストごとに5-10本ガチャして決定（テキスト依存）
normalize: True        # 必須（数字・略語の正規化）
denoise: False         # LoRA使用時は不要
```

---

## ジョブ実行の典型的なコマンド

ローカルファイルではなく、 **DatasetにアップしたスクリプトのURL** を指定する（バケットマウント問題回避）。

```bash
# 推論
hf jobs uv run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --env RUN_NAME=ryuken-lora-full-5k \
  --env LORA_STEP=latest \
  --env OUTPUT_TAG=demo \
  --env CFG_VALUE=2.0 \
  --env INFERENCE_TIMESTEPS=50 \
  --env SEED=42 \
  --env "TARGET_TEXT=...日本語...." \
  --timeout 30m -d \
  "https://huggingface.co/datasets/masayasu/ryuken-voice/resolve/main/jobs/test_inference.py"
```

スクリプトを変更した場合は、 **必ずDatasetに再アップロードしてから** ジョブを叩く：

```bash
hf upload masayasu/ryuken-voice ./jobs/test_inference.py jobs/test_inference.py --repo-type dataset
```

---

## やってはいけないこと

- `products/VoxCPM/` に自作ファイルを足さない（あれはOpenBMB公式clone、git pull競合する）
- `chunked_inference.py` は廃止方針（境界の不自然さが解決できない）
- HF tokenをコミット・チャットに貼らない（既に1回貼ってしまっており、後でローテート予定）
- ジョブを大量に並列実行しない（`/whoami-v2` レートリミットに引っかかる）

---

## 既知の落とし穴（tech-spec.md §9 詳細）

1. ボリュームマウント間欠失敗 → URL指定で回避
2. cuBLAS/cuDNN/nvrtc 不足 → `nvidia-cublas-cu12` 等を deps に明示
3. manifest内の相対パス → 絶対パスに書き換えてから渡す
4. 学習スクリプト引数 → `--config_path` を使う（`--args.load` は無効）
5. レートリミット → 60〜180秒待つ

---

## 次にやるべきこと候補

1. **品質向上**：データ追加（3h→6h）、iter延長（5k→10k）、LoRA rank↑
2. **運用化**：voxcpm-personal Space に LoRA 組み込み、推論UI化
3. **自動評価**：Whisper-WER + 話者類似度で seed ガチャ自動選別
4. **公開準備**：認証・課金・利用規約・同意フロー
5. **チャンク分割の再設計**：境界の抑揚自然化（クロスフェード等）

詳細・優先度判断は `experiment-log.md §9` を参照。

---

## 関連ファイル（このディレクトリ外）

```
/Users/suzukimotoyasumain/Desktop/samplevoice/
├── sample_full.wav                  # 元音声 2.5GB（ローカルのみ）
└── processed/sample_16k_mono.flac   # 変換済み 228MB（HFにもUP済み）

/Users/suzukimotoyasumain/Desktop/MASAYASU/products/VoxCPM/
└── （OpenBMB公式clone、touchするな）
```
