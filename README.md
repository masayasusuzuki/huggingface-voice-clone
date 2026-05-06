# Voice Clone

VoxCPM2 + LoRA fine-tuning による日本語音声クローンシステム。

やまもとりゅうけん氏の声を約3時間49分学習させ、テキストから本人ボイスを生成できます。

## 引き継ぎについて

詳細は [HANDOVER.md](./HANDOVER.md) を参照してください。

## 構成

```
├── HANDOVER.md          # 引き継ぎドキュメント（使い方・コマンド集）
├── tech-spec.md         # 技術仕様（パイプライン・パラメータ・既知問題）
├── cost.md              # コスト分析（推論単価・バッチ化効果）
├── experiment-log.md    # 実験ログ（時系列の試行錯誤記録）
└── jobs/                # HF Jobs用Pythonスクリプト
    ├── prepare_data.py       # 元音声 → セグメント分割 + ASR書き起こし
    ├── train_lora.py         # LoRA学習（VoxCPM2ベース）
    ├── test_inference.py     # 単発推論
    ├── chunked_inference.py  # チャンク分割推論（廃止）
    └── test_env.py           # 環境検証用
```

## 技術スタック

- ベースモデル: [OpenBMB/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)（Apache-2.0）
- 学習手法: LoRA (Low-Rank Adaptation)
- 実行環境: HuggingFace Jobs（L4 GPU）
- ASR: faster-whisper large-v3 + silero VAD
