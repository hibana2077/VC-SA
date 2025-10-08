# VC-SA - Video Action Recognition with Graph-based Sampling

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Code style: modular](https://img.shields.io/badge/code%20style-modular-green.svg)](docs/ARCHITECTURE.md)

高效的視頻動作識別框架，採用模塊化設計，結合 Vision Transformer、幀/Token 智能選擇和圖記憶網絡。

## ✨ 特色

- 🎯 **智能採樣**: 自動選擇最具信息量的幀和 Token
- 🔗 **圖記憶網絡**: 基於圖結構的時序建模
- ⚡ **高效訓練**: 支持混合精度、梯度累積和分佈式訓練
- 🧩 **模塊化設計**: 清晰的程式碼結構，易於擴展和維護
- 📊 **完整流程**: 從資料載入到模型訓練的端到端解決方案

## 📁 專案結構

```
VC-SA/
├── src/
│   ├── run.py              # 主程式入口
│   ├── core/               # 核心模塊
│   │   ├── models.py       # 模型架構
│   │   └── data.py         # 資料處理
│   └── utils/              # 工具函數
│       ├── config.py       # 配置管理
│       └── video_utils.py  # 視頻工具
├── example/
│   └── core.py             # FrameTokenCoSelector & GraphBasedMemBank
├── docs/
│   ├── idea.md             # 設計理念
│   ├── ARCHITECTURE.md     # 架構文檔
│   └── MIGRATION.md        # 遷移指南
└── requirements.txt        # 依賴列表
```

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 準備資料

建立 CSV 標註檔案（train.csv, val.csv, test.csv）：

```csv
video_path,label
/path/to/video1.mp4,0
/path/to/video2.avi,1
```

### 訓練模型

```bash
python -m src.run \
    --data-root data/videos \
    --train-anno train.csv \
    --val-anno val.csv \
    --test-anno test.csv \
    --num-classes 400 \
    --frames-per-clip 16 \
    --frame-topk 8 \
    --token-topk 32 \
    --batch-size 2 \
    --max-epochs 50
```

### Python API 使用

```python
from src.core import GraphSamplerActionModel, VideoDataModule
from src.utils import get_default_config

# 建立資料模塊
datamodule = VideoDataModule(
    data_root='data/videos',
    train_csv='train.csv',
    val_csv='val.csv',
    test_csv='test.csv',
    frames_per_clip=16,
    batch_size=2,
)

# 建立模型
model = GraphSamplerActionModel(
    num_classes=400,
    frames_per_clip=16,
    frame_topk=8,
    token_topk=32,
)

# 訓練（使用 PyTorch Lightning）
import lightning as L
trainer = L.Trainer(max_epochs=50)
trainer.fit(model, datamodule=datamodule)
```

## 📚 文檔

- [架構概覽](docs/ARCHITECTURE.md) - 詳細的系統架構說明
- [遷移指南](docs/MIGRATION.md) - 從舊版本遷移的指南
- [程式碼文檔](src/README.md) - 模塊化程式碼說明
- [設計理念](docs/idea.md) - 專案設計思路

## 🔧 主要模塊

### Core Modules

- **models.py**: 模型架構定義
  - `ViTTokenBackbone`: Vision Transformer 骨幹網絡
  - `GraphSamplerActionModel`: 完整的動作識別模型

- **data.py**: 資料處理模塊
  - `VideoDataModule`: Lightning 資料模塊
  - `SimpleVideoDataset`: 視頻資料集
  - `FrameCache`: 幀快取系統

### Utils Modules

- **config.py**: 配置管理
  - `parse_args()`: 命令列參數解析
  - `TrainingConfig`: 配置類別
  - `get_default_config()`: 獲取預設配置

- **video_utils.py**: 視頻處理工具
  - `hash_path()`: 路徑哈希函數

## 🎓 引用

如果這個專案對您的研究有幫助，請考慮引用。

## 📄 授權

[在此添加授權資訊]

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📧 聯絡

[在此添加聯絡資訊]