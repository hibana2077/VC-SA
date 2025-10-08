# Video Action Recognition - Source Code Structure

本專案已完成模塊化重構，將原本的單一檔案拆分為清晰的模塊結構。

## 📁 目錄結構

```
src/
├── run.py              # 主程式入口點
├── core/               # 核心模塊 (Core modules)
│   ├── __init__.py    # 模塊導出
│   ├── models.py      # 模型架構定義
│   └── data.py        # 資料處理模塊
└── utils/             # 工具模塊 (Utility modules)
    ├── __init__.py    # 模塊導出
    ├── config.py      # 配置管理
    └── video_utils.py # 視頻處理工具
```

## 🎯 模塊說明

### Core Modules (核心功能)

#### `core/models.py`
包含模型架構的核心組件：

- **`ViTTokenBackbone`**: Vision Transformer 骨幹網絡
  - 封裝 timm ViT 模型
  - 輸出 patch tokens (移除 CLS token)
  - 支持凍結參數以進行遷移學習

- **`GraphSamplerActionModel`**: 主要的動作識別模型
  - 整合 ViT backbone、frame/token 選擇器和圖記憶網絡
  - 實現完整的訓練/驗證/測試流程
  - 支持 PyTorch Lightning 的所有功能

#### `core/data.py`
資料處理相關的類別：

- **`VideoRecord`**: 視頻記錄容器
  - 簡單的資料結構，存儲視頻路徑和標籤

- **`VideoCSVAnnotation`**: CSV 標註解析器
  - 解析視頻標註檔案
  - 支持相對/絕對路徑
  - 驗證資料格式

- **`FrameCache`**: 幀快取系統
  - 將解碼的視頻幀存儲為 JPEG
  - 大幅提升重複訪問速度
  - 自動管理快取目錄

- **`SimpleVideoDataset`**: PyTorch 資料集
  - 統一時間採樣
  - 支持快取和即時解碼
  - 可配置的資料增強

- **`VideoDataModule`**: Lightning 資料模塊
  - 管理訓練/驗證/測試資料集
  - 統一的 DataLoader 配置

### Utils Modules (工具函數)

#### `utils/config.py`
配置管理模塊：

- **`TrainingConfig`**: 訓練配置類別
  - 結構化的配置介面
  - 參數驗證

- **`parse_args()`**: 命令列參數解析
  - 完整的參數文檔
  - 分組的參數選項
  - 預設值管理

- **`get_default_config()`**: 獲取預設配置
  - 用於程式化配置

#### `utils/video_utils.py`
視頻處理工具函數：

- **`hash_path()`**: 路徑哈希函數
  - 為快取生成唯一目錄名

## 🚀 使用方式

### 基本訓練

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

### 程式化使用

```python
from src.core import GraphSamplerActionModel, VideoDataModule
from src.utils import get_default_config

# 創建配置
config = get_default_config()
config.update({
    'num_classes': 400,
    'frames_per_clip': 16,
    # ... 其他參數
})

# 初始化模型
model = GraphSamplerActionModel(**config)

# 初始化資料模塊
datamodule = VideoDataModule(
    data_root='data/videos',
    train_csv='train.csv',
    val_csv='val.csv',
    test_csv='test.csv',
    frames_per_clip=16,
    batch_size=2,
)
```

## 📊 CSV 格式

標註 CSV 檔案格式：

```csv
video_path,label
/path/to/video1.mp4,0
relative/path/video2.avi,1
...
```

- `video_path`: 視頻檔案路徑（可以是絕對路徑或相對於 `--data-root` 的路徑）
- `label`: 整數類別 ID，範圍 [0, num_classes-1]

## 🔧 模塊化優勢

1. **清晰的職責分離**
   - 模型架構與資料處理分離
   - 核心功能與工具函數分離

2. **易於擴展**
   - 可輕鬆添加新的模型組件
   - 可獨立擴展資料處理功能

3. **程式碼重用**
   - 模塊可以在其他專案中重用
   - 更容易進行單元測試

4. **維護性提升**
   - 更小的檔案更容易理解
   - 問題定位更加容易

5. **文檔化**
   - 每個模塊都有清晰的文檔字串
   - 類別和函數都有詳細說明

## 📝 注意事項

- Lightning 相關的匯入錯誤是正常的 IDE 提示，執行時會正確匯入
- 確保已安裝所有必要的依賴套件（見 `requirements.txt`）
- 建議使用 GPU 進行訓練以獲得更好的性能

## 🔗 相關檔案

- `example/core.py`: FrameTokenCoSelector 和 GraphBasedMemBank 的實現
- `requirements.txt`: 專案依賴列表
- `docs/idea.md`: 專案理念和設計思路
