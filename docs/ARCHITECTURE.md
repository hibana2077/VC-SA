# 專案架構概覽

## 🏗️ 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                        src/run.py                            │
│                     (主程式入口點)                            │
│  • 解析命令列參數                                             │
│  • 設置訓練環境                                               │
│  • 協調各模塊運作                                             │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
         ┌────────▼────────┐     ┌───────▼────────┐
         │   src/core/     │     │   src/utils/   │
         │   (核心模塊)    │     │   (工具模塊)   │
         └────────┬────────┘     └────────┬───────┘
                  │                       │
      ┌───────────┴───────────┐          │
      │                       │          │
┌─────▼──────┐      ┌────────▼─────┐   ┌▼──────────────┐
│ models.py  │      │   data.py    │   │   config.py   │
│            │      │              │   │               │
│ • ViT      │      │ • VideoRec   │   │ • parse_args  │
│ • GraphSAM │      │ • CSVAnno    │   │ • Config cls  │
│            │      │ • FrameCache │   │ • defaults    │
└────────────┘      │ • Dataset    │   └───────────────┘
                    │ • DataModule │   
                    └──────────────┘   ┌───────────────┐
                                       │video_utils.py │
                                       │               │
                                       │ • hash_path   │
                                       └───────────────┘
```

## 📦 模塊依賴關係

```
run.py
  ├── core.models
  │     ├── GraphSamplerActionModel (Lightning Module)
  │     │     ├── ViTTokenBackbone
  │     │     ├── FrameTokenCoSelector (from example.core)
  │     │     ├── GraphBasedMemBank (from example.core)
  │     │     └── Classification Head
  │     └── ViTTokenBackbone
  │           └── timm.ViT
  │
  ├── core.data
  │     ├── VideoDataModule (Lightning DataModule)
  │     │     ├── train/val/test datasets
  │     │     └── data loaders
  │     ├── SimpleVideoDataset
  │     │     ├── VideoCSVAnnotation
  │     │     ├── FrameCache
  │     │     └── transforms
  │     ├── FrameCache
  │     │     └── hash_path (from utils.video_utils)
  │     ├── VideoCSVAnnotation
  │     │     └── VideoRecord
  │     └── VideoRecord
  │
  └── utils.config
        ├── parse_args
        ├── TrainingConfig
        └── get_default_config
```

## 🔄 資料流程

```
                         訓練流程
                            │
    ┌───────────────────────▼───────────────────────┐
    │            1. 載入配置 (parse_args)            │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │      2. 初始化資料模塊 (VideoDataModule)       │
    │         • 讀取 CSV 標註                        │
    │         • 建立 FrameCache                      │
    │         • 準備 DataLoaders                     │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │    3. 初始化模型 (GraphSamplerActionModel)     │
    │         • 載入 ViT backbone                    │
    │         • 初始化 co-selector                   │
    │         • 初始化 graph memory                  │
    │         • 建立分類頭                           │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │        4. 設置 Trainer 和 Callbacks            │
    │         • ModelCheckpoint                      │
    │         • LearningRateMonitor                  │
    │         • Logger                               │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │              5. 開始訓練循環                   │
    │                                                │
    │  每個 Batch:                                   │
    │  Video → Frames → ViT → Tokens → Co-Select    │
    │       → Graph Memory → Pooling → Classify     │
    │                                                │
    │  每個 Epoch:                                   │
    │  Train → Validate → (Optional) Test           │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │           6. 最終測試和保存結果                │
    └───────────────────────────────────────────────┘
```

## 🎯 模型前向傳播流程

```
Input Video Clip [B, T, C, H, W]
        │
        ├─► Reshape to [B*T, C, H, W]
        │
        ▼
┌───────────────────┐
│  ViT Backbone     │  Extract patch tokens
│  (timm.ViT)       │  
└────────┬──────────┘
         │ [B*T, N, D]
         ▼
┌───────────────────┐
│  Reshape          │  [B, T, N, D]
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Co-Selector       │  Select top-K frames and tokens
│ (FrameToken       │  
│  CoSelector)      │  
└────────┬──────────┘
         │ [B, T', M, D]
         ▼              T' = frame_topk
┌───────────────────┐   M = token_topk
│ Graph Memory      │  Build temporal graphs
│ (GraphBased       │  Apply graph convolutions
│  MemBank)         │  
└────────┬──────────┘
         │ [B, T', M, D]
         ▼
┌───────────────────┐
│ Global Pooling    │  Mean over time & tokens
│ mean(dim=(1,2))   │  
└────────┬──────────┘
         │ [B, D]
         ▼
┌───────────────────┐
│ Classification    │  LayerNorm + Linear
│ Head              │  
└────────┬──────────┘
         │
         ▼
    Logits [B, num_classes]
```

## 📊 類別關係圖

```
Lightning Components:
┌──────────────────────────────────────┐
│     GraphSamplerActionModel          │
│     (LightningModule)                │
│  ┌────────────────────────────────┐  │
│  │ ViTTokenBackbone               │  │
│  │   └── timm.ViT                 │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ FrameTokenCoSelector           │  │
│  │   (from example.core)          │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ GraphBasedMemBank              │  │
│  │   (from example.core)          │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ Classification Head            │  │
│  │   (LayerNorm + Linear)         │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│     VideoDataModule                  │
│     (LightningDataModule)            │
│  ┌────────────────────────────────┐  │
│  │ SimpleVideoDataset (train)     │  │
│  ├────────────────────────────────┤  │
│  │ SimpleVideoDataset (val)       │  │
│  ├────────────────────────────────┤  │
│  │ SimpleVideoDataset (test)      │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ FrameCache (shared)            │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘

Data Components:
┌──────────────────────────────────────┐
│     SimpleVideoDataset               │
│     (torch.utils.data.Dataset)       │
│  ┌────────────────────────────────┐  │
│  │ VideoCSVAnnotation             │  │
│  │   └── List[VideoRecord]        │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ FrameCache                     │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ transforms.Compose             │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## 🔌 外部依賴

```
PyTorch Stack:
  • torch (核心張量運算)
  • torchvision (視頻 I/O 和轉換)
  • lightning (訓練框架)

Vision Models:
  • timm (Vision Transformer 模型)

Custom:
  • example.core (自定義的選擇器和記憶模塊)
    ├── FrameTokenCoSelector
    └── GraphBasedMemBank
```

## 🚀 執行流程總結

1. **初始化階段**
   - 解析命令列參數
   - 設置隨機種子
   - 創建輸出目錄

2. **資料準備階段**
   - 載入 CSV 標註
   - 建立視頻索引
   - 初始化快取系統

3. **模型構建階段**
   - 載入預訓練 ViT
   - 組裝完整模型
   - 設置優化器和調度器

4. **訓練階段**
   - 批次資料載入
   - 前向傳播
   - 損失計算
   - 反向傳播
   - 參數更新

5. **評估階段**
   - 驗證集評估
   - (可選) 測試集評估
   - 模型檢查點保存

6. **完成階段**
   - 最終測試
   - 保存結果
   - 清理資源
