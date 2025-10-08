# Dataset 格式需求說明

## 📋 CSV 標註格式

本專案使用 CSV 格式的標註檔案來組織視頻資料集。

### 基本格式

```csv
video_path,label
/path/to/video1.mp4,0
/path/to/video2.avi,1
relative/path/video3.mkv,2
```

### 格式規範

| 欄位 | 說明 | 類型 | 必填 |
|------|------|------|------|
| `video_path` | 視頻檔案路徑 | string | ✅ 是 |
| `label` | 動作類別 ID | integer | ✅ 是 |

## 📝 詳細說明

### 1. 視頻路徑 (video_path)

視頻路徑可以使用以下兩種方式：

#### 絕對路徑
```csv
video_path,label
/home/user/videos/action1.mp4,0
C:\Users\videos\action2.avi,1
```

#### 相對路徑（推薦）
```csv
video_path,label
train/class_0/video001.mp4,0
train/class_1/video002.mp4,1
```

當使用相對路徑時，需要在執行時指定 `--data-root` 參數：

```bash
python -m src.run --data-root /path/to/videos --train-anno train.csv ...
```

系統會自動將路徑組合為：`/path/to/videos/train/class_0/video001.mp4`

### 2. 標籤 (label)

- **類型**: 整數 (integer)
- **範圍**: `[0, num_classes-1]`
- **說明**: 
  - 標籤必須是從 0 開始的連續整數
  - 最大值必須是 `num_classes - 1`
  - 例如：100 個類別，標籤範圍是 0-99

**示例：**

```csv
# 正確 ✅ - 10 個類別 (0-9)
video_path,label
video1.mp4,0
video2.mp4,1
video3.mp4,9

# 錯誤 ❌ - 標籤超出範圍
video_path,label
video1.mp4,0
video2.mp4,10  # 錯誤：如果只有 10 個類別，最大應該是 9
```

## 📁 檔案組織建議

### 方案一：按類別組織（推薦）

```
data/
├── videos/
│   ├── train/
│   │   ├── class_0/
│   │   │   ├── video001.mp4
│   │   │   ├── video002.mp4
│   │   │   └── ...
│   │   ├── class_1/
│   │   │   ├── video001.mp4
│   │   │   └── ...
│   │   └── ...
│   ├── val/
│   │   ├── class_0/
│   │   └── ...
│   └── test/
│       ├── class_0/
│       └── ...
├── train.csv
├── val.csv
└── test.csv
```

**train.csv:**
```csv
video_path,label
train/class_0/video001.mp4,0
train/class_0/video002.mp4,0
train/class_1/video001.mp4,1
train/class_1/video002.mp4,1
```

### 方案二：扁平結構

```
data/
├── videos/
│   ├── video001.mp4
│   ├── video002.mp4
│   ├── video003.mp4
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

**train.csv:**
```csv
video_path,label
video001.mp4,0
video002.mp4,0
video003.mp4,1
video004.mp4,1
```

## 🎬 支援的視頻格式

系統使用 `torchvision.io.read_video` 讀取視頻，支援常見格式：

- ✅ `.mp4` (推薦)
- ✅ `.avi`
- ✅ `.mkv`
- ✅ `.mov`
- ✅ `.webm`
- ✅ `.flv`

## 📋 CSV 進階功能

### 1. 註解行

使用 `#` 開頭的行會被自動跳過：

```csv
# 這是註解行，會被忽略
# video_path,label
video1.mp4,0
# 這也是註解
video2.mp4,1
```

### 2. 空行處理

空行會被自動跳過：

```csv
video_path,label
video1.mp4,0

video2.mp4,1

video3.mp4,2
```

### 3. 編碼格式

CSV 檔案應使用 **UTF-8** 編碼，以支援包含非 ASCII 字元的路徑。

## ✅ 範例：完整的資料集配置

### 目錄結構

```
my_dataset/
├── videos/
│   ├── train/
│   │   ├── running/
│   │   │   ├── run_001.mp4
│   │   │   └── run_002.mp4
│   │   ├── walking/
│   │   │   ├── walk_001.mp4
│   │   │   └── walk_002.mp4
│   │   └── jumping/
│   │       ├── jump_001.mp4
│   │       └── jump_002.mp4
│   ├── val/
│   │   ├── running/
│   │   ├── walking/
│   │   └── jumping/
│   └── test/
│       ├── running/
│       ├── walking/
│       └── jumping/
├── annotations/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
```

### train.csv

```csv
video_path,label
train/running/run_001.mp4,0
train/running/run_002.mp4,0
train/walking/walk_001.mp4,1
train/walking/walk_002.mp4,1
train/jumping/jump_001.mp4,2
train/jumping/jump_002.mp4,2
```

### val.csv

```csv
video_path,label
val/running/run_001.mp4,0
val/walking/walk_001.mp4,1
val/jumping/jump_001.mp4,2
```

### test.csv

```csv
video_path,label
test/running/run_001.mp4,0
test/walking/walk_001.mp4,1
test/jumping/jump_001.mp4,2
```

### 執行命令

```bash
python -m src.run \
    --data-root my_dataset/videos \
    --train-anno my_dataset/annotations/train.csv \
    --val-anno my_dataset/annotations/val.csv \
    --test-anno my_dataset/annotations/test.csv \
    --num-classes 3 \
    --frames-per-clip 16 \
    --batch-size 4 \
    --max-epochs 50
```

## 🔧 程式化建立 CSV

### Python 腳本範例

```python
import os
import csv
from pathlib import Path

def create_annotation_csv(video_dir, output_csv, class_to_idx):
    """
    建立 CSV 標註檔案
    
    Args:
        video_dir: 視頻目錄
        output_csv: 輸出 CSV 路徑
        class_to_idx: 類別名稱到索引的映射，例如 {'running': 0, 'walking': 1}
    """
    records = []
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = Path(video_dir) / class_name
        if not class_dir.exists():
            continue
            
        for video_file in class_dir.glob('*.mp4'):
            # 使用相對路徑
            rel_path = f"{class_name}/{video_file.name}"
            records.append((rel_path, class_idx))
    
    # 寫入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label'])
        writer.writerows(records)
    
    print(f"Created {output_csv} with {len(records)} videos")

# 使用範例
class_to_idx = {
    'running': 0,
    'walking': 1,
    'jumping': 2
}

create_annotation_csv('my_dataset/videos/train', 'train.csv', class_to_idx)
create_annotation_csv('my_dataset/videos/val', 'val.csv', class_to_idx)
create_annotation_csv('my_dataset/videos/test', 'test.csv', class_to_idx)
```

## ⚠️ 常見錯誤

### 1. 標籤格式錯誤

```csv
# 錯誤 ❌ - 標籤是字串
video_path,label
video1.mp4,running

# 正確 ✅ - 標籤是整數
video_path,label
video1.mp4,0
```

### 2. 路徑不存在

```csv
# 確保視頻檔案存在
video_path,label
non_existent_video.mp4,0  # ❌ 會在載入時報錯
```

### 3. 標籤範圍錯誤

```bash
# 命令列指定 --num-classes 10
# 但 CSV 中有標籤 10，超出範圍 [0, 9]

# 錯誤示例：
video_path,label
video1.mp4,10  # ❌ 錯誤：標籤應該是 0-9
```

### 4. CSV 分隔符錯誤

```csv
# 錯誤 ❌ - 使用分號
video_path;label
video1.mp4;0

# 正確 ✅ - 使用逗號
video_path,label
video1.mp4,0
```

## 🧪 驗證 CSV 格式

### 快速驗證腳本

```python
import csv
import os
from pathlib import Path

def validate_csv(csv_path, data_root=None, num_classes=None):
    """驗證 CSV 標註檔案格式"""
    errors = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if not row or row[0].startswith('#'):
                continue
            
            if len(row) < 2:
                errors.append(f"Line {i}: 缺少必要欄位")
                continue
            
            video_path, label_str = row[0], row[1]
            
            # 檢查標籤格式
            try:
                label = int(label_str)
            except ValueError:
                errors.append(f"Line {i}: 標籤 '{label_str}' 不是整數")
                continue
            
            # 檢查標籤範圍
            if num_classes and (label < 0 or label >= num_classes):
                errors.append(f"Line {i}: 標籤 {label} 超出範圍 [0, {num_classes-1}]")
            
            # 檢查檔案是否存在
            if data_root:
                full_path = os.path.join(data_root, video_path) if not os.path.isabs(video_path) else video_path
                if not os.path.exists(full_path):
                    errors.append(f"Line {i}: 視頻檔案不存在: {full_path}")
    
    if errors:
        print(f"❌ 發現 {len(errors)} 個錯誤：")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✅ CSV 格式驗證通過")
        return True

# 使用範例
validate_csv('train.csv', data_root='my_dataset/videos', num_classes=3)
```

## 📚 相關資源

- [資料處理模塊文檔](../src/README.md#coredatapy)
- [完整使用範例](../README.md#-快速開始)
- [架構說明](ARCHITECTURE.md)

## 💡 最佳實踐

1. ✅ 使用相對路徑 + `--data-root` 參數
2. ✅ 按類別組織視頻檔案
3. ✅ 使用 UTF-8 編碼保存 CSV
4. ✅ 在訓練前驗證 CSV 格式
5. ✅ 保持標籤連續（0, 1, 2, ...）
6. ✅ 使用 `.mp4` 格式以獲得最佳兼容性
7. ✅ 添加註解行記錄類別對應關係

```csv
# Class mapping:
# 0: running
# 1: walking
# 2: jumping
video_path,label
train/running/run_001.mp4,0
train/walking/walk_001.mp4,1
train/jumping/jump_001.mp4,2
```
