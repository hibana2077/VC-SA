# Data Module 结构说明

本目录包含视频动作识别的数据处理模块，已重构为模块化结构。

## 📁 目录结构

```
data/
├── __init__.py          # 包初始化和公共接口导出
├── record.py            # VideoRecord: 视频记录容器
├── annotation.py        # VideoCSVAnnotation: CSV注解解析器
├── cache.py             # FrameCache: 帧缓存系统
├── dataset.py           # SimpleVideoDataset: PyTorch数据集
├── datamodule.py        # VideoDataModule: Lightning数据模块
├── preparation.py       # 数据集准备工具函数
└── README.md            # 本文件
```

## 📦 模块说明

### `record.py`
- **类**: `VideoRecord`
- **用途**: 简单的视频记录容器，存储视频路径和标签
- **属性**:
  - `path`: 视频文件路径
  - `label`: 整数类别标签

### `annotation.py`
- **类**: `VideoCSVAnnotation`
- **用途**: 解析CSV格式的视频注解文件
- **功能**:
  - 从CSV文件加载视频路径和标签
  - 支持相对路径解析
  - 提供索引访问接口

### `cache.py`
- **类**: `FrameCache`
- **用途**: 基于磁盘的视频帧缓存系统
- **功能**:
  - 将解码的视频帧缓存为JPEG文件
  - 加速重复访问
  - 使用路径哈希管理缓存目录

### `dataset.py`
- **类**: `SimpleVideoDataset`
- **用途**: PyTorch视频数据集实现
- **功能**:
  - 加载视频并均匀采样帧
  - 应用图像变换和归一化
  - 支持多后端视频解码（torchvision、PyAV、OpenCV）
  - 可选的帧缓存支持

### `datamodule.py`
- **类**: `VideoDataModule`
- **用途**: PyTorch Lightning数据模块
- **功能**:
  - 统一管理训练/验证/测试数据集
  - 自动创建DataLoader
  - 配置批处理和多进程加载

### `preparation.py`
- **函数**:
  - `prepare_hmdb51_annotations()`: 准备HMDB51数据集注解
  - `prepare_diving48_annotations()`: 准备Diving48数据集注解
  - `create_datamodule_for()`: 为内置数据集创建DataModule的工厂函数
- **用途**: 数据集准备和注解生成工具

## 🔄 向后兼容性

原来的 `data.py` 文件已更新为兼容层，所有现有代码无需修改即可继续工作：

```python
# 旧的导入方式（仍然有效）
from src.core.data import VideoRecord, VideoDataModule

# 新的导入方式（推荐）
from src.core.data import VideoRecord, VideoDataModule
# 或者更具体地：
from src.core.data.record import VideoRecord
from src.core.data.datamodule import VideoDataModule
```

## 📚 使用示例

### 基本使用

```python
from src.core.data import (
    VideoCSVAnnotation,
    FrameCache,
    SimpleVideoDataset,
    VideoDataModule,
)

# 创建注解
anno = VideoCSVAnnotation('train.csv', data_root='/path/to/videos')

# 创建缓存
cache = FrameCache(cache_root='./frame_cache')

# 创建数据集
dataset = SimpleVideoDataset(
    anno=anno,
    num_frames=16,
    frame_cache=cache,
    is_train=True,
)

# 创建数据模块
dm = VideoDataModule(
    data_root='/path/to/videos',
    train_csv='train.csv',
    val_csv='val.csv',
    test_csv='test.csv',
    frames_per_clip=16,
    batch_size=8,
)
```

### 使用内置数据集

```python
from src.core.data import create_datamodule_for

# HMDB51
dm = create_datamodule_for(
    dataset='hmdb51',
    root_dir='/path/to/hmdb51',
    frames_per_clip=16,
    batch_size=8,
)

# Diving48
dm = create_datamodule_for(
    dataset='diving48',
    root_dir='/path/to/diving48/rgb',
    frames_per_clip=16,
    batch_size=8,
)
```

## 🎯 设计优势

1. **模块化**: 每个类独立文件，便于维护和测试
2. **清晰职责**: 每个模块有明确的单一职责
3. **易于扩展**: 添加新功能无需修改大文件
4. **向后兼容**: 现有代码无需改动
5. **更好的IDE支持**: 更精确的代码补全和导航

## 🔧 开发建议

- 新代码应使用具体的模块导入
- 修改某个类时只需关注对应的文件
- 添加新数据集支持时扩展 `preparation.py`
- 单元测试可以针对每个模块独立编写
