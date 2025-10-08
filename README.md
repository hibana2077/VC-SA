# VC-SA - Video Action Recognition with Graph-based Sampling

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Code style: modular](https://img.shields.io/badge/code%20style-modular-green.svg)](docs/ARCHITECTURE.md)

An efficient video action recognition framework with modular design, combining Vision Transformer, intelligent frame/token selection, and graph memory network.

## ✨ Features

- 🎯 **Smart Sampling**: Automatically selects the most informative frames and tokens
- 🔗 **Graph Memory Network**: Graph-based temporal modeling
- ⚡ **Efficient Training**: Supports mixed precision, gradient accumulation, and distributed training
- 🧩 **Modular Design**: Clear code structure, easy to extend and maintain
- 📊 **Complete Pipeline**: End-to-end solution from data loading to model training

## 📁 Project Structure

```
VC-SA/
├── src/
│   ├── run.py              # Main program entry
│   ├── core/               # Core modules
│   │   ├── models.py       # Model architectures
│   │   └── data.py         # Data processing
│   └── utils/              # Utility functions
│       ├── config.py       # Configuration management
│       └── video_utils.py  # Video utilities
├── example/
│   └── core.py             # FrameTokenCoSelector & GraphBasedMemBank
├── docs/
│   ├── idea.md             # Design philosophy
│   ├── ARCHITECTURE.md     # Architecture documentation
│   └── MIGRATION.md        # Migration guide
└── requirements.txt        # Dependency list
```

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Data

Create CSV annotation files (train.csv, val.csv, test.csv):

```csv
video_path,label
/path/to/video1.mp4,0
/path/to/video2.avi,1
```

### Train the Model

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

### Python API Usage

```python
from src.core import GraphSamplerActionModel, VideoDataModule
from src.utils import get_default_config

# Create data module
datamodule = VideoDataModule(
    data_root='data/videos',
    train_csv='train.csv',
    val_csv='val.csv',
    test_csv='test.csv',
    frames_per_clip=16,
    batch_size=2,
)

# Create model
model = GraphSamplerActionModel(
    num_classes=400,
    frames_per_clip=16,
    frame_topk=8,
    token_topk=32,
)

# Train (using PyTorch Lightning)
import lightning as L
trainer = L.Trainer(max_epochs=50)
trainer.fit(model, datamodule=datamodule)
```

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed system architecture description
- [Migration Guide](docs/MIGRATION.md) - Guide for migrating from older versions
- [Code Documentation](src/README.md) - Modular code explanation
- [Design Philosophy](docs/idea.md) - Project design ideas

## 🔧 Main Modules

### Core Modules

- **models.py**: Model architecture definitions
  - `ViTTokenBackbone`: Vision Transformer backbone network
  - `GraphSamplerActionModel`: Complete action recognition model

- **data.py**: Data processing module
  - `VideoDataModule`: Lightning data module
  - `SimpleVideoDataset`: Video dataset
  - `FrameCache`: Frame caching system

### Utils Modules

- **config.py**: Configuration management
  - `parse_args()`: Command-line argument parsing
  - `TrainingConfig`: Configuration class
  - `get_default_config()`: Get default configuration

- **video_utils.py**: Video processing utilities
  - `hash_path()`: Path hashing function

<!-- ## 🎓 Citation

If this project helps your research, please consider citing it.

## 📄 License

[Add license information here]

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

## 📧 Contact

[Add contact information here] -->