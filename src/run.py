#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for Video Action Recognition training pipeline.

This script provides a command-line interface for training video action recognition
models using PyTorch Lightning with efficient frame/token selection and graph-based
temporal modeling.

Features:
  * CLI control (see --help for all options)
  * On-the-fly video decoding with optional frame caching
  * Uniform temporal sampling from video clips
  * Frame+token co-selection before temporal modeling
  * Graph-based memory for efficient temporal relationships
  * Automatic test evaluation after each epoch (optional)
  * Mixed precision training, gradient accumulation, checkpointing

Expected annotation CSV format (train / val / test):
    video_path,label
    /path/to/video1.mp4,0
    relative/path/video2.avi,1
    ...

Where video_path can be absolute or relative to --data-root.
Labels should be integer class ids in range [0, num_classes-1].

Example usage:
  python -m src.run \
      --data-root data/videos \
      --train-anno train.csv --val-anno val.csv --test-anno test.csv \
      --num-classes 400 --frames-per-clip 16 --frame-topk 8 --token-topk 32 \
      --batch-size 2 --max-epochs 50 --lr 5e-4 --accumulate 2

For more options, run:
  python -m src.run --help

Notes:
  * For large-scale training, consider implementing batched video decoders
    (PyAV, decord) and multi-epoch frame caching
  * Data augmentations here are minimal; extend in core.data as needed
  * GraphBasedMemBank can be further optimized with vectorization
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

# Import modularized components
from src.core import GraphSamplerActionModel, VideoDataModule
from src.core.data import create_datamodule_for
from src.utils import parse_args


def setup_callbacks(output_dir: str, project_name: str, monitor_metric: str = 'val/acc'):
    """
    Setup PyTorch Lightning callbacks.
    
    Args:
        output_dir: Root output directory
        project_name: Project name for organizing experiments
        
    Returns:
        List of callback instances
    """
    filename_tmpl = 'epoch{epoch:02d}'
    if monitor_metric == 'val/acc':
        filename_tmpl += '-val_acc{val/acc:.3f}'
    elif monitor_metric == 'test/acc':
        filename_tmpl += '-test_acc{test/acc:.3f}'
    elif monitor_metric == 'train/acc':
        filename_tmpl += '-train_acc{train/acc:.3f}'

    ckpt_kwargs = dict(
        dirpath=os.path.join(output_dir, project_name, 'checkpoints'),
        filename=filename_tmpl,
        monitor=monitor_metric,
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    ckpt_cb = ModelCheckpoint(**ckpt_kwargs)
    
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    
    return [ckpt_cb, lr_cb]


def setup_trainer(args, monitor_metric: str) -> L.Trainer:
    """
    Setup PyTorch Lightning Trainer.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configured Trainer instance
    """
    os.makedirs(args.output, exist_ok=True)
    logger = CSVLogger(save_dir=args.output, name=args.project)
    callbacks = setup_callbacks(args.output, args.project, monitor_metric=monitor_metric)
    
    # Determine accelerator
    if args.devices > 0 and torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'
    
    # Control progress bar display
    enable_progress_bar = not getattr(args, 'no_tqdm', False)
    
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices if args.devices > 0 else None,
        accelerator=accelerator,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate,
        strategy=args.strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=False,
        gradient_clip_val=10.0,
        enable_progress_bar=enable_progress_bar,
    )
    
    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Suppress FutureWarnings if requested
    if getattr(args, 'no_future_warning', False):
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed, workers=True)
    
    # Setup data module (support built-in datasets)
    if args.dataset is not None:
        dataset_name = args.dataset
        dm = create_datamodule_for(
            dataset=dataset_name,
            root_dir=args.data_root,
            frames_per_clip=args.frames_per_clip,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            frame_cache_dir=args.frame_cache,
            resize=224,
            cache_dir=args.anno_cache,
            diving48_train_json=args.diving48_train_json,
            diving48_test_json=args.diving48_test_json,
            diving48_val_ratio=getattr(args, 'diving48_val_ratio', 0.1),
            use_test_as_val=getattr(args, 'use_test_as_val', False),
        )
        # Infer num_classes if not provided
        if args.num_classes is None:
            # Try to read mapping json in cache dir
            mapping_file = None
            if dataset_name.lower() == 'hmdb51':
                mapping_file = Path(args.anno_cache) / 'hmdb51_label_mapping.json'
            elif dataset_name.lower() == 'diving48':
                mapping_file = Path(args.anno_cache) / 'diving48_label_mapping.json'
            if mapping_file and mapping_file.is_file():
                import json
                with mapping_file.open('r', encoding='utf-8') as f:
                    mapping = json.load(f)
                if 'label_to_id' in mapping:
                    args.num_classes = len(mapping['label_to_id'])
    else:
        dm = VideoDataModule(
            data_root=args.data_root,
            train_csv=args.train_anno,
            val_csv=args.val_anno,
            test_csv=args.test_anno,
            frames_per_clip=args.frames_per_clip,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            frame_cache_dir=args.frame_cache,
            resize=224,
        )

    if args.num_classes is None:
        raise ValueError('Could not determine num_classes. Specify --num-classes or rely on built-in dataset mapping.')
    
    # Decide which metric to monitor
    monitor_metric = 'val/acc'
    if getattr(args, 'use_test_as_val', False) and args.dataset is not None:
        # Validation shares test -> trainer will log test metrics only during explicit test stage.
        # We alias validation to test inside datamodule, so metrics emitted should still be 'val/acc'.
        # If no separate val_set was created, we fallback to 'train/acc' during training for checkpointing.
        # Check if datamodule actually has val_dataloader
        has_val = dm.val_dataloader() is not None
        if not has_val:
            monitor_metric = 'train/acc'

    # Setup model
    model = GraphSamplerActionModel(
        num_classes=args.num_classes,
        frames_per_clip=args.frames_per_clip,
        frame_topk=args.frame_topk,
        token_topk=args.token_topk,
        vit_name=args.vit_name,
        vit_pretrained=not args.no_pretrained,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        tau_frame=args.tau_frame,
        tau_token=args.tau_token,
        graph_knn=args.graph_knn,
        graph_tw=args.graph_tw,
        graph_layers=args.graph_layers,
        use_gat=not args.no_gat,
        label_smoothing=args.label_smoothing,
        test_each_epoch=args.test_each_epoch,
        selector_type='fps',       # 或 'learnable'
		membank_type='latent',     # 'graph' | 'latent' | 'tcn'
		latent_slots=64,
		latent_heads=8
    )
    
    # Setup trainer
    trainer = setup_trainer(args, monitor_metric=monitor_metric)
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Final test evaluation
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    main()

