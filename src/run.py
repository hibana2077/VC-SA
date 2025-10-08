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

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

# Import modularized components
from src.core import GraphSamplerActionModel, VideoDataModule
from src.utils import parse_args


def setup_callbacks(output_dir: str, project_name: str):
    """
    Setup PyTorch Lightning callbacks.
    
    Args:
        output_dir: Root output directory
        project_name: Project name for organizing experiments
        
    Returns:
        List of callback instances
    """
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, project_name, 'checkpoints'),
        filename='epoch{epoch:02d}-val_acc{val/acc:.3f}',
        monitor='val/acc',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    
    return [ckpt_cb, lr_cb]


def setup_trainer(args) -> L.Trainer:
    """
    Setup PyTorch Lightning Trainer.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configured Trainer instance
    """
    os.makedirs(args.output, exist_ok=True)
    logger = CSVLogger(save_dir=args.output, name=args.project)
    callbacks = setup_callbacks(args.output, args.project)
    
    # Determine accelerator
    if args.devices > 0 and torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'
    
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
    )
    
    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed, workers=True)
    
    # Setup data module
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
    )
    
    # Setup trainer
    trainer = setup_trainer(args)
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Final test evaluation
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    main()

