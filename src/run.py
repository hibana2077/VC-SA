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


class PeriodicPrinterCallback(L.Callback):
    """Lightweight callback to print training status every N steps when tqdm is disabled.

    Prints: epoch, global_step, loss (if logged), lr (if logged), and any accuracy metric
    present in the logger's latest logged metrics dictionary.
    """
    def __init__(self, interval: int):
        super().__init__()
        self.interval = max(1, interval)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx: int):
        if self.interval <= 0:
            return
        global_step = trainer.global_step
        if global_step == 0 or global_step % self.interval != 0:
            return
        metrics = trainer.callback_metrics  # includes latest logged metrics
        epoch = trainer.current_epoch
        pieces = [f"[Periodic] epoch={epoch}", f"step={global_step}"]
        for key in [
            'train/loss', 'loss', 'train_loss',
            'train/acc', 'val/acc', 'test/acc',
            'train/grad_norm_pre_clip', 'train/grad_norm_post_clip'
        ]:
            if key in metrics:
                val = metrics[key]
                # Convert tensors to python scalars
                try:
                    val = float(val)
                except Exception:
                    pass
                pieces.append(f"{key}={val:.4f}" if isinstance(val, (float, int)) else f"{key}={val}")
        # Try to get LR from optimizer param groups (first group)
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get('lr')
            if lr is not None:
                pieces.append(f"lr={lr:.2e}")
        print(' | '.join(pieces), flush=True)


class EpochSummaryPrinter(L.Callback):
    """Callback to print a concise metric summary once每個 epoch 結束.

    行為:
      * 在 validation epoch 結束後 (on_validation_epoch_end) 輸出彙總 metrics
      * 若沒有 validation (沒有 val dataloader) 則在 on_train_epoch_end 輸出 (避免重複)
    列印內容包含: epoch, 監控的常見指標 (train/acc, train/loss, val/acc, val/loss,
    epoch_test/acc, epoch_test/loss, test/acc, test/loss) 以及第一個 optimizer 的 lr。
    """
    def __init__(self, extra_keys: list[str] | None = None):
        super().__init__()
        self.default_keys = [
            'train/acc', 'train/loss',
            'val/acc', 'val/loss',
            'epoch_test/acc', 'epoch_test/loss',
            'test/acc', 'test/loss'
        ]
        self.extra_keys = extra_keys or []

    def _print_summary(self, trainer: L.Trainer):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        pieces = [f"[EpochSummary] epoch={epoch}"]
        for key in self.default_keys + self.extra_keys:
            if key in metrics:
                val = metrics[key]
                try:
                    val = float(val)
                    pieces.append(f"{key}={val:.4f}")
                except Exception:
                    pieces.append(f"{key}={val}")
        # learning rate
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0].get('lr')
                if lr is not None:
                    pieces.append(f"lr={lr:.2e}")
            except Exception:
                pass
        print(' | '.join(pieces), flush=True)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module):  # type: ignore
        # 有 validation -> 在此列印
        if sum(trainer.num_val_batches) > 0:  # type: ignore[attr-defined]
            self._print_summary(trainer)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module):  # type: ignore
        # 沒有 validation -> 在 train epoch end 列印 (避免重複)
        if sum(trainer.num_val_batches) == 0:  # type: ignore[attr-defined]
            self._print_summary(trainer)


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
    # Inject periodic printer when tqdm disabled and interval specified
    if getattr(args, 'no_tqdm', False) and getattr(args, 'print_interval', 0) > 0:
        callbacks.append(PeriodicPrinterCallback(args.print_interval))
    # Always add epoch summary printer for clear per-epoch aggregated metrics
    callbacks.append(EpochSummaryPrinter())
    
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
    
    # Suppress UserWarnings if requested
    if getattr(args, 'no_user_warning', False):
        warnings.filterwarnings('ignore', category=UserWarning)
    
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
        graph_knn=args.graph_knn,  # (legacy args kept for backward CLI compatibility, unused)
        graph_tw=args.graph_tw,
        graph_layers=args.graph_layers,
        use_gat=not args.no_gat,
        label_smoothing=args.label_smoothing,
        test_each_epoch=args.test_each_epoch,
        square_num_dirs=args.square_num_dirs,
        square_quantiles=tuple(args.square_quantiles),
        square_poly_order=args.square_poly_order,
        square_beta=args.square_beta,
        square_ortho=args.square_ortho,
    )
    
    # Setup trainer
    trainer = setup_trainer(args, monitor_metric=monitor_metric)
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Final test evaluation
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    main()

