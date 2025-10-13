#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration management for Video Action Recognition training.

This module handles command-line argument parsing and configuration.
"""

from __future__ import annotations

import argparse
from typing import Any


class TrainingConfig:
    """
    Configuration class for training parameters.
    
    Provides a structured interface to training configuration with
    validation and default values.
    """
    
    def __init__(self, **kwargs):
        """Initialize configuration from keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create configuration from parsed arguments."""
        return cls(**vars(args))
    
    def __repr__(self) -> str:
        items = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"TrainingConfig({', '.join(items)})"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for video action recognition training.
    
    Returns:
        Namespace containing all configuration parameters
        
    Example:
        >>> args = parse_args()
        >>> print(args.num_classes)
        400
    """
    p = argparse.ArgumentParser(
        description='Video Action Recognition with Graph-based Frame/Token Sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ===== Data Configuration =====
    data_group = p.add_argument_group('Data')
    data_group.add_argument(
        '--data-root',
        type=str,
        default='.',
        help='Root directory to prepend to relative video paths in CSV'
    )
    # Built-in dataset selector (hmdb51 / diving48) OR provide explicit CSVs below
    data_group.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['hmdb51', 'diving48', 'ssv2'],
        help='Name of built-in dataset to auto-generate annotation CSVs (overrides --train-anno/--val-anno/--test-anno)'
    )
    data_group.add_argument(
        '--anno-cache',
        type=str,
        default='.cache_annotations',
        help='Directory to store generated annotation CSV/mapping when using --dataset'
    )
    data_group.add_argument(
        '--train-anno',
        type=str,
        default=None,
        help='Path to training annotation CSV file (ignored if --dataset used)'
    )
    data_group.add_argument(
        '--val-anno',
        type=str,
        default=None,
        help='Path to validation annotation CSV file (ignored if --dataset used)'
    )
    data_group.add_argument(
        '--test-anno',
        type=str,
        default=None,
        help='Path to test annotation CSV file (ignored if --dataset used)'
    )
    # Diving48 specific optional overrides
    data_group.add_argument(
        '--diving48-train-json',
        type=str,
        default=None,
        help='Override path to Diving48_V2_train.json (only if --dataset diving48/div48)'
    )
    data_group.add_argument(
        '--diving48-test-json',
        type=str,
        default=None,
        help='Override path to Diving48_V2_test.json (only if --dataset diving48/div48)'
    )
    data_group.add_argument(
        '--diving48-val-ratio',
        type=float,
        default=0.1,
        help='Validation ratio split when auto-generating Diving48 annotations'
    )
    data_group.add_argument(
        '--use-test-as-val',
        action='store_true',
        help='Use test split also as validation when no explicit validation set (e.g., hmdb51 without validation)'
    )
    data_group.add_argument(
        '--frame-cache',
        type=str,
        default=None,
        help='Directory to cache decoded frames (optional, improves speed)'
    )
    
    # ===== Model Hyperparameters =====
    model_group = p.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Number of action classes (optional if --dataset used; will be inferred)'
    )
    model_group.add_argument(
        '--frames-per-clip',
        type=int,
        default=16,
        help='Number of frames to sample from each video'
    )
    model_group.add_argument(
        '--frame-topk',
        type=int,
        default=8,
        help='Number of top frames to select for temporal modeling'
    )
    model_group.add_argument(
        '--token-topk',
        type=int,
        default=32,
        help='Number of top tokens to select per frame'
    )
    model_group.add_argument(
        '--vit-name',
        type=str,
        default='vit_base_patch16_224',
        help='Name of Vision Transformer backbone model from timm'
    )
    model_group.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not use pretrained ViT weights'
    )
    model_group.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone during training'
    )
    
    # ===== Selection and Graph Parameters =====
    graph_group = p.add_argument_group('Selection (legacy)')
    graph_group.add_argument(
        '--tau-frame',
        type=float,
        default=1.0,
        help='Temperature for frame selection (higher = more uniform)'
    )
    graph_group.add_argument(
        '--tau-token',
        type=float,
        default=0.7,
        help='Temperature for token selection (higher = more uniform)'
    )
    # Keep legacy graph args for backward-compatibility (no effect now)
    graph_group.add_argument('--graph-knn', type=int, default=8, help='[legacy] graph K (unused)')
    graph_group.add_argument('--graph-tw', type=int, default=2, help='[legacy] graph temporal window (unused)')
    graph_group.add_argument('--graph-layers', type=int, default=1, help='[legacy] graph layers (unused)')
    graph_group.add_argument('--no-gat', action='store_true', help='[legacy] disable GAT (unused)')
    
    # ===== Optimization =====
    opt_group = p.add_argument_group('Optimization')
    opt_group.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size per device'
    )
    opt_group.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    opt_group.add_argument(
        '--max-epochs',
        type=int,
        default=30,
        help='Maximum number of training epochs'
    )
    opt_group.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='Learning rate (backbone uses lr * 0.5)'
    )
    opt_group.add_argument(
        '--weight-decay',
        type=float,
        default=0.05,
        help='Weight decay (L2 regularization)'
    )
    opt_group.add_argument(
        '--label-smoothing',
        type=float,
        default=0.0,
        help='Label smoothing factor for cross-entropy loss'
    )
    opt_group.add_argument(
        '--accumulate',
        type=int,
        default=1,
        help='Gradient accumulation steps (effective batch = batch_size * accumulate)'
    )
    opt_group.add_argument(
        '--precision',
        type=str,
        default='16-mixed',
        choices=['32', '16-mixed', 'bf16-mixed'],
        help='Training precision (mixed precision can speed up training)'
    )
    opt_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # ===== Logging & Checkpoint =====
    log_group = p.add_argument_group('Logging & Checkpointing')
    log_group.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for logs and checkpoints'
    )
    log_group.add_argument(
        '--project',
        type=str,
        default='graphsampler',
        help='Project name for organizing experiments'
    )
    log_group.add_argument(
        '--test-each-epoch',
        action='store_true',
        help='Evaluate test split after each training epoch'
    )
    log_group.add_argument(
        '--no-future-warning',
        action='store_true',
        help='Suppress Python FutureWarning messages'
    )
    log_group.add_argument(
        '--no-user-warning',
        action='store_true',
        help='Suppress Python UserWarning messages'
    )
    log_group.add_argument(
        '--no-tqdm',
        action='store_true',
        help='Disable tqdm progress bars'
    )
    log_group.add_argument(
        '--print-interval',
        type=int,
        default=0,
        help='When > 0 and progress bar disabled, print training status every N steps'
    )
    
    # ===== Hardware =====
    hw_group = p.add_argument_group('Hardware')
    hw_group.add_argument(
        '--devices',
        type=int,
        default=1,
        help='Number of GPUs to use (0 for CPU only)'
    )
    hw_group.add_argument(
        '--strategy',
        type=str,
        default='auto',
        help='Distributed training strategy (auto, ddp, ddp_spawn, etc.)'
    )
    
    # ===== Temporal Fusion (SQuaRe-Fuse) =====
    b_group = p.add_argument_group('Temporal Fusion (SQuaRe-Fuse)')
    b_group.add_argument('--frieren-num-dirs', type=int, default=8, help='[compat] Number of projection directions K (alias for SQuaRe-Fuse)')
    b_group.add_argument('--frieren-beta', type=float, default=0.5, help='[compat] Initial residual gate beta (alias for SQuaRe-Fuse)')
    b_group.add_argument('--frieren-ortho', action='store_true', default=True, help='[compat] Orthonormalize projection each forward')
    b_group.add_argument('--no-frieren-ortho', dest='frieren_ortho', action='store_false', help='[compat] Disable orthonormalization')

    args = p.parse_args()

    # Canonicalize dataset aliases
    # (Alias removed) Only canonical names accepted now.

    # ---- Post-parse validation ----
    if args.dataset is None:
        missing = [n for n in ['train_anno', 'val_anno', 'test_anno'] if getattr(args, n) is None]
        if missing:
            p.error(f"Missing annotation CSV(s): {missing}. Provide them or use --dataset.")
        if args.num_classes is None:
            p.error('--num-classes is required when not using --dataset')
    else:
        # If dataset chosen, CSV paths are auto-generated; ignore user-provided ones
        pass

    # Remove deprecated mappings (older flags) and map loosely to SQuaRe-Fuse when possible
    if getattr(args, 'bdrf_bound_scale', None) is not None:
        args.frieren_bound_scale = args.bdrf_bound_scale
        try:
            print('[warn] --bdrf-bound-scale is deprecated and ignored by SQuaRe-Fuse.', flush=True)
        except Exception:
            pass
        delattr(args, 'bdrf_bound_scale')
    if getattr(args, 'bdrf_poly_order', None) is not None:
        # Heuristic mapping: poly order P roughly maps to largest scale ~ 2^P
        P = int(args.bdrf_poly_order)
        mapped = [1]
        # produce up to 3 scales similar to default
        for r in [2, 4, 8, 16]:
            if r <= max(2 ** max(P, 1), 16):
                mapped.append(r)
        args.frieren_scales = mapped[:3]
        try:
            print(f"[warn] --bdrf-poly-order is deprecated and ignored by SQuaRe-Fuse.", flush=True)
        except Exception:
            pass
        delattr(args, 'bdrf_poly_order')
    # Older square_* flags
    if getattr(args, 'square_quantiles', None) is not None:
        try:
            print('[warn] --square-quantiles is deprecated and ignored.', flush=True)
        except Exception:
            pass
        delattr(args, 'square_quantiles')
    if getattr(args, 'square_poly_order', None) is not None:
        P = int(args.square_poly_order)
        mapped = [1]
        for r in [2, 4, 8, 16]:
            if r <= max(2 ** max(P, 1), 16):
                mapped.append(r)
        args.frieren_scales = mapped[:3]
        try:
            print(f"[warn] --square-poly-order is deprecated and ignored by SQuaRe-Fuse.", flush=True)
        except Exception:
            pass
        delattr(args, 'square_poly_order')

    return args


def get_default_config() -> dict:
    """
    Get default configuration as a dictionary.
    
    Useful for programmatic configuration without CLI.
    
    Returns:
        Dictionary of default configuration values
    """
    return {
        # Data
        'data_root': '.',
        'dataset': None,
        'anno_cache': '.cache_annotations',
        'frame_cache': None,
        'train_anno': None,
        'val_anno': None,
        'test_anno': None,
        'diving48_train_json': None,
        'diving48_test_json': None,
        'diving48_val_ratio': 0.1,
        
        # Model
        'num_classes': None,
        'frames_per_clip': 16,
        'frame_topk': 8,
        'token_topk': 32,
        'vit_name': 'vit_base_patch16_224',
        'vit_pretrained': True,
        'freeze_backbone': False,
        
        # Selection & Graph
        'tau_frame': 1.0,
        'tau_token': 0.7,
        'graph_knn': 8,
        'graph_tw': 2,
        'graph_layers': 1,
        'use_gat': True,
        
        # Optimization
        'batch_size': 2,
        'num_workers': 4,
        'max_epochs': 30,
        'lr': 5e-4,
        'weight_decay': 0.05,
        'label_smoothing': 0.0,
        'accumulate': 1,
        'precision': '16-mixed',
        'seed': 42,
        
        # Logging
        'output': 'outputs',
        'project': 'graphsampler',
        'test_each_epoch': False,
        'print_interval': 0,
        
        # Hardware
        'devices': 1,
        'strategy': 'auto',
    # SQuaRe-Fuse defaults (compat keys)
    'frieren_num_dirs': 8,
    'frieren_beta': 0.5,
    'frieren_ortho': True,
    }
