#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Lightning data module for video action recognition.
"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader
import lightning as L

from .annotation import VideoCSVAnnotation
from .cache import FrameCache
from .dataset import SimpleVideoDataset


class VideoDataModule(L.LightningDataModule):
    """
    PyTorch Lightning data module for video action recognition.
    
    Manages train/val/test datasets and dataloaders with consistent configuration.
    
    Args:
        data_root: Root directory for video files
        train_csv: Path to training annotation CSV
        val_csv: Path to validation annotation CSV
        test_csv: Path to test annotation CSV
        frames_per_clip: Number of frames to sample per video
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader worker processes
        frame_cache_dir: Optional directory for frame caching
        resize: Size to resize frames to
    """
    
    def __init__(
        self,
        data_root: str,
        train_csv: str,
        val_csv: Optional[str],
        test_csv: str,
        frames_per_clip: int,
        batch_size: int,
        num_workers: int = 4,
        frame_cache_dir: Optional[str] = None,
        resize: int = 224,
        use_test_as_val: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.frame_cache = FrameCache(frame_cache_dir)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for the specified stage."""
        h = self.hparams
        
        if stage in (None, 'fit'):
            self.train_set = SimpleVideoDataset(
                VideoCSVAnnotation(h.train_csv, h.data_root),
                h.frames_per_clip,
                self.frame_cache,
                is_train=True,
                resize=h.resize,
            )
            if h.val_csv is not None:
                self.val_set = SimpleVideoDataset(
                    VideoCSVAnnotation(h.val_csv, h.data_root),
                    h.frames_per_clip,
                    self.frame_cache,
                    is_train=False,
                    resize=h.resize,
                )
            elif h.use_test_as_val:
                # Build test set early and alias as validation
                self.test_set = SimpleVideoDataset(
                    VideoCSVAnnotation(h.test_csv, h.data_root),
                    h.frames_per_clip,
                    self.frame_cache,
                    is_train=False,
                    resize=h.resize,
                )
                self.val_set = self.test_set
        
        if stage in (None, 'test', 'predict') and not hasattr(self, 'test_set'):
            self.test_set = SimpleVideoDataset(
                VideoCSVAnnotation(h.test_csv, h.data_root),
                h.frames_per_clip,
                self.frame_cache,
                is_train=False,
                resize=h.resize,
            )
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """Create validation dataloader (optional)."""
        if not hasattr(self, 'val_set'):
            return None
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


__all__ = ["VideoDataModule"]
