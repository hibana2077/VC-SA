#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data handling modules for Video Action Recognition.

This module contains:
  - VideoRecord: Simple container for video path and label
  - VideoCSVAnnotation: Parser for CSV annotation files
  - FrameCache: Disk-based frame caching system
  - SimpleVideoDataset: PyTorch dataset for video clips
  - VideoDataModule: PyTorch Lightning data module
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.io import read_video

import lightning as L

from ..utils.video_utils import hash_path


class VideoRecord:
    """
    Simple container for a video record.
    
    Attributes:
        path: Path to the video file
        label: Integer class label
    """
    
    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label
    
    def __repr__(self) -> str:
        return f"VideoRecord(path='{self.path}', label={self.label})"


class VideoCSVAnnotation:
    """
    Parser for video annotation CSV files.
    
    Expected CSV format:
        video_path,label
        /path/to/video1.mp4,0
        /path/to/video2.mp4,1
        ...
    
    Args:
        csv_path: Path to the annotation CSV file
        data_root: Optional root directory to prepend to relative video paths
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid or no records found
    """
    
    def __init__(self, csv_path: str, data_root: Optional[str] = None):
        self.csv_path = csv_path
        self.data_root = data_root
        self.records: List[VideoRecord] = []
        
        self._load_annotations()
    
    def _load_annotations(self):
        """Load annotations from CSV file."""
        csv_path = Path(self.csv_path)
        
        if not csv_path.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
        
        with csv_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip empty lines and comments
                if not row or row[0].startswith('#'):
                    continue
                
                if len(row) < 2:
                    raise ValueError(
                        f"CSV row must have at least 2 columns (path,label): {row}"
                    )
                
                vp = row[0]
                # Resolve relative paths
                if self.data_root and not os.path.isabs(vp):
                    vp = os.path.join(self.data_root, vp)
                
                label = int(row[1])
                self.records.append(VideoRecord(vp, label))
        
        if len(self.records) == 0:
            raise ValueError(f"No valid records in {csv_path}")
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> VideoRecord:
        return self.records[idx]
    
    def __repr__(self) -> str:
        return f"VideoCSVAnnotation(csv_path='{self.csv_path}', num_records={len(self)})"


class FrameCache:
    """
    Disk-based frame caching system.
    
    Caches decoded video frames as JPEG files to speed up repeated access.
    Each video is cached in a directory named by its path hash, containing
    frame_00000.jpg, frame_00001.jpg, etc.
    
    Args:
        cache_root: Root directory for frame cache. If None, caching is disabled.
    """
    
    def __init__(self, cache_root: Optional[str]):
        self.cache_root = Path(cache_root) if cache_root else None
        
        if self.cache_root:
            self.cache_root.mkdir(parents=True, exist_ok=True)
    
    def get_or_extract(self, video_path: str) -> List[Path]:
        """
        Get cached frames or extract and cache them.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of paths to cached frame files, or empty list if caching disabled
            
        Raises:
            RuntimeError: If video reading fails
        """
        if self.cache_root is None:
            return []  # Signal to decode on-the-fly
        
        vid_hash = hash_path(video_path)
        out_dir = self.cache_root / vid_hash
        
        # Check if already cached
        if out_dir.is_dir():
            frame_files = sorted(out_dir.glob('frame_*.jpg'))
            if frame_files:
                return frame_files
        
        # Extract and cache frames
        return self._extract_and_cache(video_path, out_dir)
    
    def _extract_and_cache(self, video_path: str, out_dir: Path) -> List[Path]:
        """Extract frames from video and cache them."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            video, _, _ = read_video(video_path, pts_unit='sec')  # [T, H, W, C] uint8
        except Exception as e:
            raise RuntimeError(f"Failed to read video {video_path}: {e}")
        
        # Save each frame as JPEG
        for i in range(video.shape[0]):
            frame = video[i].numpy()
            img = torchvision.transforms.functional.to_pil_image(frame)
            img.save(out_dir / f"frame_{i:05d}.jpg", quality=90)
        
        frame_files = sorted(out_dir.glob('frame_*.jpg'))
        return frame_files
    
    def clear_cache(self):
        """Clear all cached frames."""
        if self.cache_root and self.cache_root.exists():
            import shutil
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)


class SimpleVideoDataset(Dataset):
    """
    PyTorch dataset for video action recognition.
    
    Loads videos, samples frames uniformly, and applies transformations.
    Supports optional frame caching for efficiency.
    
    Args:
        anno: VideoCSVAnnotation instance containing video records
        num_frames: Number of frames to sample from each video
        frame_cache: FrameCache instance for caching decoded frames
        is_train: Whether this is training set (affects augmentation)
        resize: Size to resize frames to
    """
    
    def __init__(
        self,
        anno: VideoCSVAnnotation,
        num_frames: int,
        frame_cache: FrameCache,
        is_train: bool = True,
        resize: int = 224,
    ):
        self.anno = anno
        self.num_frames = num_frames
        self.frame_cache = frame_cache
        self.is_train = is_train
        self.resize = resize
        self.tx = self._build_transform()
    
    def _build_transform(self):
        """Build image transformation pipeline."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        aug: List[transforms.Module] = [
            transforms.Resize((self.resize, self.resize))
        ]
        
        if self.is_train:
            # Placeholder for more advanced augmentation
            # Can add RandomHorizontalFlip, ColorJitter, etc.
            pass
        
        aug.extend([
            transforms.ToTensor(),
            normalize,
        ])
        
        return transforms.Compose(aug)
    
    def _sample_indices(self, total: int) -> List[int]:
        """
        Sample frame indices uniformly from the video.
        
        Args:
            total: Total number of frames in the video
            
        Returns:
            List of frame indices to sample
        """
        if total <= self.num_frames:
            return list(range(total))
        
        # Uniform sampling
        stride = total / self.num_frames
        return [int(stride * i + stride * 0.5) for i in range(self.num_frames)]
    
    def _load_frames(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess frames from video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tensor of shape [T, C, H, W] containing preprocessed frames
            
        Raises:
            RuntimeError: If no frames could be extracted
        """
        frame_files = self.frame_cache.get_or_extract(video_path)
        frames: List[torch.Tensor] = []
        
        if frame_files:
            # Use cached frames
            frames = self._load_from_cache(frame_files)
        else:
            # Decode directly
            frames = self._load_from_video(video_path)
        
        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted for {video_path}")
        
        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        return clip
    
    def _load_from_cache(self, frame_files: List[Path]) -> List[torch.Tensor]:
        """Load frames from cached files."""
        total = len(frame_files)
        idxs = self._sample_indices(total)
        frames = []
        
        for i in idxs:
            img = torchvision.io.read_image(str(frame_files[i]))  # [C,H,W] uint8
            img = img.float() / 255.0
            pil = transforms.functional.to_pil_image(img)
            frames.append(self.tx(pil))
        
        return frames
    
    def _load_from_video(self, video_path: str) -> List[torch.Tensor]:
        """Load frames directly from video file."""
        video, _, _ = read_video(video_path, pts_unit='sec')  # [T,H,W,C]
        total = video.shape[0]
        idxs = self._sample_indices(total)
        frames = []
        
        for i in idxs:
            frame = video[i]  # [H,W,C]
            pil = transforms.functional.to_pil_image(frame)
            frames.append(self.tx(pil))
        
        return frames
    
    def __len__(self) -> int:
        return len(self.anno)
    
    def __getitem__(self, idx: int):
        """
        Get a video clip and its label.
        
        Args:
            idx: Index of the video
            
        Returns:
            Tuple of (clip, label) where clip has shape [T, C, H, W]
        """
        record = self.anno[idx]
        clip = self._load_frames(record.path)  # [T,C,H,W]
        label = record.label
        return clip, label


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
        val_csv: str,
        test_csv: str,
        frames_per_clip: int,
        batch_size: int,
        num_workers: int = 4,
        frame_cache_dir: Optional[str] = None,
        resize: int = 224,
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
            self.val_set = SimpleVideoDataset(
                VideoCSVAnnotation(h.val_csv, h.data_root),
                h.frames_per_clip,
                self.frame_cache,
                is_train=False,
                resize=h.resize,
            )
        
        if stage in (None, 'test', 'predict'):
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
        """Create validation dataloader."""
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
