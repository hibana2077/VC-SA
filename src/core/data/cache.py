#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Disk-based frame caching system for video data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torchvision
from torchvision.io import read_video

from ...utils.video_utils import hash_path


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


__all__ = ["FrameCache"]
