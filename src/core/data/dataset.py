#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch dataset for video action recognition.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.io import read_video

from .annotation import VideoCSVAnnotation
from .cache import FrameCache


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
        # Always return exactly self.num_frames indices
        if total <= 0:
            return []
        if total < self.num_frames:
            # Repeat indices to reach desired length (stable and simple)
            idxs: List[int] = list(range(total))
            # Repeat whole cycle until sufficient length
            while len(idxs) < self.num_frames:
                idxs.extend(idxs)
            return idxs[:self.num_frames]

        # Uniform sampling for longer videos
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
        
        # Safety: if decoder returned fewer than requested frames (shouldn't happen),
        # pad by repeating the last frame.
        if len(frames) < self.num_frames and len(frames) > 0:
            last = frames[-1]
            frames.extend([last] * (self.num_frames - len(frames)))

        clip = torch.stack(frames, dim=0).contiguous()  # [T, C, H, W]
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
        """Load frames directly from video file with multi-backend fallback.

        Order tried:
          1. torchvision.read_video (pts_unit='sec')
          2. torchvision.read_video (pts_unit='pts') if 0 frames
          3. PyAV (if installed) - slower but robust
          4. OpenCV (if installed)

        Raises:
            RuntimeError with detailed diagnostics if all backends fail / return 0 frames.
        """

        def _torchvision_decode(path: str):
            try:
                v, _, info = read_video(path, pts_unit='sec')
                if v.shape[0] == 0:  # fallback to pts
                    v2, _, info2 = read_video(path, pts_unit='pts')
                    if v2.shape[0] > 0:
                        return v2, info2, 'torchvision(read_video,pts)'
                else:
                    return v, info, 'torchvision(read_video,sec)'
                return v, info, 'torchvision(read_video,sec-empty)'
            except Exception as e:
                return None, {'error': str(e)}, 'torchvision-exception'

        def _pyav_decode(path: str):
            try:
                import av  # type: ignore
            except ImportError:
                return None, {'error': 'pyav-not-installed'}, 'pyav-missing'
            try:
                container = av.open(path)
                stream = container.streams.video[0]
                frames = []
                for frame in container.decode(stream):
                    arr = frame.to_rgb().to_ndarray()
                    frames.append(torch.from_numpy(arr))  # [H,W,C]
                if frames:
                    video = torch.stack(frames, 0)
                else:
                    video = torch.empty((0,))
                return video, {'pyav_frames': len(frames)}, 'pyav'
            except Exception as e:
                return None, {'error': str(e)}, 'pyav-exception'

        def _opencv_decode(path: str):
            try:
                import cv2  # type: ignore
            except ImportError:
                return None, {'error': 'opencv-not-installed'}, 'opencv-missing'
            try:
                cap = cv2.VideoCapture(path)
                frames = []
                ok = cap.isOpened()
                while ok:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(torch.from_numpy(frame))
                cap.release()
                if frames:
                    video = torch.stack(frames, 0)
                else:
                    video = torch.empty((0,))
                return video, {'opencv_frames': len(frames)}, 'opencv'
            except Exception as e:
                return None, {'error': str(e)}, 'opencv-exception'

        attempts = []
        video, info, backend = _torchvision_decode(video_path)
        attempts.append((backend, (None if isinstance(info, dict) and 'error' in info else info), info))
        if video is None or (hasattr(video, 'shape') and video.shape[0] == 0):
            pv_video, pv_info, pv_backend = _pyav_decode(video_path)
            attempts.append((pv_backend, pv_info, pv_info))
            if pv_video is not None and getattr(pv_video, 'shape', [0])[0] > 0:
                video, info, backend = pv_video, pv_info, pv_backend
        if video is None or (hasattr(video, 'shape') and video.shape[0] == 0):
            ocv_video, ocv_info, ocv_backend = _opencv_decode(video_path)
            attempts.append((ocv_backend, ocv_info, ocv_info))
            if ocv_video is not None and getattr(ocv_video, 'shape', [0])[0] > 0:
                video, info, backend = ocv_video, ocv_info, ocv_backend

        if video is None or (hasattr(video, 'shape') and video.shape[0] == 0):
            # Collect diagnostics
            try:
                file_size = os.path.getsize(video_path)
            except OSError:
                file_size = -1
            try:
                import torchvision.io.video as _tvid
                has_video_opt = getattr(_tvid, '_has_video_opt', lambda: 'N/A')()
            except Exception:
                has_video_opt = 'error'
            diag = {
                'attempts': attempts,
                'file_size': file_size,
                'torchvision_has_video_opt': has_video_opt,
                'path': video_path,
            }
            raise RuntimeError(
                "Failed to decode video (no frames) after fallbacks. "
                f"Diagnostics: {diag}. Consider installing system ffmpeg or rebuilding torchvision with ffmpeg."
            )

        # video is a tensor [T,H,W,C] uint8 or int
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)
        total = video.shape[0]
        idxs = self._sample_indices(total)
        frames: List[torch.Tensor] = []
        for i in idxs:
            frame = video[i]
            # --- Minimal & robust channel ordering fix (ensure HWC with C in {1,3,4}) ---
            if not isinstance(frame, torch.Tensor):
                raise TypeError(f"Expected frame tensor, got {type(frame)}")
            if frame.ndim != 3:
                raise ValueError(f"Decoded frame must be 3D, got shape {tuple(frame.shape)}")

            # Fast path: already HWC
            if frame.shape[-1] in (1, 3, 4):
                pass
            # CHW -> HWC
            elif frame.shape[0] in (1, 3, 4) and frame.shape[1] > 4 and frame.shape[2] > 4:
                frame = frame.permute(1, 2, 0).contiguous()
            else:
                # Locate any dim that looks like channel count
                chan_dim = None
                for d in range(3):
                    if frame.shape[d] in (1, 3, 4):
                        chan_dim = d
                        break
                if chan_dim is None:
                    # Fall back: treat smallest dim as channel if <=4
                    smallest_dim = int(torch.tensor(frame.shape).argmin().item())
                    if frame.shape[smallest_dim] <= 4:
                        chan_dim = smallest_dim
                if chan_dim is not None and chan_dim != 2:
                    order = [d for d in range(3) if d != chan_dim] + [chan_dim]
                    frame = frame.permute(order).contiguous()

            if frame.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"Unable to coerce frame to HWC with valid channel dim. Final shape={tuple(frame.shape)}"
                )

            # Convert to uint8 if necessary for PIL
            if frame.dtype != torch.uint8:
                # Assume frame is in 0..255 range already (read_video returns uint8 normally)
                frame_to_pil = frame.to(torch.uint8)
            else:
                frame_to_pil = frame

            try:
                # Ensure frame is in CHW format for to_pil_image
                frame_for_pil = frame_to_pil.permute(2, 0, 1).contiguous()
                pil = transforms.functional.to_pil_image(frame_for_pil)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert frame to PIL. Original shape before tx: {tuple(frame.shape)}; error: {e}"
                ) from e
            frames.append(self.tx(pil))
        if backend != 'torchvision(read_video,sec)' and backend != 'torchvision(read_video,pts)':
            warnings.warn(f"Video {video_path} decoded using fallback backend {backend}")
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


__all__ = ["SimpleVideoDataset"]
