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
import json
import random
from pathlib import Path
from typing import List, Optional
import warnings

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
            # --- Robust shape/channel sanitation ---------------------------------
            # Expected acceptable shapes for a single frame:
            #   [H, W, C]  (C in {1,3,4})
            #   [C, H, W]  (C in {1,3,4})
            # Some backends / edge cases (or inadvertent dim ordering) may yield
            # permutations like [W, C, H] or other orders where neither the first
            # nor last dim is in {1,3,4}. This caused the observed error where
            # torchvision's to_pil_image() interpreted the last dim (e.g. 240 = H)
            # as channels: "pic should not have > 4 channels".
            # We attempt to automatically recover by detecting a single dim in
            # {1,3,4} and moving it to the end (HWC) before converting.
            if isinstance(frame, torch.Tensor):
                if frame.ndim != 3:
                    raise ValueError(f"Decoded frame must be 3D, got shape {tuple(frame.shape)}")
                h, w, c = None, None, None
                shape = list(frame.shape)
                # Case 1: already HWC
                if shape[-1] in (1, 3, 4):
                    pass  # OK
                # Case 2: CHW -> permute to HWC
                elif shape[0] in (1, 3, 4):
                    frame = frame.permute(1, 2, 0).contiguous()
                else:
                    # Try to locate a channel-like dim somewhere else
                    channel_candidates = [d for d, s in enumerate(shape) if s in (1, 3, 4)]
                    if channel_candidates:
                        cdim = channel_candidates[0]
                        # Move channel dim to end preserving order of others
                        order = [d for d in range(3) if d != cdim] + [cdim]
                        frame = frame.permute(order).contiguous()
                    else:
                        # Fallback heuristic: assume smallest dim is channel
                        cdim = int(torch.tensor(shape).argmin().item())
                        if shape[cdim] > 4:
                            raise ValueError(
                                "Unable to infer channel dimension for frame shape {} (no dim in {1,3,4})".format(
                                    shape
                                )
                            )
                        order = [d for d in range(3) if d != cdim] + [cdim]
                        frame = frame.permute(order).contiguous()

                # At this point last dim should be channel
                if frame.shape[-1] not in (1, 3, 4):
                    raise ValueError(
                        f"Sanitization failed; frame shape after permute {tuple(frame.shape)} does not end with channels."
                    )
            else:
                raise TypeError(f"Expected frame tensor, got {type(frame)}")

            # Convert to uint8 if necessary for PIL
            if frame.dtype != torch.uint8:
                # Assume frame is in 0..255 range already (read_video returns uint8 normally)
                frame_to_pil = frame.to(torch.uint8)
            else:
                frame_to_pil = frame

            try:
                pil = transforms.functional.to_pil_image(frame_to_pil)
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


# ---------------------------------------------------------------------------
# Dataset preparation utilities for fixed (built-in) datasets
# ---------------------------------------------------------------------------

HMDB51_SPLITS = ["train", "test"]  # validation split removed per user request


def _ensure_cache_dir(cache_dir: Optional[str]) -> Path:
    if cache_dir is None:
        cache_dir = ".cache_annotations"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def prepare_hmdb51_annotations(root_dir: str, cache_dir: Optional[str] = None) -> dict:
    """Prepare CSV annotation files for HMDB51 dataset.

    Expected directory layout (as documented):
        root_dir/
            train/metadata.csv
            validation/metadata.csv
            test/metadata.csv
            each directory contains the video *.mp4 files.

    metadata.csv columns (at least): video_id,file_name,label,...

    This function consolidates all unique class labels across splits -> int ids,
    writes per-split CSVs in the standard (video_path,label_id) format consumed
    by ``VideoCSVAnnotation`` and returns their paths.

    Args:
        root_dir: HMDB51 dataset root path.
        cache_dir: Where to write generated CSVs & label mapping JSON.

    Returns:
        dict with keys 'train','val','test' mapping to CSV file paths.
    """
    root = Path(root_dir)
    cache_path = _ensure_cache_dir(cache_dir)

    # 1. Collect labels
    label_set = []  # preserve insertion order
    split_rows: dict[str, list] = {}
    for split in HMDB51_SPLITS:
        meta_path = root / split / "metadata.csv"
        if not meta_path.is_file():
            raise FileNotFoundError(f"HMDB51 metadata not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                lbl = r.get("label")
                fname = r.get("file_name")
                if lbl is None or fname is None:
                    continue
                if lbl not in label_set:
                    label_set.append(lbl)
                rows.append(r)
            split_rows[split] = rows

    label_to_id = {lbl: i for i, lbl in enumerate(sorted(label_set))}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    # Save mapping for reference
    mapping_file = cache_path / "hmdb51_label_mapping.json"
    with mapping_file.open("w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, ensure_ascii=False, indent=2)

    # 2. Write CSVs (only train & test now)
    out_paths = {}
    for split, rows in split_rows.items():
        out_csv = cache_path / f"hmdb51_{split}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                video_path = (root / split / r["file_name"]).resolve()
                writer.writerow([str(video_path), label_to_id[r["label"]]])
        out_paths[split] = str(out_csv)

    return {"train": out_paths["train"], "test": out_paths["test"], "label_mapping": str(mapping_file)}


def prepare_diving48_annotations(
    rgb_root: str,
    train_json: str,
    test_json: str,
    cache_dir: Optional[str] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Prepare CSV annotation files for Diving48 dataset.

    Args:
        rgb_root: Path to the folder containing Diving48 RGB videos (mp4 files).
        train_json: Path to Diving48_V2_train.json.
        test_json: Path to Diving48_V2_test.json.
        cache_dir: Where to write generated CSVs & label vocab mapping.
        val_ratio: Fraction of training samples to reserve for validation.
        seed: RNG seed for deterministic split.

    Returns:
        dict with keys 'train','val','test' (and 'label_mapping').
    """
    cache_path = _ensure_cache_dir(cache_dir)
    rng = random.Random(seed)

    def _read_json(p: str):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    train_items = _read_json(train_json)
    test_items = _read_json(test_json)

    # Collect labels (already numeric) -> build contiguous mapping if needed
    labels = sorted({int(item["label"]) for item in (train_items + test_items)})
    # If labels already 0..N-1 contiguous we keep them; else re-map
    contiguous = labels == list(range(len(labels)))
    if contiguous:
        label_to_id = {lbl: lbl for lbl in labels}
    else:
        label_to_id = {lbl: i for i, lbl in enumerate(labels)}

    mapping_file = cache_path / "diving48_label_mapping.json"
    with mapping_file.open("w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id}, f, ensure_ascii=False, indent=2)

    # Split train into train/val
    indices = list(range(len(train_items)))
    rng.shuffle(indices)
    val_count = int(len(indices) * val_ratio)
    val_idx_set = set(indices[:val_count])

    split_data = {"train": [], "val": []}
    for i, item in enumerate(train_items):
        split_name = "val" if i in val_idx_set else "train"
        split_data[split_name].append(item)

    def _write_csv(items: list, out_path: Path):
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for it in items:
                vid = it["vid_name"]
                label_raw = int(it["label"])
                video_path = Path(rgb_root) / f"{vid}.mp4"
                writer.writerow([str(video_path), label_to_id[label_raw]])

    out_paths = {}
    for split_name in ["train", "val"]:
        out_csv = cache_path / f"diving48_{split_name}.csv"
        _write_csv(split_data[split_name], out_csv)
        out_paths[split_name] = str(out_csv)
    # test split
    test_csv = cache_path / "diving48_test.csv"
    _write_csv(test_items, test_csv)
    out_paths["test"] = str(test_csv)

    out_paths["label_mapping"] = str(mapping_file)
    return out_paths


def create_datamodule_for(
    dataset: str,
    root_dir: str,
    frames_per_clip: int,
    batch_size: int,
    num_workers: int = 4,
    frame_cache_dir: Optional[str] = None,
    resize: int = 224,
    cache_dir: Optional[str] = None,
    # Diving48 specific overrides
    diving48_train_json: Optional[str] = None,
    diving48_test_json: Optional[str] = None,
    diving48_val_ratio: float = 0.1,
    use_test_as_val: bool = False,
) -> VideoDataModule:
    """Factory helper to create a ``VideoDataModule`` for built-in datasets.

    Supported dataset identifiers (case-insensitive):
        - 'hmdb51'
        - 'diving48', 'div48'

    For HMDB51, only ``root_dir`` (dataset root) is required.
    For Diving48, either provide explicit JSON label file paths or rely on
    repository defaults: ``src/core/constant/Div48/*.json``.
    """
    ds = dataset.lower()

    if ds == "hmdb51":
        annos = prepare_hmdb51_annotations(root_dir, cache_dir=cache_dir)
        dm = VideoDataModule(
            data_root=root_dir,
            train_csv=annos["train"],
            val_csv=None,
            test_csv=annos["test"],
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
            num_workers=num_workers,
            frame_cache_dir=frame_cache_dir,
            resize=resize,
            use_test_as_val=use_test_as_val,
        )
        return dm

    if ds in {"diving48", "div48"}:
        # Resolve default JSON paths if not provided
        if diving48_train_json is None or diving48_test_json is None:
            base = Path(__file__).parent / "constant" / "Div48"
            if diving48_train_json is None:
                diving48_train_json = str(base / "Diving48_V2_train.json")
            if diving48_test_json is None:
                diving48_test_json = str(base / "Diving48_V2_test.json")
        annos = prepare_diving48_annotations(
            rgb_root=root_dir,
            train_json=diving48_train_json,
            test_json=diving48_test_json,
            cache_dir=cache_dir,
            val_ratio=diving48_val_ratio,
        )
        dm = VideoDataModule(
            data_root=root_dir,
            train_csv=annos["train"],
            val_csv=annos["val"],
            test_csv=annos["test"],
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
            num_workers=num_workers,
            frame_cache_dir=frame_cache_dir,
            resize=resize,
            use_test_as_val=use_test_as_val,
        )
        return dm

    raise ValueError(f"Unsupported dataset: {dataset}. Supported: hmdb51, diving48")


__all__ = [
    "VideoRecord",
    "VideoCSVAnnotation",
    "FrameCache",
    "SimpleVideoDataset",
    "VideoDataModule",
    "prepare_hmdb51_annotations",
    "prepare_diving48_annotations",
    "create_datamodule_for",
]
