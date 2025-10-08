#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset preparation utilities for fixed (built-in) datasets.

Supports HMDB51 and Diving48 dataset preparation and annotation generation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Optional

from .datamodule import VideoDataModule

# Constants
HMDB51_SPLITS = ["train", "test"]  # validation split removed per user request


def _ensure_cache_dir(cache_dir: Optional[str]) -> Path:
    """Ensure cache directory exists and return Path object."""
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
            base = Path(__file__).parent.parent / "constant" / "Div48"
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
    "prepare_hmdb51_annotations",
    "prepare_diving48_annotations",
    "create_datamodule_for",
]
