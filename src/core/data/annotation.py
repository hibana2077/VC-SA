#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video annotation parser for CSV files.
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Optional

from .record import VideoRecord


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


__all__ = ["VideoCSVAnnotation"]
