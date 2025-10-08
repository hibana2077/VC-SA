#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple video record container.
"""

from __future__ import annotations


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


__all__ = ["VideoRecord"]
