#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data handling package for Video Action Recognition.

This package provides modular components for video data loading,
caching, annotation parsing, and dataset preparation.
"""

from .record import VideoRecord
from .annotation import VideoCSVAnnotation
from .cache import FrameCache
from .dataset import SimpleVideoDataset
from .datamodule import VideoDataModule
from .preparation import (
    prepare_hmdb51_annotations,
    prepare_diving48_annotations,
    prepare_ssv2_annotations,
    create_datamodule_for,
)

__all__ = [
    "VideoRecord",
    "VideoCSVAnnotation",
    "FrameCache",
    "SimpleVideoDataset",
    "VideoDataModule",
    "prepare_hmdb51_annotations",
    "prepare_diving48_annotations",
    "prepare_ssv2_annotations",
    "create_datamodule_for",
]
