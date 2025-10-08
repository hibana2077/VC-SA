#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data handling modules for Video Action Recognition.

This module provides backward compatibility by re-exporting all classes
and functions from the modular data subpackage.

DEPRECATED: This file is maintained for backward compatibility.
New code should import directly from `src.core.data.*` submodules.
"""

from __future__ import annotations

# Import everything from the new modular structure
from .data import (
    VideoRecord,
    VideoCSVAnnotation,
    FrameCache,
    SimpleVideoDataset,
    VideoDataModule,
    prepare_hmdb51_annotations,
    prepare_diving48_annotations,
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
    "create_datamodule_for",
]
