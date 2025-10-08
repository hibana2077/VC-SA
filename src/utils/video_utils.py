#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video processing utility functions.

This module contains helper functions for video handling and processing.
"""

from __future__ import annotations

import hashlib


def hash_path(p: str) -> str:
    """
    Generate a short hash for a file path.
    
    Useful for creating unique directory names for caching.
    
    Args:
        p: File path string
        
    Returns:
        16-character hexadecimal hash string
    
    Example:
        >>> hash_path("/path/to/video.mp4")
        '3f7a2b8c1d4e5f6a'
    """
    return hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]
