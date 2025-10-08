#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick import test for refactored data module.

This script verifies that all imports work correctly after the refactoring.
Run this to ensure backward compatibility is maintained.
"""

def test_imports():
    """Test all import paths."""
    
    print("Testing imports from src.core.data (backward compatibility)...")
    try:
        from src.core.data import (
            VideoRecord,
            VideoCSVAnnotation,
            FrameCache,
            SimpleVideoDataset,
            VideoDataModule,
            prepare_hmdb51_annotations,
            prepare_diving48_annotations,
            create_datamodule_for,
        )
        print("✅ All imports from src.core.data successful")
    except ImportError as e:
        print(f"❌ Import from src.core.data failed: {e}")
        return False
    
    print("\nTesting imports from specific submodules...")
    try:
        from src.core.data.record import VideoRecord as VR
        from src.core.data.annotation import VideoCSVAnnotation as VCA
        from src.core.data.cache import FrameCache as FC
        from src.core.data.dataset import SimpleVideoDataset as SVD
        from src.core.data.datamodule import VideoDataModule as VDM
        from src.core.data.preparation import (
            prepare_hmdb51_annotations as prep_hmdb,
            prepare_diving48_annotations as prep_div48,
            create_datamodule_for as create_dm,
        )
        print("✅ All imports from submodules successful")
    except ImportError as e:
        print(f"❌ Import from submodules failed: {e}")
        return False
    
    print("\nTesting imports from src.core (package level)...")
    try:
        from src.core import (
            VideoRecord,
            VideoDataModule,
        )
        print("✅ Package level imports successful")
    except ImportError as e:
        print(f"❌ Package level import failed: {e}")
        return False
    
    print("\n✅ All import tests passed!")
    print("\n📦 Module structure:")
    print("  src/core/data/")
    print("    ├── __init__.py")
    print("    ├── record.py")
    print("    ├── annotation.py")
    print("    ├── cache.py")
    print("    ├── dataset.py")
    print("    ├── datamodule.py")
    print("    ├── preparation.py")
    print("    └── README.md")
    print("\n💡 All components are properly modularized and accessible!")
    
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")  # Add current directory to path
    
    success = test_imports()
    sys.exit(0 if success else 1)
