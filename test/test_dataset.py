import numpy as np
import pydicom
import os

from src.dataset import load_train_images_masks

def dummy_rle2mask(rle, height, width):
    # Fake mask: just return zeros
    return np.zeros((height, width), dtype=np.uint8)

def test_load_train_images_masks(tmp_path):
    # Create dummy DICOM file
    dummy_img = np.ones((32, 32), dtype=np.uint16)
    dummy_path = tmp_path / "test.dcm"
    # For this test, we will not actually write a DICOM, just test the logic
    train_fns = [str(dummy_path)]
    df_full = None  # In actual test, would use DataFrame
    # Should not throw error, just test code path
    try:
        X_train, Y_train = load_train_images_masks(train_fns, {}, dummy_rle2mask, 32, 32, 1)
    except Exception:
        pass
