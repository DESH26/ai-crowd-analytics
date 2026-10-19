"""
Automated tests for Density Heatmap module.
"""

from pathlib import Path
import numpy as np
import pytest

from src.heatmap import DensityHeatmap
from src.detector import Detection


def test_heatmap_empty_initialization():
    hm = DensityHeatmap(width=200, height=150, sigma=10.0)
    assert hm.total_points_accumulated == 0

    # Generating heatmap with zero points should not crash or produce NaNs
    img = hm.generate_heatmap()
    assert img.shape == (150, 200, 3)
    assert img.dtype == np.uint8
    assert not np.isnan(img).any()


def test_heatmap_accumulation_and_normalization():
    hm = DensityHeatmap(width=100, height=100, sigma=5.0)

    # Create dummy detections
    dets = [
        Detection(bbox=(20, 20, 40, 50), confidence=0.8, class_id=0, class_name="person", center=(30, 35), footpoint=(30, 50)),
        Detection(bbox=(25, 25, 45, 55), confidence=0.9, class_id=0, class_name="person", center=(35, 40), footpoint=(35, 55)),
    ]

    hm.update(dets)
    assert hm.total_points_accumulated == 2

    img = hm.generate_heatmap()
    assert img.shape == (100, 100, 3)
    # Ensure values are within standard 8-bit image range
    assert img.min() >= 0
    assert img.max() <= 255


def test_heatmap_overlay_shape():
    hm = DensityHeatmap(width=100, height=80)
    frame = np.ones((80, 100, 3), dtype=np.uint8) * 128

    dets = [
        Detection(bbox=(10, 10, 30, 30), confidence=0.85, class_id=0, class_name="person", center=(20, 20), footpoint=(20, 30))
    ]
    hm.update(dets)

    overlay = hm.generate_overlay(frame, alpha=0.5)
    assert overlay.shape == (80, 100, 3)
    assert overlay.dtype == np.uint8


def test_heatmap_quadrant_distribution():
    hm = DensityHeatmap(width=100, height=100)
    # Place detection in Top-Left (NW) quadrant
    dets_nw = [
        Detection(bbox=(10, 10, 20, 20), confidence=0.9, class_id=0, class_name="person", center=(15, 15), footpoint=(15, 20))
    ]
    hm.update(dets_nw)

    q = hm.get_quadrant_distribution()
    assert q["NW"] == 100.0
    assert q["NE"] == 0.0
    assert q["SW"] == 0.0
    assert q["SE"] == 0.0


def test_heatmap_save(tmp_path):
    hm = DensityHeatmap(width=60, height=40)
    save_path = tmp_path / "heatmap_test.png"
    hm.save_heatmap(save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
