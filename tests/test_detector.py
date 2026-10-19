"""
Automated unit tests for PersonDetector module.
"""

from pathlib import Path
import numpy as np
import pytest

from src.detector import PersonDetector, Detection


def test_detector_initialization():
    detector = PersonDetector("models/yolov8n.pt", conf_threshold=0.3)
    assert detector.model is not None
    assert detector.conf_threshold == 0.3
    assert detector.target_classes == [0]
    assert detector.device in ["cpu", "cuda"]


def test_detector_empty_frame():
    detector = PersonDetector("models/yolov8n.pt")
    # Empty black frame
    black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dets = detector.detect(black_frame)
    # Should safely return an empty list or list of detections without crashing
    assert isinstance(dets, list)


def test_detection_dataclass():
    det = Detection(
        bbox=(10, 20, 30, 60),
        confidence=0.85432,
        class_id=0,
        class_name="person",
        center=(20, 40),
        footpoint=(20, 60),
    )
    d_dict = det.to_dict()
    assert d_dict["bbox"] == [10, 20, 30, 60]
    assert d_dict["confidence"] == 0.8543
    assert d_dict["footpoint"] == [20, 60]


def test_aerial_domain_notes():
    notes = PersonDetector.get_aerial_domain_notes()
    assert "VisDrone" in notes
    assert "aerial" in notes.lower()
