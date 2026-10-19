"""
Automated tests for Analytics, Tracking, Counting, Anomalies, and Line Crossing.
"""

import pytest
import numpy as np

from src.detector import Detection
from src.tracker import PersonTracker, TrackedPerson, angle_to_cardinal
from src.crowd_counter import CrowdCounter
from src.movement import MovementAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.line_crossing import LineCrossingCounter, segments_intersect
from src.validation import AccuracyValidator


def test_angle_to_cardinal():
    # In image coordinates (y increases downward):
    # dx > 0, dy = 0 -> East (0 deg)
    assert angle_to_cardinal(0) == "E"
    # dx = 0, dy > 0 -> South (90 deg)
    assert angle_to_cardinal(90) == "S"
    # dx < 0, dy = 0 -> West (180 deg)
    assert angle_to_cardinal(180) == "W"
    # dx = 0, dy < 0 -> North (270 deg or -90 deg)
    assert angle_to_cardinal(270) == "N"
    assert angle_to_cardinal(-90) == "N"


def test_crowd_counter_statistics():
    counter = CrowdCounter(sample_interval=1)
    counter.record_frame(frame_number=0, timestamp_seconds=0.0, detection_count=5, tracked_count=5)
    counter.record_frame(frame_number=1, timestamp_seconds=0.1, detection_count=12, tracked_count=10)
    counter.record_frame(frame_number=2, timestamp_seconds=0.2, detection_count=7, tracked_count=7)

    stats = counter.get_summary_statistics()
    assert stats["total_processed_frames"] == 3
    assert stats["min_observed_count"] == 5
    assert stats["max_observed_count"] == 12
    assert stats["peak_observed_count"] == 12
    assert stats["peak_frame_number"] == 1
    assert stats["mean_observed_count"] == pytest.approx(8.0, 0.1)


def test_anomaly_detector_surge_and_density():
    detector = AnomalyDetector(
        rolling_window_size=5,
        surge_percentage_threshold=40.0,
        high_density_threshold=20,
    )

    # Establish baseline of 5 persons
    for i in range(5):
        events = detector.evaluate_frame(frame_number=i, timestamp_seconds=i * 0.1, current_count=5)
        assert len(events) == 0

    # Sudden surge to 15 persons (+200%)
    events = detector.evaluate_frame(frame_number=5, timestamp_seconds=0.5, current_count=15)
    assert len(events) >= 1
    assert any(e.event_type == "SUDDEN_CROWD_SURGE" for e in events)

    # High density event when count reaches 25 (> 20)
    events_density = detector.evaluate_frame(frame_number=6, timestamp_seconds=0.6, current_count=25)
    assert any(e.event_type == "HIGH_DENSITY_CONGESTION" for e in events_density)


def test_segments_intersect():
    # Two intersecting line segments forming an 'X'
    A = (0, 0)
    B = (10, 10)
    C = (0, 10)
    D = (10, 0)
    assert segments_intersect(A, B, C, D) is True

    # Two parallel non-intersecting line segments
    E = (0, 0)
    F = (10, 0)
    G = (0, 5)
    H = (10, 5)
    assert segments_intersect(E, F, G, H) is False


def test_line_crossing_counter():
    # Virtual line at y = 50 from x=0 to x=100
    counter = LineCrossingCounter(line_start=(0, 50), line_end=(100, 50), in_label="IN", out_label="OUT")

    # Create a track that moves from (50, 40) to (50, 60) across y=50
    track1 = TrackedPerson(
        track_id=1,
        bbox=(45, 55, 55, 65),
        confidence=0.9,
        center=(50, 60),
        footpoint=(50, 65),
        history=[(50, 40, 0.0), (50, 60, 0.1)],
    )

    events = counter.process_tracks([track1], frame_number=1, timestamp_seconds=0.1)
    assert len(events) == 1
    assert counter.in_count + counter.out_count == 1

    summary = counter.get_summary()
    assert summary["total_events"] == 1


def test_accuracy_validator_metrics():
    val = AccuracyValidator()
    # Frame 1: Pred=10, GT=10 (err=0)
    # Frame 2: Pred=12, GT=10 (err=+2)
    # Frame 3: Pred=8, GT=10 (err=-2)
    val.add_sample(frame_number=1, timestamp_seconds=0.0, predicted_count=10, ground_truth_count=10)
    val.add_sample(frame_number=2, timestamp_seconds=1.0, predicted_count=12, ground_truth_count=10)
    val.add_sample(frame_number=3, timestamp_seconds=2.0, predicted_count=8, ground_truth_count=10)

    metrics = val.calculate_metrics()
    assert metrics["num_samples"] == 3
    # MAE = (0 + 2 + 2) / 3 = 1.33
    assert metrics["mae"] == pytest.approx(1.33, 0.05)
    # Bias = (0 + 2 - 2) / 3 = 0.0
    assert metrics["bias"] == pytest.approx(0.0, 0.01)

    df = val.to_dataframe()
    assert len(df) == 3
    assert "error" in df.columns
