"""
Automated tests for Data Export module.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.export import DataExporter


def test_export_frame_csv(tmp_path):
    records = [
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "observed_person_count": 5,
            "detection_count": 5,
            "tracked_person_count": 5,
            "processing_time_seconds": 0.045,
            "is_sampled_frame": False,
            "movement_summary": {"dominant_direction": "SE", "avg_speed_px_per_sec": 12.4},
            "detection_details": [{"bbox": [10, 10, 30, 40], "confidence": 0.88}],
            "tracking_details": [{"track_id": 1, "speed_px_per_sec": 12.4}],
            "event_flags": [],
        }
    ]

    out_csv = tmp_path / "test_frames.csv"
    res_path = DataExporter.export_frame_csv(records, out_csv)

    assert Path(res_path).exists()
    df = pd.read_csv(res_path)
    assert len(df) == 1
    assert df["observed_person_count"].iloc[0] == 5
    assert df["dominant_direction"].iloc[0] == "SE"
    assert "detection_details" in df.columns


def test_export_complete_json(tmp_path):
    metadata = {"width": 640, "height": 480, "fps": 30.0, "duration_seconds": 10.0}
    summary_stats = {"peak_observed_count": 15, "mean_observed_count": 8.2}
    cumulative_mov = {"dominant_movement_direction": "E"}
    events = [{"event_id": 1, "title": "Surge", "severity": "WARNING"}]
    line_crossing = {"total_in": 4, "total_out": 2}
    records = []
    suggestions = [{"priority": "INFO", "recommendation": "Wide overview"}]

    out_json = tmp_path / "analytics.json"
    res_path = DataExporter.export_complete_json(
        metadata=metadata,
        summary_statistics=summary_stats,
        cumulative_movement=cumulative_mov,
        events=events,
        line_crossing_summary=line_crossing,
        frame_records=records,
        camera_suggestions=suggestions,
        output_json_path=out_json,
    )

    assert Path(res_path).exists()
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["project"] == "AI Crowd Analytics on Aerial Video"
    assert data["summary_statistics"]["peak_observed_count"] == 15
    assert len(data["events_detected"]) == 1


def test_export_summary_report_markdown(tmp_path):
    metadata = {"filename": "aerial_demo.mp4", "width": 1280, "height": 720, "fps": 30.0, "duration_seconds": 5.0, "total_frames": 150}
    summary_stats = {"peak_observed_count": 22, "mean_observed_count": 14.5, "min_observed_count": 8, "total_processed_frames": 150}
    cumulative_mov = {"dominant_movement_direction": "SE", "overall_avg_speed_px_per_sec": 18.2, "speed_95th_percentile_px_per_sec": 32.1}
    events = []
    suggestions = [{"priority": "ATTENTION", "target_area": "NE Quadrant", "recommendation": "Inspect cluster", "rationale": "High density"}]

    out_md = tmp_path / "report.md"
    res_path = DataExporter.export_summary_report_markdown(
        metadata=metadata,
        summary_stats=summary_stats,
        cumulative_movement=cumulative_mov,
        events=events,
        camera_suggestions=suggestions,
        output_md_path=out_md,
    )

    assert Path(res_path).exists()
    content = Path(res_path).read_text(encoding="utf-8")
    assert "# AI Crowd Analytics on Aerial Video: Operational Summary Report" in content
    assert "Susan Future Technologies (SFT)" in content
    assert "Peak Observed Count" in content
