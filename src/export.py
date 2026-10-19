"""
Data Export module for AI Crowd Analytics on Aerial Video.
Generates frame-level CSV and structured JSON files, event reports,
validation summaries, and methodology documentation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


class DataExporter:
    """
    Exports structured video analytics outputs to standard CSV and JSON formats.
    """

    @staticmethod
    def export_frame_csv(
        frame_records: List[Dict[str, Any]],
        output_csv_path: str | Path,
    ) -> str:
        """
        Export frame-level structured records to CSV (one row per processed frame).
        Nested collections (detections, tracks, events) are serialized to JSON strings.
        """
        path_obj = Path(output_csv_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in frame_records:
            row = {
                "frame_number": r["frame_number"],
                "timestamp_seconds": r["timestamp_seconds"],
                "observed_person_count": r["observed_person_count"],
                "detection_count": r["detection_count"],
                "tracked_person_count": r["tracked_person_count"],
                "processing_time_seconds": r["processing_time_seconds"],
                "is_sampled_frame": r.get("is_sampled_frame", False),
                "dominant_direction": r.get("movement_summary", {}).get("dominant_direction", "Stationary"),
                "avg_speed_px_per_sec": r.get("movement_summary", {}).get("avg_speed_px_per_sec", 0.0),
                "event_flags": json.dumps(r.get("event_flags", [])),
                "movement_summary": json.dumps(r.get("movement_summary", {})),
                "detection_details": json.dumps(r.get("detection_details", [])),
                "tracking_details": json.dumps(r.get("tracking_details", [])),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(str(path_obj), index=False)
        return str(path_obj)

    @staticmethod
    def export_complete_json(
        metadata: Dict[str, Any],
        summary_statistics: Dict[str, Any],
        cumulative_movement: Dict[str, Any],
        events: List[Dict[str, Any]],
        line_crossing_summary: Dict[str, Any],
        frame_records: List[Dict[str, Any]],
        camera_suggestions: List[Dict[str, Any]],
        output_json_path: str | Path,
    ) -> str:
        """
        Export complete run metadata, summary aggregates, and per-frame records to JSON.
        """
        path_obj = Path(output_json_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "project": "AI Crowd Analytics on Aerial Video",
            "organization": "Susan Future Technologies (SFT)",
            "schema_version": "1.0",
            "video_metadata": metadata,
            "summary_statistics": summary_statistics,
            "cumulative_movement": cumulative_movement,
            "line_crossing_summary": line_crossing_summary,
            "events_detected": events,
            "camera_operator_suggestions": camera_suggestions,
            "frame_level_records": frame_records,
        }

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(path_obj)

    @staticmethod
    def export_events_csv(
        events: List[Dict[str, Any]],
        output_csv_path: str | Path,
    ) -> str:
        """Export crowd anomaly and line crossing events table."""
        path_obj = Path(output_csv_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not events:
            df = pd.DataFrame(
                columns=[
                    "event_id",
                    "event_type",
                    "frame_number",
                    "timestamp_seconds",
                    "severity",
                    "title",
                    "description",
                    "supporting_metrics",
                ]
            )
        else:
            flat_events = []
            for e in events:
                flat_events.append({
                    "event_id": e.get("event_id"),
                    "event_type": e.get("event_type"),
                    "frame_number": e.get("frame_number"),
                    "timestamp_seconds": e.get("timestamp_seconds"),
                    "severity": e.get("severity"),
                    "title": e.get("title"),
                    "description": e.get("description"),
                    "supporting_metrics": json.dumps(e.get("supporting_metrics", {})),
                })
            df = pd.DataFrame(flat_events)

        df.to_csv(str(path_obj), index=False)
        return str(path_obj)

    @staticmethod
    def export_summary_report_markdown(
        metadata: Dict[str, Any],
        summary_stats: Dict[str, Any],
        cumulative_movement: Dict[str, Any],
        events: List[Dict[str, Any]],
        camera_suggestions: List[Dict[str, Any]],
        output_md_path: str | Path,
    ) -> str:
        """Generate a concise executive markdown summary report."""
        path_obj = Path(output_md_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# AI Crowd Analytics on Aerial Video: Operational Summary Report",
            "**Client / Reviewer:** Susan Future Technologies (SFT)",
            f"**Video Source:** {metadata.get('filename', 'Aerial Footage')}",
            f"**Resolution:** {metadata.get('width', 0)}x{metadata.get('height', 0)} @ {metadata.get('fps', 0)} FPS",
            f"**Duration:** {metadata.get('duration_seconds', 0.0)} seconds ({metadata.get('total_frames', 0)} frames)",
            "",
            "## 1. Crowd Counting Summary",
            f"- **Peak Observed Count:** {summary_stats.get('peak_observed_count', 0)} persons (at {summary_stats.get('peak_timestamp_seconds', 0.0)}s)",
            f"- **Mean Observed Count:** {summary_stats.get('mean_observed_count', 0.0)} persons",
            f"- **Minimum Observed Count:** {summary_stats.get('min_observed_count', 0)} persons",
            f"- **Frames Analyzed:** {summary_stats.get('total_processed_frames', 0)} (Sampling Interval: {summary_stats.get('sampling_interval', 1)})",
            "",
            "## 2. Movement Dynamics",
            f"- **Dominant Direction:** {cumulative_movement.get('dominant_movement_direction', 'Stationary')}",
            f"- **Average Image Speed:** {cumulative_movement.get('overall_avg_speed_px_per_sec', 0.0)} px/sec",
            f"- **95th Percentile Speed:** {cumulative_movement.get('speed_95th_percentile_px_per_sec', 0.0)} px/sec",
            "",
            "## 3. Flagged Crowd Events (Human Review Required)",
        ]

        if not events:
            lines.append("- No abnormal crowd surges or velocity spikes were detected.")
        else:
            for e in events:
                lines.append(f"- **[{e.get('severity', 'INFO')}] {e.get('title')}** at {e.get('timestamp_seconds', 0.0)}s (Frame {e.get('frame_number', 0)}): {e.get('description')}")

        lines.extend([
            "",
            "## 4. Camera Operation Recommendations (Advisory Only)",
        ])

        for s in camera_suggestions:
            lines.append(f"- **[{s.get('priority')}] {s.get('target_area')}:** {s.get('recommendation')}")
            lines.append(f"  *Rationale:* {s.get('rationale')}")

        lines.extend([
            "",
            "---",
            "*Report generated automatically by SFT Aerial AI Crowd Analytics Pipeline.*",
        ])

        content = "\n".join(lines)
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)

        return str(path_obj)
