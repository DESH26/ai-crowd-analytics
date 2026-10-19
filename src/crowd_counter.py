"""
Crowd Counter module for AI Crowd Analytics on Aerial Video.
Records and tracks per-frame observed counts, computes aggregate statistics
(min, max, mean, peak), and ensures detection-count consistency.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class FrameCountRecord:
    """Record of crowd count information for a single frame."""
    frame_number: int
    timestamp_seconds: float
    observed_person_count: int
    detection_count: int
    tracked_person_count: int
    is_sampled_frame: bool
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "observed_person_count": self.observed_person_count,
            "detection_count": self.detection_count,
            "tracked_person_count": self.tracked_person_count,
            "is_sampled_frame": self.is_sampled_frame,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
        }


class CrowdCounter:
    """
    Manages crowd counting statistics across all processed frames.
    Guarantees that observed count matches retained person detections.
    """

    def __init__(self, sample_interval: int = 1):
        self.sample_interval = sample_interval
        self.records: List[FrameCountRecord] = []

    def record_frame(
        self,
        frame_number: int,
        timestamp_seconds: float,
        detection_count: int,
        tracked_count: int,
        processing_time_seconds: float = 0.0,
    ) -> FrameCountRecord:
        """
        Record counts for a processed frame.

        Args:
            frame_number: Original video frame index (0-indexed or 1-indexed).
            timestamp_seconds: Video timestamp in seconds.
            detection_count: Number of person detections retained for this frame.
            tracked_count: Number of active tracks in this frame.
            processing_time_seconds: Inference and processing latency.

        Returns:
            Recorded FrameCountRecord instance.
        """
        # Observed count strictly matches detection count retained for the frame
        observed_count = int(detection_count)
        is_sampled = (self.sample_interval > 1)

        record = FrameCountRecord(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            observed_person_count=observed_count,
            detection_count=detection_count,
            tracked_person_count=tracked_count,
            is_sampled_frame=is_sampled,
            processing_time_seconds=processing_time_seconds,
        )
        self.records.append(record)
        return record

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate aggregate summary statistics across all processed frames.
        Returns:
            Dictionary with min, max, mean, peak, total_frames, etc.
        """
        if not self.records:
            return {
                "total_processed_frames": 0,
                "min_observed_count": 0,
                "max_observed_count": 0,
                "mean_observed_count": 0.0,
                "peak_observed_count": 0,
                "peak_frame_number": None,
                "peak_timestamp_seconds": None,
                "sampling_interval": self.sample_interval,
                "is_sampled": self.sample_interval > 1,
            }

        counts = [r.observed_person_count for r in self.records]
        peak_idx = int(np.argmax(counts))
        peak_record = self.records[peak_idx]

        return {
            "total_processed_frames": len(self.records),
            "min_observed_count": int(np.min(counts)),
            "max_observed_count": int(np.max(counts)),
            "mean_observed_count": round(float(np.mean(counts)), 2),
            "peak_observed_count": int(np.max(counts)),
            "peak_frame_number": peak_record.frame_number,
            "peak_timestamp_seconds": round(peak_record.timestamp_seconds, 2),
            "sampling_interval": self.sample_interval,
            "is_sampled": self.sample_interval > 1,
        }

    def get_counts_over_time(self) -> List[Dict[str, Any]]:
        """Return list of time-series points suitable for charting."""
        return [
            {
                "frame": r.frame_number,
                "time": r.timestamp_seconds,
                "count": r.observed_person_count,
                "tracked": r.tracked_person_count,
            }
            for r in self.records
        ]

    def reset(self):
        """Reset internal records."""
        self.records.clear()
