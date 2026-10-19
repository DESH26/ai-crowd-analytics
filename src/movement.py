"""
Movement Analysis module for AI Crowd Analytics on Aerial Video.
Computes image-space velocity vectors, speed in pixels per second,
cardinal headings, directional distributions, and dominant flow summaries.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

from src.tracker import TrackedPerson


@dataclass
class MovementMetrics:
    """Summary of crowd movement dynamics for a frame or sequence."""
    total_tracked: int
    moving_count: int
    stationary_count: int
    avg_speed_px_per_sec: float
    max_speed_px_per_sec: float
    dominant_direction: str
    direction_counts: Dict[str, int]
    direction_percentages: Dict[str, float]
    disclaimer: str = (
        "Note: Speed and displacement are measured strictly in 2D image-space pixels. "
        "Physical metric speed (meters/sec) requires intrinsic/extrinsic camera calibration "
        "and a planar homography transformation which are not assumed."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tracked": self.total_tracked,
            "moving_count": self.moving_count,
            "stationary_count": self.stationary_count,
            "avg_speed_px_per_sec": round(self.avg_speed_px_per_sec, 2),
            "max_speed_px_per_sec": round(self.max_speed_px_per_sec, 2),
            "dominant_direction": self.dominant_direction,
            "direction_counts": self.direction_counts,
            "direction_percentages": {k: round(v, 1) for k, v in self.direction_percentages.items()},
            "disclaimer": self.disclaimer,
        }


class MovementAnalyzer:
    """
    Analyzes trajectories of tracked individuals to determine flow patterns,
    speed statistics, and directional trends across the crowd.
    """

    CARDINAL_KEYS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "Stationary"]

    def __init__(self, min_speed_threshold_px: float = 3.0):
        """
        Args:
            min_speed_threshold_px: Speed in px/sec below which movement is classified as Stationary.
        """
        self.min_speed_threshold_px = float(min_speed_threshold_px)
        self.cumulative_direction_counts: Dict[str, int] = {k: 0 for k in self.CARDINAL_KEYS}
        self.all_observed_speeds: List[float] = []

    def analyze_frame_movement(self, tracks: List[TrackedPerson]) -> MovementMetrics:
        """
        Compute movement dynamics for the currently active tracks in a frame.

        Args:
            tracks: List of currently active TrackedPerson instances.

        Returns:
            MovementMetrics summary.
        """
        counts = {k: 0 for k in self.CARDINAL_KEYS}
        speeds: List[float] = []

        for track in tracks:
            # Check if moving faster than stationary threshold
            if track.speed_px_per_sec >= self.min_speed_threshold_px:
                d = track.cardinal_direction
                if d in counts:
                    counts[d] += 1
                else:
                    counts["Stationary"] += 1
                speeds.append(track.speed_px_per_sec)
                self.cumulative_direction_counts[d] = self.cumulative_direction_counts.get(d, 0) + 1
                self.all_observed_speeds.append(track.speed_px_per_sec)
            else:
                counts["Stationary"] += 1
                self.cumulative_direction_counts["Stationary"] += 1

        total = len(tracks)
        moving = len(speeds)
        stationary = counts["Stationary"]

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0

        # Percentages
        percentages = {}
        for k in self.CARDINAL_KEYS:
            percentages[k] = (counts[k] / total * 100.0) if total > 0 else 0.0

        # Determine dominant moving direction (excluding stationary if moving exists)
        moving_dirs = {k: v for k, v in counts.items() if k != "Stationary"}
        if moving > 0 and sum(moving_dirs.values()) > 0:
            dominant = max(moving_dirs, key=moving_dirs.get)
        else:
            dominant = "Stationary"

        return MovementMetrics(
            total_tracked=total,
            moving_count=moving,
            stationary_count=stationary,
            avg_speed_px_per_sec=avg_speed,
            max_speed_px_per_sec=max_speed,
            dominant_direction=dominant,
            direction_counts=counts,
            direction_percentages=percentages,
        )

    def get_cumulative_summary(self) -> Dict[str, Any]:
        """
        Generate aggregate movement summary over all processed frames.
        """
        total_observations = sum(self.cumulative_direction_counts.values())
        percentages = {}
        for k, v in self.cumulative_direction_counts.items():
            percentages[k] = (v / total_observations * 100.0) if total_observations > 0 else 0.0

        moving_counts = {k: v for k, v in self.cumulative_direction_counts.items() if k != "Stationary"}
        if sum(moving_counts.values()) > 0:
            dominant = max(moving_counts, key=moving_counts.get)
        else:
            dominant = "Stationary"

        avg_speed = float(np.mean(self.all_observed_speeds)) if self.all_observed_speeds else 0.0
        max_speed = float(np.max(self.all_observed_speeds)) if self.all_observed_speeds else 0.0
        p95_speed = float(np.percentile(self.all_observed_speeds, 95)) if self.all_observed_speeds else 0.0

        return {
            "total_movement_observations": total_observations,
            "cumulative_direction_counts": self.cumulative_direction_counts,
            "cumulative_direction_percentages": {k: round(v, 1) for k, v in percentages.items()},
            "dominant_movement_direction": dominant,
            "overall_avg_speed_px_per_sec": round(avg_speed, 2),
            "overall_max_speed_px_per_sec": round(max_speed, 2),
            "speed_95th_percentile_px_per_sec": round(p95_speed, 2),
        }

    def reset(self):
        """Reset internal history."""
        self.cumulative_direction_counts = {k: 0 for k in self.CARDINAL_KEYS}
        self.all_observed_speeds.clear()
