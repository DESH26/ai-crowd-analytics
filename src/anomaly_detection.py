"""
Anomaly Detection module for AI Crowd Analytics on Aerial Video.
Rule-based and rolling-statistical detection of sudden crowd surges,
sustained high congestion, and unusual velocity spikes.
All events are explicitly designated as potential anomalies requiring human review.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from src.movement import MovementMetrics


@dataclass
class CrowdEvent:
    """Represents a flagged potential crowd event."""
    event_id: int
    event_type: str  # "SUDDEN_CROWD_SURGE", "HIGH_DENSITY_CONGESTION", "VELOCITY_SPIKE", "RAPID_DISPERSAL"
    frame_number: int
    timestamp_seconds: float
    severity: str    # "INFO", "WARNING", "ALERT"
    title: str
    description: str
    supporting_metrics: Dict[str, Any]
    requires_human_review: bool = True
    disclaimer: str = "Potential anomaly flagged for operator review. Not a confirmed emergency."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 2),
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "supporting_metrics": self.supporting_metrics,
            "requires_human_review": self.requires_human_review,
            "disclaimer": self.disclaimer,
        }


class AnomalyDetector:
    """
    Evaluates rolling metrics against configurable statistical baselines
    to flag explainable crowd dynamic events.
    """

    def __init__(
        self,
        rolling_window_size: int = 15,
        surge_percentage_threshold: float = 40.0,
        high_density_threshold: int = 25,
        speed_spike_multiplier: float = 2.0,
        alert_cooldown_frames: int = 15,
    ):
        """
        Args:
            rolling_window_size: Number of past processed frames to compute moving baseline.
            surge_percentage_threshold: % increase in count relative to rolling mean to trigger surge.
            high_density_threshold: Absolute count threshold for congestion warnings.
            speed_spike_multiplier: Ratio of current average speed over rolling speed to trigger spike.
            alert_cooldown_frames: Minimum frames between duplicate event types.
        """
        self.rolling_window_size = max(3, int(rolling_window_size))
        self.surge_percentage_threshold = float(surge_percentage_threshold)
        self.high_density_threshold = int(high_density_threshold)
        self.speed_spike_multiplier = float(speed_spike_multiplier)
        self.alert_cooldown_frames = int(alert_cooldown_frames)

        self.history_counts: List[int] = []
        self.history_speeds: List[float] = []
        self.events: List[CrowdEvent] = []
        self.last_event_frame: Dict[str, int] = {}
        self.event_counter = 1

    def evaluate_frame(
        self,
        frame_number: int,
        timestamp_seconds: float,
        current_count: int,
        movement_metrics: Optional[MovementMetrics] = None,
    ) -> List[CrowdEvent]:
        """
        Analyze current frame metrics against historical baseline.

        Returns:
            List of CrowdEvent instances triggered in this frame.
        """
        new_events: List[CrowdEvent] = []

        # Check rolling count baseline
        if len(self.history_counts) >= self.rolling_window_size:
            baseline_window = self.history_counts[-self.rolling_window_size:]
            rolling_mean_count = float(np.mean(baseline_window))
            rolling_std_count = float(np.std(baseline_window))

            # 1. Sudden crowd surge
            if rolling_mean_count > 2.0:
                pct_increase = ((current_count - rolling_mean_count) / rolling_mean_count) * 100.0
                if (
                    pct_increase >= self.surge_percentage_threshold
                    and (current_count - rolling_mean_count) >= 4
                    and self._check_cooldown("SUDDEN_CROWD_SURGE", frame_number)
                ):
                    event = CrowdEvent(
                        event_id=self.event_counter,
                        event_type="SUDDEN_CROWD_SURGE",
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                        severity="WARNING" if pct_increase < 80.0 else "ALERT",
                        title="Sudden Crowd Surge Detected",
                        description=(
                            f"Observed count rose from baseline of {rolling_mean_count:.1f} to {current_count} "
                            f"(+{pct_increase:.1f}%) within {self.rolling_window_size} frames."
                        ),
                        supporting_metrics={
                            "current_count": current_count,
                            "baseline_mean": round(rolling_mean_count, 1),
                            "percentage_increase": round(pct_increase, 1),
                        },
                    )
                    self.event_counter += 1
                    new_events.append(event)
                    self.last_event_frame["SUDDEN_CROWD_SURGE"] = frame_number

        # 2. High density congestion alert
        if (
            current_count >= self.high_density_threshold
            and self._check_cooldown("HIGH_DENSITY_CONGESTION", frame_number)
        ):
            event = CrowdEvent(
                event_id=self.event_counter,
                event_type="HIGH_DENSITY_CONGESTION",
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                severity="WARNING" if current_count < self.high_density_threshold * 1.5 else "ALERT",
                title="High Crowd Density Threshold Exceeded",
                description=(
                    f"Observed count reached {current_count}, exceeding the configured high-density limit "
                    f"of {self.high_density_threshold} individuals."
                ),
                supporting_metrics={
                    "current_count": current_count,
                    "threshold": self.high_density_threshold,
                },
            )
            self.event_counter += 1
            new_events.append(event)
            self.last_event_frame["HIGH_DENSITY_CONGESTION"] = frame_number

        # 3. Unusual movement speed spike (potential panic or rapid dispersal)
        if movement_metrics and movement_metrics.moving_count >= 3:
            curr_speed = movement_metrics.avg_speed_px_per_sec
            if len(self.history_speeds) >= self.rolling_window_size:
                baseline_speed = float(np.mean(self.history_speeds[-self.rolling_window_size:]))
                if (
                    baseline_speed > 5.0
                    and curr_speed >= baseline_speed * self.speed_spike_multiplier
                    and self._check_cooldown("VELOCITY_SPIKE", frame_number)
                ):
                    event = CrowdEvent(
                        event_id=self.event_counter,
                        event_type="VELOCITY_SPIKE",
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                        severity="WARNING",
                        title="Unusual Movement Speed Spike",
                        description=(
                            f"Average crowd movement speed spiked to {curr_speed:.1f} px/sec "
                            f"(baseline: {baseline_speed:.1f} px/sec across {movement_metrics.moving_count} tracked individuals)."
                        ),
                        supporting_metrics={
                            "current_speed_px_per_sec": round(curr_speed, 1),
                            "baseline_speed_px_per_sec": round(baseline_speed, 1),
                            "moving_count": movement_metrics.moving_count,
                        },
                    )
                    self.event_counter += 1
                    new_events.append(event)
                    self.last_event_frame["VELOCITY_SPIKE"] = frame_number

        # Update historical windows
        self.history_counts.append(current_count)
        if movement_metrics and movement_metrics.moving_count > 0:
            self.history_speeds.append(movement_metrics.avg_speed_px_per_sec)

        self.events.extend(new_events)
        return new_events

    def _check_cooldown(self, event_type: str, current_frame: int) -> bool:
        """Check if enough frames elapsed since last event of this type."""
        last_frame = self.last_event_frame.get(event_type, -9999)
        return (current_frame - last_frame) >= self.alert_cooldown_frames

    def get_all_events(self) -> List[CrowdEvent]:
        """Return all recorded crowd events."""
        return self.events

    def reset(self):
        """Reset internal history and recorded events."""
        self.history_counts.clear()
        self.history_speeds.clear()
        self.events.clear()
        self.last_event_frame.clear()
        self.event_counter = 1
