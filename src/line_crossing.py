"""
Virtual Line Crossing and Entry/Exit Counting module for AI Crowd Analytics.
Detects bidirectional crossings of virtual boundary lines or gates,
prevents double-counting, and logs individual crossing events.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np

from src.tracker import TrackedPerson


@dataclass
class CrossingEvent:
    """Represents a single verified line crossing event."""
    track_id: int
    crossing_direction: str  # "IN" (Entry) or "OUT" (Exit)
    frame_number: int
    timestamp_seconds: float
    crossing_point: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "crossing_direction": self.crossing_direction,
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 2),
            "crossing_point": list(self.crossing_point),
        }


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Check whether three points are listed in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """Check if line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def signed_line_side(
    pt: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """
    Computes signed 2D cross product of vector (line_start -> line_end)
    with vector (line_start -> pt).
    > 0 means point is on left side, < 0 means right side.
    """
    return (line_end[0] - line_start[0]) * (pt[1] - line_start[1]) - (
        line_end[1] - line_start[1]
    ) * (pt[0] - line_start[0])


class LineCrossingCounter:
    """
    Tracks individuals crossing a defined 2D virtual line.
    Calculates entry, exit, and net flow statistics.
    """

    def __init__(
        self,
        line_start: Tuple[int, int] = (100, 300),
        line_end: Tuple[int, int] = (700, 300),
        in_label: str = "IN",
        out_label: str = "OUT",
    ):
        """
        Args:
            line_start: (x, y) coordinates of virtual line origin.
            line_end: (x, y) coordinates of virtual line destination.
            in_label: Label for positive cross product crossing.
            out_label: Label for negative cross product crossing.
        """
        self.line_start = line_start
        self.line_end = line_end
        self.in_label = in_label
        self.out_label = out_label

        self.in_count = 0
        self.out_count = 0
        self.events: List[CrossingEvent] = []

        # Keep track of last recorded side for each active track ID
        self.track_last_side: Dict[int, float] = {}
        # Cooldown per track to prevent jitter oscillation crossing
        self.track_last_crossed_frame: Dict[int, int] = {}
        self.cooldown_frames = 10

    def update_line(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """Dynamically update line coordinates."""
        self.line_start = line_start
        self.line_end = line_end

    def process_tracks(
        self,
        tracks: List[TrackedPerson],
        frame_number: int,
        timestamp_seconds: float,
    ) -> List[CrossingEvent]:
        """
        Evaluate currently active tracks against the virtual line.

        Returns:
            List of new CrossingEvents generated in this frame.
        """
        new_events: List[CrossingEvent] = []

        for track in tracks:
            # Need at least 2 historical positions to detect crossing
            if len(track.history) < 2:
                continue

            prev_pt = (track.history[-2][0], track.history[-2][1])
            curr_pt = (track.center[0], track.center[1])

            # Check if trajectory segment crossed the virtual line
            if segments_intersect(prev_pt, curr_pt, self.line_start, self.line_end):
                last_crossed = self.track_last_crossed_frame.get(track.track_id, -999)
                if (frame_number - last_crossed) > self.cooldown_frames:
                    # Determine crossing direction
                    side_prev = signed_line_side(prev_pt, self.line_start, self.line_end)
                    side_curr = signed_line_side(curr_pt, self.line_start, self.line_end)

                    direction = self.in_label if side_curr > 0 else self.out_label

                    if direction == self.in_label:
                        self.in_count += 1
                    else:
                        self.out_count += 1

                    event = CrossingEvent(
                        track_id=track.track_id,
                        crossing_direction=direction,
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                        crossing_point=curr_pt,
                    )
                    self.events.append(event)
                    new_events.append(event)
                    self.track_last_crossed_frame[track.track_id] = frame_number

        return new_events

    def get_summary(self) -> Dict[str, Any]:
        """Return cumulative crossing counts and net flow."""
        return {
            "total_in": self.in_count,
            "total_out": self.out_count,
            "net_flow": self.in_count - self.out_count,
            "total_events": len(self.events),
            "line_coordinates": [list(self.line_start), list(self.line_end)],
            "limitations_note": (
                "Virtual line counting accuracy may be affected by drone camera ego-motion, "
                "perspective foreshortening, occlusions, and track ID switches."
            ),
        }

    def reset(self):
        """Reset all counters and event logs."""
        self.in_count = 0
        self.out_count = 0
        self.events.clear()
        self.track_last_side.clear()
        self.track_last_crossed_frame.clear()
