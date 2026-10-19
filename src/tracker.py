"""
Multi-Object Tracking module for AI Crowd Analytics on Aerial Video.
Tracks detected individuals across frames using ByteTrack / IoU-Centroid association,
maintaining track IDs, historical trajectories, and velocity estimations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import math
import warnings

# Suppress supervision ByteTrack future deprecation notices
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

from src.detector import Detection


@dataclass
class TrackedPerson:
    """Represents an active track of an individual across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    footpoint: Tuple[int, int]
    # History of (center_x, center_y, timestamp_seconds)
    history: List[Tuple[int, int, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/sec
    speed_px_per_sec: float = 0.0
    heading_degrees: float = 0.0  # -180 to 180, 0 is East
    cardinal_direction: str = "Stationary"
    lost_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "center": list(self.center),
            "footpoint": list(self.footpoint),
            "speed_px_per_sec": round(float(self.speed_px_per_sec), 2),
            "heading_degrees": round(float(self.heading_degrees), 1),
            "cardinal_direction": self.cardinal_direction,
            "trail_length": len(self.history),
        }


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def angle_to_cardinal(degrees: float) -> str:
    """
    Convert an angle in degrees (-180 to 180, where 0=East, 90=South in image space)
    to a compass cardinal direction.
    Image coordinates: y increases downwards!
    dx > 0, dy = 0 -> East (0 deg)
    dx = 0, dy > 0 -> South (+90 deg)
    dx = 0, dy < 0 -> North (-90 deg)
    dx < 0, dy = 0 -> West (180 deg)
    """
    # Normalize angle to [0, 360)
    norm_deg = degrees % 360
    # Compass mapping:
    # North is 270 (up), South is 90 (down), East is 0, West is 180
    directions = [
        ("E", 0),
        ("SE", 45),
        ("S", 90),
        ("SW", 135),
        ("W", 180),
        ("NW", 225),
        ("N", 270),
        ("NE", 315),
    ]

    best_dir = "E"
    min_diff = 360.0
    for d_name, d_angle in directions:
        diff = abs(norm_deg - d_angle)
        diff = min(diff, 360 - diff)
        if diff < min_diff:
            min_diff = diff
            best_dir = d_name

    return best_dir


class PersonTracker:
    """
    Robust multi-object tracker for aerial video person tracking.
    Uses ByteTrack association with supervision when available, or a resilient
    built-in IoU-Centroid association tracker.
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        iou_threshold: float = 0.25,
        max_trail_length: int = 25,
        use_supervision: bool = True,
    ):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.max_trail_length = max_trail_length
        self.active_tracks: Dict[int, TrackedPerson] = {}
        self.next_track_id = 1
        self.sv_tracker = None

        if use_supervision:
            try:
                import supervision as sv
                self.sv_tracker = sv.ByteTrack(
                    track_activation_threshold=0.25,
                    lost_track_buffer=max_lost_frames,
                    minimum_matching_threshold=iou_threshold,
                    frame_rate=30,
                )
            except Exception:
                self.sv_tracker = None

    def update(
        self,
        detections: List[Detection],
        timestamp_sec: float,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[TrackedPerson]:
        """
        Update tracker with new frame detections.

        Args:
            detections: List of detections in current frame.
            timestamp_sec: Current video time in seconds.
            frame_shape: (height, width) of the frame.

        Returns:
            List of active TrackedPerson instances.
        """
        if self.sv_tracker is not None:
            return self._update_supervision(detections, timestamp_sec)
        else:
            return self._update_internal(detections, timestamp_sec)

    def _update_supervision(
        self,
        detections: List[Detection],
        timestamp_sec: float,
    ) -> List[TrackedPerson]:
        """Update using Supervision ByteTrack implementation."""
        import supervision as sv

        if len(detections) == 0:
            # Empty detection update
            empty_sv_det = sv.Detections.empty()
            self.sv_tracker.update_with_detections(empty_sv_det)
            # Increment lost frames for existing tracks
            current_active = []
            for tid in list(self.active_tracks.keys()):
                self.active_tracks[tid].lost_frames += 1
                if self.active_tracks[tid].lost_frames > self.max_lost_frames:
                    del self.active_tracks[tid]
            return list(self.active_tracks.values())

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self.sv_tracker.update_with_detections(sv_detections)
        updated_tracks: List[TrackedPerson] = []
        seen_tids = set()

        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                box = tuple(int(round(coord)) for coord in tracked.xyxy[i])
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.8
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                footpoint = (cx, box[3])

                track_person = self.active_tracks.get(tid)
                if track_person is None:
                    track_person = TrackedPerson(
                        track_id=tid,
                        bbox=box,
                        confidence=conf,
                        center=(cx, cy),
                        footpoint=footpoint,
                        history=[(cx, cy, timestamp_sec)],
                    )
                    self.active_tracks[tid] = track_person
                else:
                    track_person.bbox = box
                    track_person.confidence = conf
                    track_person.center = (cx, cy)
                    track_person.footpoint = footpoint
                    track_person.lost_frames = 0
                    track_person.history.append((cx, cy, timestamp_sec))
                    if len(track_person.history) > self.max_trail_length:
                        track_person.history.pop(0)

                self._calculate_movement(track_person)
                updated_tracks.append(track_person)
                seen_tids.add(tid)

        # Increment lost frames for unseen tracks
        for tid in list(self.active_tracks.keys()):
            if tid not in seen_tids:
                self.active_tracks[tid].lost_frames += 1
                if self.active_tracks[tid].lost_frames > self.max_lost_frames:
                    del self.active_tracks[tid]

        return updated_tracks

    def _update_internal(
        self,
        detections: List[Detection],
        timestamp_sec: float,
    ) -> List[TrackedPerson]:
        """Fallback association algorithm based on IoU and Euclidean distance."""
        active_ids = list(self.active_tracks.keys())
        det_indices = list(range(len(detections)))
        matched_pairs = []

        if len(active_ids) > 0 and len(detections) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(active_ids), len(detections)), dtype=float)
            for i, tid in enumerate(active_ids):
                track = self.active_tracks[tid]
                for j, det in enumerate(detections):
                    iou = compute_iou(track.bbox, det.bbox)
                    # Also consider distance between centers
                    dx = track.center[0] - det.center[0]
                    dy = track.center[1] - det.center[1]
                    dist = math.hypot(dx, dy)
                    # Score combines IoU and center proximity
                    score = iou + max(0.0, 1.0 - (dist / 100.0)) * 0.5
                    cost_matrix[i, j] = score

            # Greedy matching
            matched_tracks = set()
            matched_dets = set()

            flat_indices = np.argsort(-cost_matrix, axis=None)
            for idx in flat_indices:
                t_idx, d_idx = divmod(int(idx), len(detections))
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                if cost_matrix[t_idx, d_idx] >= self.iou_threshold:
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)
                    matched_pairs.append((active_ids[t_idx], detections[d_idx]))

        updated_tracks: List[TrackedPerson] = []
        matched_tids = set()

        for tid, det in matched_pairs:
            track = self.active_tracks[tid]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.center = det.center
            track.footpoint = det.footpoint
            track.lost_frames = 0
            track.history.append((det.center[0], det.center[1], timestamp_sec))
            if len(track.history) > self.max_trail_length:
                track.history.pop(0)

            self._calculate_movement(track)
            updated_tracks.append(track)
            matched_tids.add(tid)

        # Initialize new tracks for unmatched detections
        matched_det_objs = [pair[1] for pair in matched_pairs]
        for det in detections:
            if det not in matched_det_objs:
                new_tid = self.next_track_id
                self.next_track_id += 1
                new_track = TrackedPerson(
                    track_id=new_tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    center=det.center,
                    footpoint=det.footpoint,
                    history=[(det.center[0], det.center[1], timestamp_sec)],
                )
                self.active_tracks[new_tid] = new_track
                updated_tracks.append(new_track)

        # Handle unmatched active tracks
        for tid in list(self.active_tracks.keys()):
            if tid not in matched_tids and tid not in [t.track_id for t in updated_tracks]:
                self.active_tracks[tid].lost_frames += 1
                if self.active_tracks[tid].lost_frames > self.max_lost_frames:
                    del self.active_tracks[tid]

        return updated_tracks

    def _calculate_movement(self, track: TrackedPerson):
        """
        Calculates image-space displacement, velocity, speed, and cardinal direction.
        Note: Image-space speed is in pixels per second.
        """
        if len(track.history) < 2:
            track.velocity = (0.0, 0.0)
            track.speed_px_per_sec = 0.0
            track.heading_degrees = 0.0
            track.cardinal_direction = "Stationary"
            return

        # Take position from ~3-5 points back for smoothing
        prev_idx = max(0, len(track.history) - 4)
        prev_x, prev_y, prev_t = track.history[prev_idx]
        curr_x, curr_y, curr_t = track.history[-1]

        dt = curr_t - prev_t
        dx = curr_x - prev_x
        dy = curr_y - prev_y

        dist = math.hypot(dx, dy)

        if dt > 0.001 and dist > 2.0:  # Noise threshold 2 pixels
            vx = dx / dt
            vy = dy / dt
            speed = dist / dt
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            track.velocity = (vx, vy)
            track.speed_px_per_sec = speed
            track.heading_degrees = angle_deg
            track.cardinal_direction = angle_to_cardinal(angle_deg)
        else:
            track.velocity = (0.0, 0.0)
            track.speed_px_per_sec = 0.0
            track.cardinal_direction = "Stationary"

    def reset(self):
        """Reset all tracking states."""
        self.active_tracks.clear()
        self.next_track_id = 1
        if self.sv_tracker is not None:
            self.sv_tracker.reset()
