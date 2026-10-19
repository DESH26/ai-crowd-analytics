"""
Visualization and Video Annotation module for AI Crowd Analytics on Aerial Video.
Renders professional bounding boxes, tracking IDs, fading trajectory trails,
directional velocity arrows, virtual line gates, and HUD metrics on video frames.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.detector import Detection
from src.tracker import TrackedPerson


class VideoAnnotator:
    """
    Renders high-visibility computer vision overlays onto aerial video frames.
    """

    # Palette
    BOX_COLOR = (46, 204, 113)       # Emerald green
    TRACK_COLOR = (52, 152, 219)     # Blue
    TRAIL_COLOR = (241, 196, 15)     # Yellow
    ARROW_COLOR = (231, 76, 60)      # Red / Orange
    LINE_COLOR = (255, 140, 0)       # Dark orange / Cyan
    HUD_BG_COLOR = (20, 24, 33)      # Dark slate
    TEXT_COLOR = (255, 255, 255)

    def __init__(
        self,
        show_boxes: bool = True,
        show_tracks: bool = True,
        show_trails: bool = True,
        show_arrows: bool = True,
        show_hud: bool = True,
    ):
        self.show_boxes = show_boxes
        self.show_tracks = show_tracks
        self.show_trails = show_trails
        self.show_arrows = show_arrows
        self.show_hud = show_hud

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        tracks: List[TrackedPerson],
        frame_idx: int,
        timestamp_sec: float,
        observed_count: int,
        peak_count: int = 0,
        fps: float = 30.0,
        dominant_direction: str = "Stationary",
        virtual_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        line_in_count: int = 0,
        line_out_count: int = 0,
        alert_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw all visual annotations on the given frame.

        Returns:
            Annotated copy of the frame.
        """
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Trajectory Trails
        if self.show_trails and tracks:
            for track in tracks:
                if len(track.history) > 1:
                    pts = np.array([[pt[0], pt[1]] for pt in track.history], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts], isClosed=False, color=self.TRAIL_COLOR, thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw Movement Arrows
        if self.show_arrows and tracks:
            for track in tracks:
                if track.speed_px_per_sec > 4.0:
                    cx, cy = track.center
                    # Scale arrow length based on speed (clamped)
                    arrow_len = min(40, max(12, int(track.speed_px_per_sec * 0.4)))
                    rad = np.radians(track.heading_degrees)
                    ex = int(cx + arrow_len * np.cos(rad))
                    ey = int(cy + arrow_len * np.sin(rad))
                    # Clamp
                    ex = max(0, min(w - 1, ex))
                    ey = max(0, min(h - 1, ey))
                    cv2.arrowedLine(
                        canvas,
                        (cx, cy),
                        (ex, ey),
                        self.ARROW_COLOR,
                        2,
                        tipLength=0.35,
                        line_type=cv2.LINE_AA,
                    )

        # 3. Draw Bounding Boxes and ID Tags
        if self.show_boxes or self.show_tracks:
            # Map detections to active track IDs if matched
            track_map = {t.center: t for t in tracks}

            for det in detections:
                x1, y1, x2, y2 = det.bbox

                # Box outline
                if self.show_boxes:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BOX_COLOR, 2, lineType=cv2.LINE_AA)

                # Badge label
                if self.show_tracks:
                    # Find track nearest to this detection center
                    matched_tid = None
                    for t in tracks:
                        if abs(t.center[0] - det.center[0]) < 15 and abs(t.center[1] - det.center[1]) < 15:
                            matched_tid = t.track_id
                            break

                    if matched_tid is not None:
                        label = f"ID:{matched_tid} {det.confidence:.2f}"
                        badge_bg = self.TRACK_COLOR
                    else:
                        label = f"P {det.confidence:.2f}"
                        badge_bg = (100, 100, 100)

                    # Text background pill
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.45
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
                    by1 = max(0, y1 - th - 6)
                    by2 = y1
                    bx2 = min(w, x1 + tw + 6)
                    cv2.rectangle(canvas, (x1, by1), (bx2, by2), badge_bg, -1)
                    cv2.putText(canvas, label, (x1 + 3, y1 - 4), font, scale, self.TEXT_COLOR, thickness, cv2.LINE_AA)

        # 4. Draw Virtual Counting Line if enabled
        if virtual_line is not None:
            p1, p2 = virtual_line
            # Shadow line
            cv2.line(canvas, p1, p2, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            # Main neon line
            cv2.line(canvas, p1, p2, self.LINE_COLOR, 2, lineType=cv2.LINE_AA)

            # End caps
            cv2.circle(canvas, p1, 5, self.LINE_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p2, 5, self.LINE_COLOR, -1, lineType=cv2.LINE_AA)

            # Center floating badge
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            gate_label = f"IN: {line_in_count} | OUT: {line_out_count}"
            (gw, gh), _ = cv2.getTextSize(gate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (mx - gw // 2 - 6, my - gh - 6), (mx + gw // 2 + 6, my + 6), (20, 20, 20), -1)
            cv2.rectangle(canvas, (mx - gw // 2 - 6, my - gh - 6), (mx + gw // 2 + 6, my + 6), self.LINE_COLOR, 1)
            cv2.putText(canvas, gate_label, (mx - gw // 2, my - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Draw Top HUD Banner
        if self.show_hud:
            banner_h = 42
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), self.HUD_BG_COLOR, -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

            mins = int(timestamp_sec // 60)
            secs = timestamp_sec % 60
            time_str = f"{mins:02d}:{secs:04.1f}"

            hud_text = (
                f"SFT AERIAL CROWD AI  |  FRAME: {frame_idx:04d}  |  TIME: {time_str}  |  "
                f"CROWD: {observed_count}  |  PEAK: {peak_count}  |  FLOW: {dominant_direction}"
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, hud_text, (15, 26), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 6. Draw Alert Banner if active
        if alert_text:
            ab_h = 32
            ab_y1 = h - ab_h - 10
            ab_y2 = h - 10
            overlay = canvas.copy()
            cv2.rectangle(overlay, (10, ab_y1), (w - 10, ab_y2), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            cv2.putText(
                canvas,
                f"ALERT: {alert_text} (Review Required)",
                (25, ab_y2 - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas
