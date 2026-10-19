"""
Video I/O module for AI Crowd Analytics on Aerial Video.
Handles video validation, metadata extraction, frame reading with configurable sampling,
and video writing with cross-platform codec fallback.
"""

import os
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional
import cv2
import numpy as np


def get_video_metadata(video_path: str | Path) -> Dict[str, Any]:
    """
    Extract comprehensive video metadata using OpenCV.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing metadata:
        - valid: bool
        - width: int
        - height: int
        - fps: float
        - total_frames: int
        - duration_seconds: float
        - fourcc: str
        - file_size_mb: float
        - file_extension: str
        - error: Optional[str]
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        return {
            "valid": False,
            "error": f"Video file not found at: {video_path}",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "total_frames": 0,
            "duration_seconds": 0.0,
            "fourcc": "",
            "file_size_mb": 0.0,
            "file_extension": path_obj.suffix.lower(),
        }

    file_size_mb = round(path_obj.stat().st_size / (1024 * 1024), 2)
    cap = cv2.VideoCapture(str(path_obj))

    if not cap.isOpened():
        cap.release()
        return {
            "valid": False,
            "error": f"OpenCV could not open video: {path_obj.name}. The codec may be unsupported or file corrupted.",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "total_frames": 0,
            "duration_seconds": 0.0,
            "fourcc": "",
            "file_size_mb": file_size_mb,
            "file_extension": path_obj.suffix.lower(),
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FourCC codec decoding
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_chars = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    # Duration calculation
    if fps > 0 and total_frames > 0:
        duration_seconds = round(total_frames / fps, 2)
    else:
        duration_seconds = 0.0

    # Read a sample frame to confirm readability
    ret, sample_frame = cap.read()
    cap.release()

    if not ret or sample_frame is None:
        return {
            "valid": False,
            "error": f"Video stream opened but failed to read the initial frame from {path_obj.name}.",
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration_seconds,
            "fourcc": fourcc_chars,
            "file_size_mb": file_size_mb,
            "file_extension": path_obj.suffix.lower(),
        }

    return {
        "valid": True,
        "error": None,
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30.0,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "fourcc": fourcc_chars,
        "file_size_mb": file_size_mb,
        "file_extension": path_obj.suffix.lower(),
    }


def validate_video(video_path: str | Path, supported_exts: Optional[list] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate that the uploaded video exists, has supported format, and can be read.

    Returns:
        (is_valid, message, metadata)
    """
    if supported_exts is None:
        supported_exts = [".mp4", ".avi", ".mov", ".mkv"]

    path_obj = Path(video_path)
    if not path_obj.exists():
        return False, f"File does not exist: {video_path}", {}

    if path_obj.suffix.lower() not in supported_exts:
        return False, f"Unsupported video extension: {path_obj.suffix}. Supported: {supported_exts}", {}

    meta = get_video_metadata(path_obj)
    if not meta["valid"]:
        return False, meta["error"] or "Invalid video stream", meta

    return True, f"Video validated successfully: {meta['width']}x{meta['height']} @ {meta['fps']} FPS ({meta['duration_seconds']}s)", meta


class VideoReader:
    """
    Robust video reader yielding frames with configurable sampling rate.
    Releases OpenCV video resources upon completion or exception.
    """

    def __init__(
        self,
        video_path: str | Path,
        sample_interval: int = 1,
        max_dimension: Optional[int] = None,
    ):
        """
        Args:
            video_path: Path to input video.
            sample_interval: 1 means process every frame, 2 means every 2nd frame, etc.
            max_dimension: Optional max dimension (width or height) to resize for performance.
        """
        self.video_path = str(video_path)
        self.sample_interval = max(1, int(sample_interval))
        self.max_dimension = max_dimension
        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata = get_video_metadata(video_path)
        self.fps = self.metadata["fps"] if self.metadata["fps"] > 0 else 30.0
        self.total_frames = self.metadata["total_frames"]

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        """Release underlying OpenCV VideoCapture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def frames(self) -> Generator[Tuple[int, float, np.ndarray, float], None, None]:
        """
        Yields (frame_idx, timestamp_seconds, frame_bgr, scale_factor) for each sampled frame.
        - frame_idx is 0-indexed original video frame number.
        - scale_factor is ratio of resized frame size to original frame size (1.0 if not resized).
        """
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)

        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            if frame_idx % self.sample_interval == 0:
                scale_factor = 1.0
                h, w = frame.shape[:2]

                if self.max_dimension and max(h, w) > self.max_dimension:
                    scale_factor = self.max_dimension / float(max(h, w))
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                timestamp_sec = round(frame_idx / self.fps, 3)
                yield frame_idx, timestamp_sec, frame, scale_factor

            frame_idx += 1

        self.release()


class VideoWriter:
    """
    Video writer for producing annotated output video with codec fallback.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
    ):
        self.output_path = Path(output_path)
        self.fps = max(1.0, float(fps))
        self.width = int(width)
        self.height = int(height)
        self.writer: Optional[cv2.VideoWriter] = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_writer()

    def _initialize_writer(self):
        """Try standard mp4/avi codecs with fallback."""
        codecs_to_try = [
            ("mp4v", ".mp4"),
            ("avc1", ".mp4"),
            ("H264", ".mp4"),
            ("XVID", ".avi"),
        ]

        # Match extension if possible
        ext = self.output_path.suffix.lower()
        if ext == ".avi":
            codecs_to_try = [("XVID", ".avi"), ("MJPG", ".avi"), ("mp4v", ".mp4")]

        for fourcc_str, candidate_ext in codecs_to_try:
            target_path = self.output_path
            if candidate_ext != ext and candidate_ext != "":
                target_path = self.output_path.with_suffix(candidate_ext)

            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(
                str(target_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )

            if writer.isOpened():
                self.writer = writer
                self.output_path = target_path
                self.codec_used = fourcc_str
                return

        raise RuntimeError(f"Could not initialize cv2.VideoWriter with any tested codecs on {self.output_path}")

    def write_frame(self, frame_bgr: np.ndarray):
        """Write a single frame ensuring matching dimensions."""
        if self.writer is None:
            raise RuntimeError("VideoWriter is not initialized or has been released.")

        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self.writer.write(frame_bgr)

    def release(self):
        """Finalize and close the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def verify_output(self) -> Tuple[bool, str]:
        """
        Reopens the generated video file using OpenCV to verify it was written properly.
        Returns:
            (is_valid, status_message)
        """
        self.release()

        if not self.output_path.exists():
            return False, f"Output file does not exist: {self.output_path}"

        if self.output_path.stat().st_size == 0:
            return False, f"Output file is 0 bytes: {self.output_path}"

        cap = cv2.VideoCapture(str(self.output_path))
        if not cap.isOpened():
            cap.release()
            return False, f"OpenCV failed to reopen output video: {self.output_path}"

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, "Output video opened but no frames could be decoded."

        return True, f"Output video verified successfully ({self.output_path.name}, {round(self.output_path.stat().st_size / 1024, 1)} KB)"
