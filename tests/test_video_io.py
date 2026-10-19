"""
Automated tests for Video I/O module.
"""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.video_io import get_video_metadata, validate_video, VideoReader, VideoWriter


@pytest.fixture
def sample_video_path(tmp_path):
    """Creates a short synthetic test video for unit testing."""
    video_path = tmp_path / "test_synth.mp4"
    width, height = 320, 240
    fps = 10.0
    num_frames = 15

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving circle
        cv2.circle(frame, (20 + i * 10, 100), 15, (0, 255, 0), -1)
        out.write(frame)
    out.release()

    return video_path


def test_get_video_metadata_valid(sample_video_path):
    meta = get_video_metadata(sample_video_path)
    assert meta["valid"] is True
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 10.0
    assert meta["total_frames"] == 15
    assert meta["duration_seconds"] > 0


def test_get_video_metadata_nonexistent():
    meta = get_video_metadata("non_existent_file.mp4")
    assert meta["valid"] is False
    assert meta["error"] is not None


def test_validate_video(sample_video_path):
    is_valid, msg, meta = validate_video(sample_video_path)
    assert is_valid is True
    assert meta["total_frames"] == 15


def test_video_reader_sampling(sample_video_path):
    # Test sampling interval = 2
    frames_read = []
    with VideoReader(sample_video_path, sample_interval=2) as reader:
        for idx, t_sec, frame, scale in reader.frames():
            frames_read.append((idx, t_sec, frame.shape))

    # In 15 frames (0 to 14), indices: 0, 2, 4, 6, 8, 10, 12, 14 -> 8 frames
    assert len(frames_read) == 8
    assert frames_read[0][0] == 0
    assert frames_read[1][0] == 2


def test_video_writer_and_verify(tmp_path):
    out_path = tmp_path / "test_write.mp4"
    writer = VideoWriter(out_path, fps=10.0, width=160, height=120)

    for _ in range(5):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        writer.write_frame(frame)

    verified, msg = writer.verify_output()
    assert verified is True
    assert "verified successfully" in msg
