"""
Command-Line Interface (CLI) runner for AI Crowd Analytics on Aerial Video.
Allows headless execution of the complete video analytics pipeline.

Usage:
    python run_pipeline.py --video sample_data/aerial_sample_preview.mp4 --sampling 2
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import VideoAnalyticsPipeline
from src.video_io import validate_video
from config import DEFAULT_MODEL_NAME, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Headless AI Crowd Analytics Pipeline on Aerial Video for Susan Future Technologies"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input aerial video (.mp4, .avi, .mov)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help=f"Directory to save outputs (default: {OUTPUTS_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"YOLO model weights name or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Detection confidence threshold (default: {DEFAULT_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"NMS IOU threshold (default: {DEFAULT_IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=1,
        help="Frame sampling interval (1 = process every frame, 2 = every 2nd frame, etc.)",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable annotated video output generation (faster data-only export)",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    is_valid, msg, meta = validate_video(video_path)
    if not is_valid:
        print(f"[ERROR] Video validation failed: {msg}")
        sys.exit(1)

    print(f"[INFO] Video validated: {meta['width']}x{meta['height']} @ {meta['fps']} FPS ({meta['duration_seconds']}s, {meta['total_frames']} frames)")
    print(f"[INFO] Initializing pipeline with model={args.model}, conf={args.conf}, sampling_interval={args.sampling}...")

    pipeline = VideoAnalyticsPipeline(
        model_name_or_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        sample_interval=args.sampling,
    )

    def cli_progress(curr, total, fps, msg):
        pct = (curr / total * 100.0) if total > 0 else 0.0
        sys.stdout.write(f"\r[PROGRESS] {pct:5.1f}% | Frame {curr}/{total} | {fps:.1f} FPS | {msg}")
        sys.stdout.flush()

    print("[INFO] Starting video processing...")
    results = pipeline.process_video(
        input_video_path=video_path,
        output_dir=args.output_dir,
        generate_annotated_video=not args.no_video,
        progress_callback=cli_progress,
    )
    print("\n[SUCCESS] Processing complete!")
    print("=" * 60)
    print(f"Total Frames Analyzed : {results['total_frames_processed']}")
    print(f"Elapsed Time           : {results['total_elapsed_seconds']}s ({results['average_processing_fps']} FPS)")
    print(f"Peak Observed Count   : {results['summary_statistics']['peak_observed_count']}")
    print(f"Mean Observed Count   : {results['summary_statistics']['mean_observed_count']}")
    print(f"Dominant Flow Heading : {results['cumulative_movement'].get('dominant_movement_direction', 'N/A')}")
    print(f"Events Flagged        : {results['total_events_detected']}")
    print(f"Annotated Video       : {results['annotated_video_path']} (Verified: {results['annotated_video_verified']})")
    print(f"Frame CSV Data        : {results['frame_csv_path']}")
    print(f"Structured JSON       : {results['complete_json_path']}")
    print(f"Density Heatmap Image : {results['heatmap_image_path']}")
    print(f"Executive Report      : {results['summary_report_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
