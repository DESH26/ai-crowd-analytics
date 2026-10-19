"""
Master Analytics Pipeline for AI Crowd Analytics on Aerial Video.
Coordinates video loading, person detection, tracking, counting,
density heatmaps, movement vector analysis, anomaly detection,
line crossing, visual annotation, and data export.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import cv2
import numpy as np

from src.video_io import VideoReader, VideoWriter, get_video_metadata
from src.detector import PersonDetector
from src.tracker import PersonTracker
from src.crowd_counter import CrowdCounter
from src.heatmap import DensityHeatmap
from src.movement import MovementAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.line_crossing import LineCrossingCounter
from src.camera_suggestions import CameraAdvisor
from src.visualization import VideoAnnotator
from src.export import DataExporter


class VideoAnalyticsPipeline:
    """
    End-to-end processing pipeline converting raw aerial video into
    quantitative crowd insights, heatmaps, annotated video, and reports.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        sample_interval: int = 1,
        max_dimension: Optional[int] = 1280,
        enable_tracking: bool = True,
        enable_heatmap: bool = True,
        enable_movement: bool = True,
        enable_anomalies: bool = True,
        enable_line_crossing: bool = True,
        virtual_line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        surge_percentage_threshold: float = 40.0,
        high_density_threshold: int = 25,
        heatmap_sigma: float = 16.0,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.sample_interval = max(1, int(sample_interval))
        self.max_dimension = max_dimension

        self.enable_tracking = enable_tracking
        self.enable_heatmap = enable_heatmap
        self.enable_movement = enable_movement
        self.enable_anomalies = enable_anomalies
        self.enable_line_crossing = enable_line_crossing
        self.virtual_line_coords = virtual_line_coords
        self.surge_percentage_threshold = surge_percentage_threshold
        self.high_density_threshold = high_density_threshold
        self.heatmap_sigma = heatmap_sigma
        self.device = device

        # Components
        self.detector = PersonDetector(
            model_name_or_path=self.model_name_or_path,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )
        self.tracker = PersonTracker()
        self.counter = CrowdCounter(sample_interval=self.sample_interval)
        self.movement_analyzer = MovementAnalyzer()
        self.anomaly_detector = AnomalyDetector(
            surge_percentage_threshold=self.surge_percentage_threshold,
            high_density_threshold=self.high_density_threshold,
        )
        self.line_crossing_counter = LineCrossingCounter()
        self.camera_advisor = CameraAdvisor()
        self.annotator = VideoAnnotator()
        self.heatmap: Optional[DensityHeatmap] = None

    def process_video(
        self,
        input_video_path: str | Path,
        output_dir: str | Path,
        generate_annotated_video: bool = True,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full analytics pipeline on input video.

        Args:
            input_video_path: Path to aerial video.
            output_dir: Directory where outputs (video, CSV, JSON, heatmap, report) are saved.
            generate_annotated_video: If True, writes annotated video to disk.
            progress_callback: Function called as (current_frame, total_frames, current_fps, status_msg).

        Returns:
            Dictionary containing paths to generated files, summary stats, and metrics.
        """
        input_path = Path(input_video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = get_video_metadata(input_path)
        if not meta["valid"]:
            raise ValueError(f"Cannot process video: {meta['error']}")

        orig_w = meta["width"]
        orig_h = meta["height"]
        fps = meta["fps"]
        total_frames = meta["total_frames"]

        # Initialize heatmap with video dimensions
        self.heatmap = DensityHeatmap(width=orig_w, height=orig_h, sigma=self.heatmap_sigma)

        # Initialize line crossing if configured
        if self.virtual_line_coords is not None:
            self.line_crossing_counter.update_line(
                self.virtual_line_coords[0], self.virtual_line_coords[1]
            )
        else:
            # Default line across center
            default_y = orig_h // 2
            self.line_crossing_counter.update_line((int(orig_w * 0.1), default_y), (int(orig_w * 0.9), default_y))

        # Output file paths
        stem = input_path.stem
        annotated_video_path = out_dir / f"{stem}_annotated.mp4"
        frame_csv_path = out_dir / f"{stem}_frame_analytics.csv"
        complete_json_path = out_dir / f"{stem}_analytics.json"
        heatmap_img_path = out_dir / f"{stem}_density_heatmap.png"
        events_csv_path = out_dir / f"{stem}_events.csv"
        summary_md_path = out_dir / f"{stem}_operational_report.md"

        video_writer: Optional[VideoWriter] = None
        if generate_annotated_video:
            # Annotated video frame rate corresponds to sampling interval
            out_fps = fps / float(self.sample_interval)
            video_writer = VideoWriter(
                output_path=annotated_video_path,
                fps=out_fps,
                width=orig_w,
                height=orig_h,
            )

        frame_records = []
        start_time = time.time()
        processed_count = 0
        current_peak = 0

        # Run frame reader
        with VideoReader(
            input_path,
            sample_interval=self.sample_interval,
            max_dimension=self.max_dimension,
        ) as reader:
            for frame_idx, timestamp_sec, frame, scale_factor in reader.frames():
                f_start = time.time()

                # Scale adjustment: if reader resized frame, coordinates must map back to original frame
                inv_scale = 1.0 / scale_factor if scale_factor != 1.0 else 1.0

                # 1. Detection
                detections = self.detector.detect(frame)
                if scale_factor != 1.0:
                    for d in detections:
                        d.bbox = (
                            int(d.bbox[0] * inv_scale),
                            int(d.bbox[1] * inv_scale),
                            int(d.bbox[2] * inv_scale),
                            int(d.bbox[3] * inv_scale),
                        )
                        d.center = (int(d.center[0] * inv_scale), int(d.center[1] * inv_scale))
                        d.footpoint = (int(d.footpoint[0] * inv_scale), int(d.footpoint[1] * inv_scale))

                # 2. Tracking
                tracks = []
                if self.enable_tracking:
                    tracks = self.tracker.update(detections, timestamp_sec, frame_shape=(orig_h, orig_w))

                # 3. Crowd Counting
                det_count = len(detections)
                observed_count = det_count  # Strict consistency: recorded count matches retained detections
                current_peak = max(current_peak, observed_count)

                # 4. Heatmap Update
                if self.enable_heatmap:
                    self.heatmap.update(detections)

                # 5. Movement Dynamics
                movement_metrics = None
                dominant_dir = "Stationary"
                if self.enable_movement and tracks:
                    movement_metrics = self.movement_analyzer.analyze_frame_movement(tracks)
                    dominant_dir = movement_metrics.dominant_direction

                # 6. Anomaly Detection
                events_in_frame = []
                alert_text = None
                if self.enable_anomalies:
                    events_in_frame = self.anomaly_detector.evaluate_frame(
                        frame_number=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        current_count=observed_count,
                        movement_metrics=movement_metrics,
                    )
                    if events_in_frame:
                        alert_text = events_in_frame[0].title

                # 7. Line Crossing
                crossing_events = []
                if self.enable_line_crossing and tracks:
                    crossing_events = self.line_crossing_counter.process_tracks(
                        tracks=tracks,
                        frame_number=frame_idx,
                        timestamp_seconds=timestamp_sec,
                    )

                f_elapsed = time.time() - f_start

                # Record frame statistics
                count_record = self.counter.record_frame(
                    frame_number=frame_idx,
                    timestamp_seconds=timestamp_sec,
                    detection_count=det_count,
                    tracked_count=len(tracks),
                    processing_time_seconds=f_elapsed,
                )

                # Frame structured data
                frame_data = {
                    "frame_number": frame_idx,
                    "timestamp_seconds": timestamp_sec,
                    "observed_person_count": observed_count,
                    "detection_count": det_count,
                    "tracked_person_count": len(tracks),
                    "processing_time_seconds": round(f_elapsed, 4),
                    "is_sampled_frame": self.sample_interval > 1,
                    "detection_details": [d.to_dict() for d in detections],
                    "tracking_details": [t.to_dict() for t in tracks],
                    "movement_summary": movement_metrics.to_dict() if movement_metrics else {},
                    "event_flags": [e.to_dict() for e in events_in_frame],
                    "line_crossings": [c.to_dict() for c in crossing_events],
                }
                frame_records.append(frame_data)

                # 8. Frame Annotation and Video Writing
                if video_writer is not None:
                    # If frame was resized for inference, resize back to original for writing
                    if frame.shape[0] != orig_h or frame.shape[1] != orig_w:
                        write_canvas = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        write_canvas = frame.copy()

                    annotated_canvas = self.annotator.annotate_frame(
                        frame_bgr=write_canvas,
                        detections=detections,
                        tracks=tracks,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        observed_count=observed_count,
                        peak_count=current_peak,
                        fps=fps,
                        dominant_direction=dominant_dir,
                        virtual_line=(self.line_crossing_counter.line_start, self.line_crossing_counter.line_end) if self.enable_line_crossing else None,
                        line_in_count=self.line_crossing_counter.in_count,
                        line_out_count=self.line_crossing_counter.out_count,
                        alert_text=alert_text,
                    )
                    video_writer.write_frame(annotated_canvas)

                processed_count += 1
                if progress_callback is not None:
                    curr_fps = processed_count / max(0.001, (time.time() - start_time))
                    progress_callback(
                        frame_idx,
                        total_frames,
                        curr_fps,
                        f"Processed frame {frame_idx}/{total_frames} (Observed: {observed_count})",
                    )

        # Finalize video writer
        video_verified = False
        video_status = "Not generated"
        if video_writer is not None:
            video_writer.release()
            video_verified, video_status = video_writer.verify_output()

        # Save standalone heatmap image
        if self.enable_heatmap and self.heatmap is not None:
            self.heatmap.save_heatmap(heatmap_img_path)

        # Generate summary statistics
        summary_stats = self.counter.get_summary_statistics()
        cumulative_movement = self.movement_analyzer.get_cumulative_summary() if self.enable_movement else {}
        all_events = [e.to_dict() for e in self.anomaly_detector.get_all_events()]
        line_crossing_summary = self.line_crossing_counter.get_summary() if self.enable_line_crossing else {}
        quadrant_dist = self.heatmap.get_quadrant_distribution() if self.heatmap is not None else {}

        # Generate Camera Operation Suggestions
        camera_suggestions = [
            s.to_dict()
            for s in self.camera_advisor.generate_suggestions(
                current_count=summary_stats.get("peak_observed_count", 0),
                quadrant_distribution=quadrant_dist,
                movement_metrics=self.movement_analyzer.analyze_frame_movement(self.tracker.active_tracks.values()) if self.tracker.active_tracks else None,
                recent_events=self.anomaly_detector.events,
            )
        ]

        # Export Structured Files
        DataExporter.export_frame_csv(frame_records, frame_csv_path)
        DataExporter.export_complete_json(
            metadata=meta,
            summary_statistics=summary_stats,
            cumulative_movement=cumulative_movement,
            events=all_events,
            line_crossing_summary=line_crossing_summary,
            frame_records=frame_records,
            camera_suggestions=camera_suggestions,
            output_json_path=complete_json_path,
        )
        DataExporter.export_events_csv(all_events, events_csv_path)
        DataExporter.export_summary_report_markdown(
            metadata=meta,
            summary_stats=summary_stats,
            cumulative_movement=cumulative_movement,
            events=all_events,
            camera_suggestions=camera_suggestions,
            output_md_path=summary_md_path,
        )

        total_elapsed = round(time.time() - start_time, 2)
        overall_fps = round(processed_count / max(0.001, total_elapsed), 1)

        return {
            "success": True,
            "total_frames_processed": processed_count,
            "total_elapsed_seconds": total_elapsed,
            "average_processing_fps": overall_fps,
            "annotated_video_path": str(annotated_video_path) if generate_annotated_video else None,
            "annotated_video_verified": video_verified,
            "annotated_video_status": video_status,
            "frame_csv_path": str(frame_csv_path),
            "complete_json_path": str(complete_json_path),
            "heatmap_image_path": str(heatmap_img_path) if self.enable_heatmap else None,
            "events_csv_path": str(events_csv_path),
            "summary_report_path": str(summary_md_path),
            "summary_statistics": summary_stats,
            "cumulative_movement": cumulative_movement,
            "quadrant_distribution": quadrant_dist,
            "camera_suggestions": camera_suggestions,
            "total_events_detected": len(all_events),
            "line_crossing_summary": line_crossing_summary,
        }
