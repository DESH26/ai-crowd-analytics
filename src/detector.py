"""
Person Detector module for AI Crowd Analytics on Aerial Video.
Wraps Ultralytics YOLO to detect persons in video frames with configurable
confidence, IOU, device, and bounding box representation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch


@dataclass
class Detection:
    """Represents a single detected person."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int]          # (cx, cy)
    footpoint: Tuple[int, int]       # (cx, y2) - ground contact estimate for aerial imagery

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "center": list(self.center),
            "footpoint": list(self.footpoint),
        }


class PersonDetector:
    """
    Detector focused on person identification using Ultralytics YOLO models.
    Supports CPU/CUDA execution, configurable thresholds, and aerial footage notes.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name_or_path: YOLO model name (e.g. 'yolov8n.pt') or local path to .pt file.
            conf_threshold: Minimum detection confidence score (0.0 to 1.0).
            iou_threshold: NMS IOU threshold.
            target_classes: List of class IDs to retain. Defaults to [0] (person in COCO).
            device: 'cuda', 'cpu', or None for automatic selection.
        """
        self.model_name_or_path = model_name_or_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes if target_classes is not None else [0]

        # Automatic device resolution
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the YOLO model via Ultralytics."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name_or_path)
            # Warm up or configure device
            if self.device != "cpu" and torch.cuda.is_available():
                self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{self.model_name_or_path}': {str(e)}. "
                "Ensure ultralytics is installed and model weights are accessible."
            )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run person detection on a single video frame.

        Args:
            frame_bgr: Video frame in BGR format.

        Returns:
            List of Detection objects. Empty list if no persons detected.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Run inference
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes = first_res.boxes.xyxy.cpu().numpy()
        confs = first_res.boxes.conf.cpu().numpy()
        cls_ids = first_res.boxes.cls.cpu().numpy().astype(int)

        names = first_res.names if hasattr(first_res, "names") else {0: "person"}

        h, w = frame_bgr.shape[:2]

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = bbox
            # Clip coordinates to frame boundaries
            x1 = max(0, min(int(round(x1)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            x2 = max(0, min(int(round(x2)), w - 1))
            y2 = max(0, min(int(round(y2)), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            footpoint = (cx, y2)

            class_name = names.get(cls_id, "person")

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                    center=(cx, cy),
                    footpoint=footpoint,
                )
            )

        return detections

    @staticmethod
    def get_aerial_domain_notes() -> str:
        """
        Returns explanatory notes regarding aerial perspective performance limitations.
        """
        return (
            "Note on Aerial Footage: Pretrained COCO models (such as standard YOLOv8n) are trained "
            "primarily on ground-level imagery. In aerial or drone footage, people appear significantly "
            "smaller, top-down, and clustered. While standard YOLO provides robust baseline detections, "
            "counts should be considered estimates. For mission-critical aerial surveillance, fine-tuning "
            "on aerial datasets like VisDrone or TinyPerson is strongly recommended."
        )
