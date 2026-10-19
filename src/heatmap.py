"""
Crowd Density Heatmap module for AI Crowd Analytics on Aerial Video.
Accumulates spatial detections (centers or footpoints) over time,
applies Gaussian smoothing, normalizes density values, and generates
color-mapped visualizations and overlays.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from src.detector import Detection


class DensityHeatmap:
    """
    Manages spatial accumulation and visualization of crowd density.
    Handles empty videos/frames safely without NaN or divide-by-zero errors.
    """

    def __init__(
        self,
        width: int,
        height: int,
        point_mode: str = "footpoint",
        sigma: float = 18.0,
        colormap: int = cv2.COLORMAP_JET,
        decay_factor: float = 1.0,  # 1.0 = permanent cumulative, < 1.0 = temporal decay
    ):
        """
        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            point_mode: 'footpoint' (recommended for aerial ground plane) or 'center'.
            sigma: Standard deviation for Gaussian smoothing kernel.
            colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
            decay_factor: Multiplier applied before each accumulation (1.0 = full history).
        """
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.point_mode = point_mode
        self.sigma = max(1.0, float(sigma))
        self.colormap = colormap
        self.decay_factor = float(decay_factor)

        # 2D floating-point accumulation grid
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_points_accumulated = 0

    def update(self, detections: List[Detection]):
        """
        Accumulate detected locations into the density grid.

        Args:
            detections: List of Detection objects in current frame.
        """
        if self.decay_factor < 1.0:
            self.grid *= self.decay_factor

        for det in detections:
            if self.point_mode == "footpoint":
                pt = det.footpoint
            else:
                pt = det.center

            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] += 1.0
                self.total_points_accumulated += 1

    def generate_heatmap(self, sigma: Optional[float] = None) -> np.ndarray:
        """
        Generate normalized, color-mapped density heatmap image.

        Args:
            sigma: Optional override for Gaussian blur standard deviation.

        Returns:
            np.ndarray of shape (H, W, 3) in BGR format.
        """
        s = float(sigma) if sigma is not None else self.sigma

        # Handle case with no detections accumulated
        if self.total_points_accumulated == 0 or np.max(self.grid) == 0.0:
            # Return dark blue/black zero heatmap
            blank = np.zeros((self.height, self.width), dtype=np.uint8)
            return cv2.applyColorMap(blank, self.colormap)

        # Apply Gaussian smoothing
        smoothed = gaussian_filter(self.grid, sigma=s)
        max_val = np.max(smoothed)

        if max_val <= 0:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            normalized = np.clip((smoothed / max_val) * 255.0, 0, 255).astype(np.uint8)

        heatmap_bgr = cv2.applyColorMap(normalized, self.colormap)
        return heatmap_bgr

    def generate_overlay(
        self,
        frame_bgr: np.ndarray,
        alpha: float = 0.45,
        min_threshold_ratio: float = 0.05,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Blend heatmap with the original video frame.
        Only overlays where density exceeds min_threshold_ratio of the max density.

        Args:
            frame_bgr: Original frame image (H, W, 3).
            alpha: Heatmap blend weight (0.0 to 1.0).
            min_threshold_ratio: Minimum relative density to show color overlay.
            sigma: Optional override for smoothing.

        Returns:
            Blended frame (H, W, 3) in BGR format.
        """
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.total_points_accumulated == 0 or np.max(self.grid) == 0.0:
            return frame_bgr.copy()

        s = float(sigma) if sigma is not None else self.sigma
        smoothed = gaussian_filter(self.grid, sigma=s)
        max_val = np.max(smoothed)

        if max_val <= 0:
            return frame_bgr.copy()

        norm = np.clip((smoothed / max_val) * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(norm, self.colormap)

        # Create mask where density is noticeable
        thresh_val = int(min_threshold_ratio * 255)
        _, mask = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0

        # Weighted blend
        blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
        # Keep background intact where density is negligible
        result = (blended * mask_3ch + frame_bgr * (1.0 - mask_3ch)).astype(np.uint8)
        return result

    def get_quadrant_distribution(self) -> Dict[str, float]:
        """
        Compute density distribution across four image quadrants:
        - Top-Left (NW)
        - Top-Right (NE)
        - Bottom-Left (SW)
        - Bottom-Right (SE)

        Returns:
            Dictionary mapping quadrant name to percentage of accumulated points.
        """
        if self.total_points_accumulated == 0:
            return {"NW": 25.0, "NE": 25.0, "SW": 25.0, "SE": 25.0}

        mid_y = self.height // 2
        mid_x = self.width // 2

        nw = float(np.sum(self.grid[:mid_y, :mid_x]))
        ne = float(np.sum(self.grid[:mid_y, mid_x:]))
        sw = float(np.sum(self.grid[mid_y:, :mid_x]))
        se = float(np.sum(self.grid[mid_y:, mid_x:]))

        total = nw + ne + sw + se
        if total <= 0:
            return {"NW": 25.0, "NE": 25.0, "SW": 25.0, "SE": 25.0}

        return {
            "NW": round((nw / total) * 100.0, 1),
            "NE": round((ne / total) * 100.0, 1),
            "SW": round((sw / total) * 100.0, 1),
            "SE": round((se / total) * 100.0, 1),
        }

    def save_heatmap(self, output_path: str | Path) -> str:
        """Save standalone heatmap image to disk."""
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        img = self.generate_heatmap()
        cv2.imwrite(str(path_obj), img)
        return str(path_obj)

    def reset(self):
        """Reset density accumulation grid."""
        self.grid.fill(0.0)
        self.total_points_accumulated = 0
