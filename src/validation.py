"""
Accuracy Validation module for AI Crowd Analytics on Aerial Video.
Enables quantitative comparison between model predictions and manual
ground-truth annotations on selected sample frames.
Calculates MAE, RMSE, Mean Bias, and exports validation tables.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class ValidationSample:
    """Represents a single evaluated frame with predicted and ground-truth counts."""
    frame_number: int
    timestamp_seconds: float
    predicted_count: int
    ground_truth_count: int

    @property
    def error(self) -> int:
        """Prediction error: predicted - ground_truth."""
        return self.predicted_count - self.ground_truth_count

    @property
    def absolute_error(self) -> int:
        return abs(self.error)

    @property
    def percentage_error(self) -> float:
        if self.ground_truth_count == 0:
            return 0.0 if self.predicted_count == 0 else 100.0
        return abs(self.error) / float(self.ground_truth_count) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 2),
            "predicted_count": self.predicted_count,
            "ground_truth_count": self.ground_truth_count,
            "error": self.error,
            "absolute_error": self.absolute_error,
            "percentage_error": round(self.percentage_error, 2),
        }


class AccuracyValidator:
    """
    Computes validation metrics on manually annotated frames.
    """

    def __init__(self):
        self.samples: List[ValidationSample] = []

    def add_sample(
        self,
        frame_number: int,
        timestamp_seconds: float,
        predicted_count: int,
        ground_truth_count: int,
    ):
        """Add or update a validation frame comparison."""
        # Check if frame already exists and update
        for i, s in enumerate(self.samples):
            if s.frame_number == frame_number:
                self.samples[i] = ValidationSample(
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    predicted_count=int(predicted_count),
                    ground_truth_count=int(ground_truth_count),
                )
                return

        self.samples.append(
            ValidationSample(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                predicted_count=int(predicted_count),
                ground_truth_count=int(ground_truth_count),
            )
        )

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate MAE, RMSE, Mean Bias, and count statistics.
        Returns:
            Dictionary with computed metrics and limitations commentary.
        """
        if not self.samples:
            return {
                "num_samples": 0,
                "mae": None,
                "rmse": None,
                "bias": None,
                "mape": None,
                "status": "No ground-truth frames recorded yet.",
                "limitations_note": self.get_limitations_statement(),
            }

        preds = np.array([s.predicted_count for s in self.samples], dtype=float)
        gts = np.array([s.ground_truth_count for s in self.samples], dtype=float)

        errors = preds - gts
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))  # Positive = overcounting, Negative = undercounting

        # MAPE with division safety
        valid_mask = (gts > 0)
        if np.any(valid_mask):
            mape = float(np.mean(abs_errors[valid_mask] / gts[valid_mask]) * 100.0)
        else:
            mape = 0.0

        return {
            "num_samples": len(self.samples),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "bias": round(bias, 2),
            "mape_percent": round(mape, 2),
            "total_predicted": int(np.sum(preds)),
            "total_ground_truth": int(np.sum(gts)),
            "bias_interpretation": (
                "Slight overcounting tendency" if bias > 0.5
                else "Slight undercounting tendency (common with small aerial targets)" if bias < -0.5
                else "Balanced / minimal systemic bias"
            ),
            "limitations_note": self.get_limitations_statement(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert samples into a pandas DataFrame."""
        if not self.samples:
            return pd.DataFrame(
                columns=[
                    "frame_number",
                    "timestamp_seconds",
                    "predicted_count",
                    "ground_truth_count",
                    "error",
                    "absolute_error",
                    "percentage_error",
                ]
            )
        return pd.DataFrame([s.to_dict() for s in self.samples])

    @staticmethod
    def get_limitations_statement() -> str:
        return (
            "Validation Limitations: Manual counting on aerial footage has inherent ambiguity. "
            "High flight altitude, severe perspective foreshortening, canopy/building occlusions, "
            "and shadow artifacts make human annotators prone to +/- 10% variance in dense crowds. "
            "A zero-error threshold is unrealistic in real-world aerial surveillance."
        )

    def reset(self):
        """Clear all validation entries."""
        self.samples.clear()
