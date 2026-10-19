"""
Camera Operation Suggestion Module for AI Crowd Analytics on Aerial Video.
Synthesizes spatial density concentrations, directional crowd flow,
and anomaly alerts to produce explainable guidance for drone/PTZ camera operators.

IMPORTANT: Clearly distinguished from automatic physical camera control.
All outputs are strictly advisory recommendations for human operators.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.movement import MovementMetrics
from src.anomaly_detection import CrowdEvent


@dataclass
class CameraSuggestion:
    """An explainable advisory recommendation for camera operators."""
    priority: str          # "INFO", "ATTENTION", "ACTION_RECOMMENDED"
    suggestion_type: str   # "CONCENTRATION_ZOOM", "FLOW_TRACKING", "SURGE_INSPECTION", "WIDE_SURVEILLANCE"
    target_area: str       # "North-East Quadrant", "Center Field", etc.
    recommendation: str    # Clear operational advice
    rationale: str         # Explainable data-driven reason
    disclaimer: str = (
        "Advisory Guidance Only: This system does not control drone hardware, flight plans, "
        "or physical PTZ actuators. All flight and camera operations require pilot authorization."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "suggestion_type": self.suggestion_type,
            "target_area": self.target_area,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "disclaimer": self.disclaimer,
        }


class CameraAdvisor:
    """
    Generates explainable, context-aware suggestions for aerial camera operators
    based on real-time and cumulative crowd analytics.
    """

    QUADRANT_LABELS = {
        "NW": "North-West (Top-Left)",
        "NE": "North-East (Top-Right)",
        "SW": "South-West (Bottom-Left)",
        "SE": "South-East (Bottom-Right)",
    }

    def generate_suggestions(
        self,
        current_count: int,
        quadrant_distribution: Dict[str, float],
        movement_metrics: Optional[MovementMetrics] = None,
        recent_events: Optional[List[CrowdEvent]] = None,
    ) -> List[CameraSuggestion]:
        """
        Produce real-time operator recommendations.
        """
        suggestions: List[CameraSuggestion] = []

        # 1. Spatial Hotspot Analysis
        if current_count >= 5:
            max_quad = max(quadrant_distribution, key=quadrant_distribution.get)
            max_pct = quadrant_distribution[max_quad]

            if max_pct >= 40.0:
                quad_name = self.QUADRANT_LABELS.get(max_quad, max_quad)
                suggestions.append(
                    CameraSuggestion(
                        priority="ACTION_RECOMMENDED",
                        suggestion_type="CONCENTRATION_ZOOM",
                        target_area=quad_name,
                        recommendation=(
                            f"Consider adjusting sensor angle towards {quad_name} and applying optical zoom "
                            f"to inspect crowd formation and potential bottlenecks."
                        ),
                        rationale=(
                            f"{max_pct:.1f}% of all detected individuals are concentrated in {quad_name}, "
                            f"indicating a local density peak."
                        ),
                    )
                )

        # 2. Crowd Flow Tracking Analysis
        if movement_metrics and movement_metrics.moving_count >= 4:
            dom_dir = movement_metrics.dominant_direction
            if dom_dir != "Stationary":
                pct = movement_metrics.direction_percentages.get(dom_dir, 0.0)
                if pct >= 35.0:
                    suggestions.append(
                        CameraSuggestion(
                            priority="ATTENTION",
                            suggestion_type="FLOW_TRACKING",
                            target_area=f"Corridor heading {dom_dir}",
                            recommendation=(
                                f"Recommend panning the aerial view slightly toward {dom_dir} to anticipate "
                                f"downstream egress, choke points, or pathway blockages."
                            ),
                            rationale=(
                                f"Dominant crowd vector is moving {dom_dir} ({pct:.1f}% of active movement), "
                                f"at an average image speed of {movement_metrics.avg_speed_px_per_sec:.1f} px/sec."
                            ),
                        )
                    )

        # 3. Anomaly and Event-driven Inspection
        if recent_events:
            critical_events = [e for e in recent_events if e.severity in ["WARNING", "ALERT"]]
            if critical_events:
                latest = critical_events[-1]
                suggestions.append(
                    CameraSuggestion(
                        priority="ACTION_RECOMMENDED",
                        suggestion_type="SURGE_INSPECTION",
                        target_area="Event Location",
                        recommendation=(
                            f"Operator attention required: Verify {latest.title.lower()} via manual inspection."
                        ),
                        rationale=latest.description,
                    )
                )

        # 4. Default Wide Surveillance when nominal
        if not suggestions:
            suggestions.append(
                CameraSuggestion(
                    priority="INFO",
                    suggestion_type="WIDE_SURVEILLANCE",
                    target_area="Full Frame",
                    recommendation=(
                        "Maintain stable wide-angle bird's-eye perspective for comprehensive situational awareness."
                    ),
                    rationale=(
                        f"Crowd metrics are nominal (Count: {current_count}). Density is evenly distributed "
                        "with no rapid surges or anomalies detected."
                    ),
                )
            )

        return suggestions
