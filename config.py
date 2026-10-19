"""
Configuration module for AI Crowd Analytics on Aerial Video.
Provides default settings, path configurations, and configurable thresholds.
"""

from pathlib import Path
import cv2

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory paths
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Detection Settings
DEFAULT_MODEL_NAME = "yolov8n.pt"  # Lightweight nano model
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0  # COCO class 0 is person
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# Processing Settings
DEFAULT_FRAME_SAMPLING = 1  # 1 = every frame, 2 = every 2nd frame, etc.
DEFAULT_MAX_DIMENSION = 1280  # Limit max frame width/height to avoid OOM

# Tracker Settings
TRACKER_TYPE = "bytetrack"
DEFAULT_TRACK_BUFFER = 30
DEFAULT_MAX_TRAIL_LENGTH = 20

# Heatmap Settings
HEATMAP_SIGMA = 15
HEATMAP_ALPHA = 0.45
HEATMAP_COLORMAP = cv2.COLORMAP_JET
HEATMAP_POINT_MODE = "footpoint"  # 'footpoint' or 'center'

# Movement Analysis Settings
MIN_TRACK_LENGTH_FOR_VELOCITY = 3
CARDINAL_DIRECTIONS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

# Anomaly Detection Settings
DEFAULT_SURGE_PERCENT = 40.0  # % increase over baseline to trigger surge alert
DEFAULT_ROLLING_WINDOW = 15   # Window in processed frames
DEFAULT_HIGH_DENSITY_THRESHOLD = 30  # Persons
DEFAULT_SPEED_SPIKE_RATIO = 2.0  # Speed relative to rolling average

# Line Crossing Default Coordinates (normalized 0.0 to 1.0)
DEFAULT_LINE_COORDS = ((0.1, 0.5), (0.9, 0.5))

# Camera Operation Suggestion Thresholds
DENSITY_HOTSPOT_RATIO = 0.40  # If 40%+ of crowd is in one quadrant
FAST_MOVEMENT_PIXELS_PER_SEC = 80.0
