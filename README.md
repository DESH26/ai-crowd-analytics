# AI Crowd Analytics on Aerial Video

![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
![YOLOv8](https://img.shields.io/badge/detector-Ultralytics%20YOLOv8-green.svg)
![Tracking](https://img.shields.io/badge/tracking-ByteTrack%20%2F%20IoU-orange.svg)
![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)

A complete, production-grade AI Crowd Analytics application designed to analyze aerial and drone video streams, extracting quantitative crowd metrics:
- Real person detection and automated counting
- Temporal spatial density heatmaps (Gaussian footpoint smoothing)
- Movement vector dynamics and cardinal heading distributions
- Explainable crowd anomaly alerts (rapid surges, high-density bottlenecks, speed spikes)
- Virtual line crossing (bidirectional entry and exit gate counting)
- Contextual camera operation recommendations (advisory guidance for UAV / PTZ operators)
- Ground-truth accuracy validation harness (MAE, RMSE, Count Bias)
- Structured per-frame CSV and complete JSON data exports
- Playable annotated video generation with HUD overlay

Developed for **Susan Future Technologies (SFT)** as a technical internship assignment.

---

## 1. System Architecture

```
                  ┌────────────────────────────────────────┐
                  │          Raw Aerial Video File         │
                  │         (.mp4 / .avi / .mov)           │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Video Ingestion (I/O)  │
                        │  Metadata / Validation /  │
                        │    Frame Rate Sampling    │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   YOLO Person Detector    │
                        │  (COCO Class 0 / Aerial)  │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
      ┌───────────────────────────┐       ┌───────────────────────────┐
      │   Multi-Object Tracker    │       │  Spatial Density Heatmap  │
      │    (ByteTrack / IoU)      │       │ (Gaussian Accumulation)   │
      └─────────────┬─────────────┘       └─────────────┬─────────────┘
                    │                                   │
        ┌───────────┴───────────┐                       │
        ▼                       ▼                       ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   Movement   │        │ Line Crossing│        │   Density    │
│   Dynamics   │        │  (Gate Flow) │        │  Quadrants   │
└───────┬──────┘        └───────┬──────┘        └───────┬──────┘
        │                       │                       │
        └───────────┬───────────┴───────────────────────┘
                    │
                    ▼
      ┌───────────────────────────┐
      │  Explainable Rule Engine  │
      │  (Surge / Speed Spikes /  │
      │   Camera Suggestions)     │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌────────────────────────────────────────────────────────┐
      │                   Output Deliverables                  │
      │  - Annotated MP4 Video (Bounding Boxes, Trails, HUD)   │
      │  - Per-Frame CSV & Complete Structured JSON            │
      │  - Standalone Density PNG & Operational Brief (MD)     │
      │  - Interactive Streamlit Web Dashboard                 │
      └────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
ai-crowd-analytics/
├── app.py                     # Interactive Streamlit Web Application
├── config.py                  # Global configurations and threshold settings
├── requirements.txt           # Core Python dependencies
├── README.md                  # Comprehensive documentation and reproduction guide
├── .gitignore                 # Standard git ignore rules (weights, caches, outputs)
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── video_io.py            # Stream metadata, reader with sampling, writer with fallback
│   ├── detector.py            # Ultralytics YOLO person detector with aerial notes
│   ├── tracker.py             # ByteTrack / IoU tracking with history trails
│   ├── crowd_counter.py       # Strict per-frame counting and aggregate statistics
│   ├── heatmap.py             # Spatial density accumulation, Gaussian blur, overlay
│   ├── movement.py            # Image-space speed, heading angles, cardinal rose
│   ├── anomaly_detection.py   # Rolling statistical crowd surge and velocity alerts
│   ├── line_crossing.py       # Virtual line gate entry/exit bidirectional counting
│   ├── camera_suggestions.py  # Explainable advisory guidance for drone operators
│   ├── validation.py          # MAE, RMSE, Count Bias ground-truth validation engine
│   ├── visualization.py       # Overlays, bounding boxes, trajectory trails, HUD
│   ├── export.py              # Per-frame CSV, JSON, event logs, Markdown reports
│   └── analytics.py           # Master pipeline orchestrator
├── tests/                     # Automated Pytest suite (23 unit tests)
│   ├── test_video_io.py
│   ├── test_detector.py
│   ├── test_heatmap.py
│   ├── test_analytics.py
│   └── test_export.py
├── docs/
│   └── methodology.md         # 1-to-2 page technical methodology & architecture report
├── sample_data/               # Sample aerial videos and sequence descriptions
│   ├── README.md
│   └── aerial_sample_preview.mp4
├── outputs/                   # Generated deliverables (videos, CSV, JSON, heatmaps)
│   └── .gitkeep
└── models/                    # Model weights directory
    ├── README.md
    └── yolov8n.pt             # Standard lightweight Nano model
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11 or 3.12 (64-bit recommended)
- `pip` package manager
- Optional: CUDA-compatible NVIDIA GPU (CPU execution is fully supported)

### Step 1: Clone or Navigate to Project
```bash
# Windows PowerShell
cd C:\Users\deshv\.gemini\antigravity\scratch\ai-crowd-analytics

# Linux / macOS
cd ~/ai-crowd-analytics
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Standard installation
pip install -r requirements.txt
```
> **Windows Installation Tip:** If installing into user site-packages without elevated permissions, install packages with `pip install -r requirements.txt`. To avoid Windows `MAX_PATH` length constraints with certain nested licenses, the repository avoids unneeded heavyweight torch re-installations.

---

## 4. Running the Application

### Launching the Interactive Streamlit Dashboard
```bash
# Windows PowerShell
python -m streamlit run app.py

# Linux / macOS
streamlit run app.py
```
Once launched, the dashboard will open in your default browser at `http://localhost:8501`.

### Step-by-Step Workflow in Dashboard
1. **Overview & Ingestion:**
   - Choose **"Select Sample / Demo Video"** to test with the bundled `aerial_sample_preview.mp4` immediately, or select **"Upload File"** for your own drone video (`.mp4`, `.avi`, `.mov`).
   - Inspect video resolution, frame rate, total frames, and duration.
   - Adjust detection confidence (default `0.25`), frame sampling interval (e.g. `1` for full analysis, `2` or `3` for faster laptop inference), and virtual line gate coordinates.
   - Click **"🚀 Start Video Analysis"**.
2. **Crowd Analytics:**
   - View key metrics: Peak Observed Count, Mean Observed Count, Minimum Observed Count.
   - Examine the interactive **Crowd Count Over Time** chart.
3. **Density Heatmap:**
   - Inspect the cumulative spatial concentration heatmap and spatial quadrant distribution (NW, NE, SW, SE).
4. **Movement Dynamics:**
   - Observe dominant flow direction, average pixel speed, and the 8-cardinal compass distribution.
5. **Events & Camera Advisory:**
   - Inspect flagged crowd surge or velocity spike events (all marked for human operator verification).
   - Review bidirectional gate entries/exits.
   - Read actionable camera operator suggestions (e.g., zoom recommendations for dense quadrants).
6. **Accuracy Validation:**
   - Select specific video frames, inspect model predictions, enter manual ground-truth counts, and view computed MAE, RMSE, and count bias.
7. **Downloads & Reports:**
   - Download the generated annotated video (`.mp4`), per-frame CSV, structured JSON, heatmap image (`.png`), and the operational markdown brief.

---

## 5. Running Automated Tests

Run the complete 23-test test suite:
```bash
# Windows PowerShell
python -m pytest tests/ -v

# Linux / macOS
pytest tests/ -v
```
All tests verify:
- Stream metadata extraction, validation, frame sampling, and video writer reopen checks.
- YOLO detector initialization, empty frame resilience, and aerial notes.
- Density heatmap accumulation, normalization, Gaussian blur, and quadrant metrics.
- Track history, image-space velocity calculation, and cardinal direction mapping.
- Line crossing intersection geometry and debounce cooldowns.
- Anomaly detection rule baselines and alert cooldowns.
- Accuracy validation metric calculations (MAE, RMSE, Bias).
- Schema conformity for per-frame CSV and JSON exports.

---

## 6. Output Schemas

### Frame-Level CSV Schema (`outputs/*_frame_analytics.csv`)
| Column Name | Type | Description |
|---|---|---|
| `frame_number` | `int` | Original video frame index (0-indexed) |
| `timestamp_seconds` | `float` | Elapsed video time in seconds |
| `observed_person_count` | `int` | Count of detected persons retained in frame |
| `detection_count` | `int` | Raw person detections matching confidence threshold |
| `tracked_person_count` | `int` | Number of active multi-object tracks in frame |
| `processing_time_seconds` | `float` | Inference and processing latency for this frame |
| `is_sampled_frame` | `bool` | True if frame was processed under sampling interval > 1 |
| `dominant_direction` | `string` | Primary compass direction (e.g. `E`, `SE`, `Stationary`) |
| `avg_speed_px_per_sec` | `float` | Mean image-space displacement speed in pixels/sec |
| `event_flags` | `json_string` | List of anomaly alerts triggered in this frame |
| `movement_summary` | `json_string` | Direction counts, moving count, and speed metrics |
| `detection_details` | `json_string` | Detailed list of bounding boxes and confidences |
| `tracking_details` | `json_string` | Active track IDs, velocity vectors, and headings |

### Structured JSON Schema (`outputs/*_analytics.json`)
```json
{
  "project": "AI Crowd Analytics on Aerial Video",
  "organization": "Susan Future Technologies (SFT)",
  "schema_version": "1.0",
  "video_metadata": {
    "width": 768,
    "height": 432,
    "fps": 12.0,
    "total_frames": 647,
    "duration_seconds": 53.92
  },
  "summary_statistics": {
    "peak_observed_count": 2,
    "mean_observed_count": 0.34,
    "min_observed_count": 0,
    "sampling_interval": 3
  },
  "cumulative_movement": {
    "dominant_movement_direction": "Stationary",
    "overall_avg_speed_px_per_sec": 0.0
  },
  "line_crossing_summary": {
    "total_in": 0,
    "total_out": 0,
    "net_flow": 0
  },
  "camera_operator_suggestions": [
    {
      "priority": "INFO",
      "target_area": "Full Frame",
      "recommendation": "Maintain stable wide-angle bird's-eye perspective..."
    }
  ],
  "frame_level_records": [...]
}
```

---

## 7. Accuracy Validation Reproduction Procedure

To reproduce or conduct the manual ground-truth accuracy validation:
1. Open the **"🎯 Accuracy Validation"** tab in the Streamlit dashboard after running analysis on a video.
2. Select a frame (e.g., Frame `0`, `50`, `100`, `200`).
3. View the frame image and observe the model prediction.
4. Manually count the number of visible individuals in the frame.
5. Enter the count into the **"Manual Ground-Truth Count"** field and click **"Record Validation Sample"**.
6. Repeat for 5–10 representative frames.
7. The table and summary cards automatically update with:
   - **MAE (Mean Absolute Error):** $\frac{1}{N}\sum |y_{\text{pred}} - y_{\text{true}}|$
   - **Count Bias:** $\frac{1}{N}\sum (y_{\text{pred}} - y_{\text{true}})$ (indicates over/undercounting)
   - **RMSE:** $\sqrt{\frac{1}{N}\sum (y_{\text{pred}} - y_{\text{true}})^2}$
8. Download the final validation table as `validation_results.csv`.

---

## 8. Responsible AI, Privacy & Limitations

1. **Privacy Protection:** Processing is performed locally. The system does not incorporate facial recognition, demographic classification, or individual re-identification across cameras.
2. **Advisory Anomaly Design:** All events (surges, speed spikes) are explicitly tagged as potential anomalies for human dispatcher review, never automated emergency alarms.
3. **Image-Space Speed:** Without a calibrated camera and ground-plane homography transformation matrix, speed is reported exclusively in pixels per second.
4. **Aerial Detector Note:** Standard YOLOv8 weights are trained on COCO (eye-level perspective). For production drone missions at altitudes $> 50$ meters, fine-tuning on aerial-specific datasets such as VisDrone2019 or UAVDT is strongly recommended.
