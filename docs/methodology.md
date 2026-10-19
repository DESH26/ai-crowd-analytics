# Technical Report: AI Crowd Analytics on Aerial Video
**Author:** Senior Computer Vision & Full-Stack AI Engineer  
**Client / Evaluator:** Susan Future Technologies (SFT)  
**Date:** October 2026  
**Assignment:** One-Week Technical Internship Assignment  

---

## 1. Executive Summary & Problem Statement

Aerial crowd monitoring using Unmanned Aerial Vehicles (UAVs) and high-altitude fixed cameras presents unique computer vision challenges:
1. **Severe Perspective & Scale Variance:** Unlike conventional CCTV cameras stationed at eye level, aerial viewpoints render pedestrians as small, top-down objects (often $15 \times 15$ to $40 \times 40$ pixels) with minimal biometric texture.
2. **Dense Occlusion & Clustering:** In festivals, public squares, or mass gatherings, individuals physically overlap and cluster, confounding standard non-maximum suppression (NMS) algorithms.
3. **Camera Ego-Motion:** UAV platforms experience gimbal vibrations, drift, and panning, making real-world metric velocity calculation impossible without sensor telemetry or planar homography.

The objective of this project is to construct a **complete, runnable, modular, end-to-end AI Crowd Analytics application** that converts raw aerial video into quantitative, explainable intelligence: accurate people counting, spatial density heatmaps, flow/movement vector analysis, anomaly event detection, virtual line gate flow, and contextual camera operation guidance.

---

## 2. System Architecture

The application adopts a decoupled, layered micro-pipeline architecture:

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
      │                   Output Generators                    │
      │  - Annotated MP4 Video (Bounding Boxes, Trails, HUD)   │
      │  - Per-Frame CSV & Complete Structured JSON            │
      │  - Standalone Density PNG & Markdown Operational Brief │
      │  - Interactive Streamlit Dashboard                     │
      └────────────────────────────────────────────────────────┘
```

---

## 3. Model and Algorithm Choices

### 3.1 Detector: Ultralytics YOLOv8 Nano (`yolov8n.pt`)
- **Rationale:** Aerial crowd monitoring applications on consumer or field laptops require high throughput (15–30+ FPS on CPU/edge GPUs) without exhausting system RAM. YOLOv8n features a lightweight CSPDarknet backbone with Path Aggregation Network (PANet) necks, delivering balanced recall on person detection with minimal FLOPs (~8.7 GFLOPs).
- **Configuration:** Restricts inference strictly to `person` (`class_id = 0`), with configurable confidence thresholds ($0.05$ to $0.90$) and NMS IoU thresholds ($0.10$ to $0.80$).
- **Aerial Adaptation Note:** While baseline COCO weights provide robust generalization for clear pedestrians, high-altitude surveillance benefits from fine-tuning on aerial-specific benchmarks such as VisDrone2019-DET or TinyPerson. The architecture exposes a clean plug-and-play parameter for custom `.pt` weights.

### 3.2 Tracking: Multi-Object Association (ByteTrack & IoU-Centroid)
- **Rationale:** ByteTrack preserves low-confidence detections rather than immediately discarding them, retaining pedestrians through temporary occlusions or aerial contrast fluctuations.
- **Trajectory Retention:** Maintains a sliding historical deque of recent spatial centroids $(cx, cy, t)$, supporting continuous trail visualization and velocity vector calculation.

### 3.3 Density Heatmap: Ground-Footpoint Gaussian Accumulation
- **Rationale:** Traditional heatmaps often mark bounding box centers. In aerial imagery with oblique angles, marking the **footpoint** $(cx, y_2)$ anchors the observation to the actual ground plane, preventing overhead perspective bias.
- **Mathematical Formulation:**
  $$\mathcal{D}(x, y) = \sum_{t=1}^{T} \sum_{i=1}^{N_t} \delta(x - x_{i, t}, y - y_{i, t})$$
  $$\mathcal{H} = \mathcal{D} * \mathcal{G}_{\sigma}, \quad \text{where } \mathcal{G}_{\sigma}(u, v) = \frac{1}{2\pi \sigma^2} e^{-\frac{u^2 + v^2}{2\sigma^2}}$$
  The smoothed matrix is normalized to $[0, 255]$ and mapped through `cv2.COLORMAP_JET`.

### 3.4 Movement Dynamics & Velocity Computation
- **Image-Space Formulation:**
  $$v_x = \frac{\Delta x}{\Delta t}, \quad v_y = \frac{\Delta y}{\Delta t}, \quad S_{\text{px/s}} = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t}$$
  Heading angle $\theta = \operatorname{atan2}(\Delta y, \Delta x)$ is classified into 8 cardinal/intercardinal compass sectors (E, SE, S, SW, W, NW, N, NE) taking into account downward-pointing image $y$-axes.
- **Physical Accuracy Disclaimer:** The system explicitly classifies velocities in **image pixels per second**, never fabricating metric speeds (m/s) without known ground-plane homography.

### 3.5 Event Detection & Camera Operation Advisory
- **Explainable Rules:** Avoids opaque black-box "anomaly AI" models. Employs rolling statistical baselines (mean $\mu$, std $\sigma$ over sliding 15-frame windows) to detect:
  1. *Sudden Crowd Surge:* Count jumps $\ge 40\%$ over baseline.
  2. *High-Density Congestion:* Count exceeds operator threshold.
  3. *Unusual Velocity Spikes:* Mean crowd speed $\ge 2.0\times$ baseline.
- **Advisory Guidance:** Automatically analyzes density quadrants and dominant vector corridors to generate contextual operator recommendations (e.g., suggesting a 2x optical zoom on the North-East quadrant or tracking a South-East egress bottleneck). All suggestions are labeled as strictly advisory.

---

## 4. Manual Count Validation Methodology

To ensure rigorous quantitative accuracy assessment without fabricating artificial benchmark numbers, the system includes an integrated validation harness:

### 4.1 Evaluation Protocol
1. **Sampling:** A stratified sample of frames spanning initial, midpoint, and peak crowd density intervals is extracted.
2. **Ground-Truth Annotation:** Human annotators manually count visible pedestrians within the frame boundaries.
3. **Metric Calculations:**
   $$\text{MAE} = \frac{1}{N} \sum_{k=1}^{N} \left| y_{\text{pred}}^{(k)} - y_{\text{true}}^{(k)} \right|$$
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^{N} \left( y_{\text{pred}}^{(k)} - y_{\text{true}}^{(k)} \right)^2}$$
   $$\text{Mean Count Bias} = \frac{1}{N} \sum_{k=1}^{N} \left( y_{\text{pred}}^{(k)} - y_{\text{true}}^{(k)} \right)$$

### 4.2 Limitations of Aerial Ground-Truth Validation
- In dense overhead clusters, human annotators exhibit inter-annotator variance of $\pm 8\text{--}12\%$.
- Trees, street furniture, and shadow contours introduce visual ambiguity.
- A slight negative bias (undercounting) is standard when using generic ground-level models on small aerial silhouettes.

---

## 5. Known Limitations & Responsible AI Guidelines

1. **No Facial Recognition or Biometrics:** The pipeline tracks anonymous spatial centroids. No biometric features, facial embeddings, or demographic classifications are extracted.
2. **Advisory Event Flagging:** Crowd dynamic anomalies are flagged solely for human dispatcher or operator review; they are never treated as automated emergency alarms.
3. **Ego-Motion Distortion:** On non-stationary drone platforms, high vehicle movement can shift image coordinates.

---

## 6. Future Work & Engineering Enhancements

With additional development time, the following enhancements are recommended:
1. **Model Fine-Tuning on VisDrone / UAVDT:** Fine-tuning YOLOv8 on VisDrone image slices will improve recall on ultra-small pedestrian heads ($< 12\text{ px}$).
2. **Optical Flow Ego-Motion Compensation:** Integrating sparse Lucas-Kanade optical flow on background keypoints to subtract drone flight displacement from pedestrian tracks.
3. **3D Ground Plane Calibration:** Introducing an interactive 4-point homography tool to enable true metric speed ($m/s$) and real-world area density ($\text{persons}/m^2$).
