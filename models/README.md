# Models Directory

This directory stores deep learning object detection weights for the crowd analytics pipeline.

## Default Model: YOLOv8n (`yolov8n.pt`)
- **Architecture:** Ultralytics YOLOv8 Nano
- **Parameters:** ~3.2M parameters
- **Input Size:** 640 x 640 (configurable)
- **Target Class:** Class 0 (`person`)
- **Primary Execution Mode:** Automatic detection (CUDA if available, CPU fallback)

## Aerial Fine-Tuning & Custom Weights
Standard COCO models are trained predominantly on eye-level perspectives. For specialized high-altitude drone operations, you can drop fine-tuned weights (e.g. trained on VisDrone-DET, UAVDT, or TinyPerson) into this folder and select them via the dashboard or configuration file:
```python
pipeline = VideoAnalyticsPipeline(model_name_or_path="models/visdrone_yolov8s.pt")
```
