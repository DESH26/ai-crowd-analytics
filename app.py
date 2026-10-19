"""
Streamlit Web Application for AI Crowd Analytics on Aerial Video.
Interactive dashboard for video analysis, person detection, tracking,
density heatmaps, movement dynamics, anomaly alerts, camera operation
suggestions, validation, and data export.
"""

import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLE_DATA_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_FRAME_SAMPLING,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from src.video_io import get_video_metadata, validate_video
from src.analytics import VideoAnalyticsPipeline
from src.validation import AccuracyValidator

# Streamlit Page Config
st.set_page_config(
    page_title="AI Crowd Analytics on Aerial Video",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-banner {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "active_video_path" not in st.session_state:
    st.session_state.active_video_path = None
if "validator" not in st.session_state:
    st.session_state.validator = AccuracyValidator()

# --- SIDEBAR: Configuration & Controls ---
st.sidebar.image("https://img.icons8.com/color/96/drone.png", width=64)
st.sidebar.title("Configuration")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Inference Model",
    options=["yolov8n.pt", "yolov8s.pt", "Custom Model Path"],
    index=0,
    help="Default is YOLOv8n (nano). Configurable for aerial-specific fine-tuned models.",
)
custom_model_path = ""
if model_choice == "Custom Model Path":
    custom_model_path = st.sidebar.text_input("Path to custom .pt weights", value="")
selected_model = custom_model_path if model_choice == "Custom Model Path" else model_choice

# Thresholds
conf_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=0.90,
    value=DEFAULT_CONF_THRESHOLD,
    step=0.05,
    help="Higher values reduce false positives; lower values help pick up small/distant aerial targets.",
)
iou_threshold = st.sidebar.slider(
    "NMS IOU Threshold",
    min_value=0.10,
    max_value=0.80,
    value=DEFAULT_IOU_THRESHOLD,
    step=0.05,
)

# Performance & Sampling
st.sidebar.subheader("Performance Settings")
sample_interval = st.sidebar.select_slider(
    "Frame Sampling Interval",
    options=[1, 2, 3, 5, 10, 15, 30],
    value=DEFAULT_FRAME_SAMPLING,
    help="1 = process every frame. Higher intervals speed up inference significantly on laptops.",
)
max_res = st.sidebar.selectbox(
    "Inference Max Resolution",
    options=[640, 960, 1280, "Original (Full)"],
    index=2,
    help="Resizes frame for detection if larger than this dimension to prevent GPU/CPU memory bottlenecks.",
)
max_dim = None if max_res == "Original (Full)" else int(max_res)

# Feature Toggles
st.sidebar.subheader("Feature Toggles")
enable_tracking = st.sidebar.checkbox("Multi-Object Tracking", value=True)
enable_heatmap = st.sidebar.checkbox("Crowd Density Heatmap", value=True)
enable_movement = st.sidebar.checkbox("Movement Vector Analysis", value=True)
enable_anomalies = st.sidebar.checkbox("Anomaly & Surge Alerts", value=True)
enable_line_crossing = st.sidebar.checkbox("Virtual Line Gate (Entry/Exit)", value=True)

# Main Title Header
st.markdown('<div class="main-title">AI Crowd Analytics on Aerial Video</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated aerial video crowd counting, spatial density mapping, movement tracking, '
    'and explainable camera operation suggestions for Susan Future Technologies.</div>',
    unsafe_allow_html=True,
)

# Tabs Navigation
tab_overview, tab_analytics, tab_density, tab_movement, tab_events, tab_validation, tab_downloads = st.tabs([
    "📹 Overview & Ingestion",
    "📊 Crowd Analytics",
    "🔥 Density Heatmap",
    "🧭 Movement Dynamics",
    "🚨 Events & Camera Advice",
    "🎯 Accuracy Validation",
    "💾 Downloads & Reports",
])

# ==========================================
# TAB 1: OVERVIEW & INGESTION
# ==========================================
with tab_overview:
    col_input, col_meta = st.columns([1.2, 1.0])

    with col_input:
        st.subheader("1. Video Source")
        source_mode = st.radio("Choose Video Source:", ["Upload File", "Select Sample / Demo Video"], horizontal=True)

        selected_video_path = None

        if source_mode == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Aerial Footage (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov", "mkv"],
                help="Aerial or overhead footage recommended.",
            )
            if uploaded_file is not None:
                # Save to temporary scratch location
                temp_dir = Path(tempfile.gettempdir()) / "sft_aerial_crowd"
                temp_dir.mkdir(parents=True, exist_ok=True)
                save_path = temp_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_video_path = save_path
        else:
            # Check for existing sample videos in sample_data
            sample_videos = list(SAMPLE_DATA_DIR.glob("*.mp4")) + list(SAMPLE_DATA_DIR.glob("*.avi"))
            if sample_videos:
                sample_choice = st.selectbox(
                    "Available Sample Aerial Videos:",
                    options=[v.name for v in sample_videos],
                )
                selected_video_path = SAMPLE_DATA_DIR / sample_choice
            else:
                st.info("No sample videos found in sample_data/. Please upload a video or generate a synthetic test clip.")

        # Virtual Line Setup
        if enable_line_crossing:
            with st.expander("Virtual Line Gate Setup (Optional)", expanded=False):
                st.write("Configure normalized coordinates for bidirectional entry/exit counting line:")
                lc_col1, lc_col2 = st.columns(2)
                with lc_col1:
                    line_x1 = st.slider("Line Start X (%)", 0.0, 1.0, 0.1, 0.05)
                    line_y1 = st.slider("Line Start Y (%)", 0.0, 1.0, 0.5, 0.05)
                with lc_col2:
                    line_x2 = st.slider("Line End X (%)", 0.0, 1.0, 0.9, 0.05)
                    line_y2 = st.slider("Line End Y (%)", 0.0, 1.0, 0.5, 0.05)

    with col_meta:
        st.subheader("2. Stream Inspection & Metadata")
        if selected_video_path and Path(selected_video_path).exists():
            is_valid, msg, meta = validate_video(selected_video_path)
            if is_valid:
                st.success(f"Stream Status: Valid ({meta['width']}x{meta['height']})")

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Duration", f"{meta['duration_seconds']} s")
                    st.metric("Total Frames", f"{meta['total_frames']}")
                with m_col2:
                    st.metric("Frame Rate", f"{meta['fps']:.1f} FPS")
                    st.metric("File Size", f"{meta['file_size_mb']} MB")

                st.session_state.active_video_path = selected_video_path

                # Show preview of first frame
                cap = cv2.VideoCapture(str(selected_video_path))
                ret, frame0 = cap.read()
                cap.release()
                if ret and frame0 is not None:
                    # Draw virtual line preview
                    if enable_line_crossing:
                        h, w = frame0.shape[:2]
                        p1 = (int(line_x1 * w), int(line_y1 * h))
                        p2 = (int(line_x2 * w), int(line_y2 * h))
                        cv2.line(frame0, p1, p2, (0, 140, 255), 2)
                        cv2.putText(frame0, "Virtual Line Gate", (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                    frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Initial Frame Preview & Gate Coordinates", use_container_width=True)
            else:
                st.error(f"Validation Error: {msg}")
        else:
            st.info("Upload or select a video to inspect stream parameters.")

    st.markdown("---")
    # Processing Action Area
    col_proc, col_prog = st.columns([1, 2])
    with col_proc:
        start_btn = st.button("🚀 Start Video Analysis", type="primary", use_container_width=True)

    with col_prog:
        prog_bar = st.empty()
        status_text = st.empty()

    if start_btn:
        if not st.session_state.active_video_path or not Path(st.session_state.active_video_path).exists():
            st.error("Please provide a valid video before starting processing.")
        else:
            vpath = Path(st.session_state.active_video_path)
            meta = get_video_metadata(vpath)
            orig_w, orig_h = meta["width"], meta["height"]

            virt_line = None
            if enable_line_crossing:
                p1 = (int(line_x1 * orig_w), int(line_y1 * orig_h))
                p2 = (int(line_x2 * orig_w), int(line_y2 * orig_h))
                virt_line = (p1, p2)

            pipeline = VideoAnalyticsPipeline(
                model_name_or_path=selected_model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                sample_interval=sample_interval,
                max_dimension=max_dim,
                enable_tracking=enable_tracking,
                enable_heatmap=enable_heatmap,
                enable_movement=enable_movement,
                enable_anomalies=enable_anomalies,
                enable_line_crossing=enable_line_crossing,
                virtual_line_coords=virt_line,
            )

            p_bar = prog_bar.progress(0.0)

            def update_progress(curr_f, total_f, current_fps, msg):
                if total_f > 0:
                    pct = min(1.0, float(curr_f) / float(total_f))
                    p_bar.progress(pct)
                status_text.text(f"{msg} | Rate: {current_fps:.1f} FPS")

            with st.spinner("Processing video frames and computing crowd dynamics..."):
                try:
                    res = pipeline.process_video(
                        input_video_path=vpath,
                        output_dir=OUTPUTS_DIR,
                        generate_annotated_video=True,
                        progress_callback=update_progress,
                    )
                    st.session_state.results = res
                    p_bar.progress(1.0)
                    status_text.success(f"Processing Complete! ({res['total_frames_processed']} frames analyzed at {res['average_processing_fps']} FPS)")
                    st.toast("Video processing completed successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")

# Ensure results are available for subsequent tabs
res = st.session_state.results

# ==========================================
# TAB 2: CROWD ANALYTICS
# ==========================================
with tab_analytics:
    if res is None:
        st.info("⚠️ Please upload and process a video in the Overview tab to view analytics.")
    else:
        st.subheader("Estimated Crowd Count Dynamics")
        stats = res["summary_statistics"]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Peak Observed", f"{stats['peak_observed_count']} persons", f"at {stats.get('peak_timestamp_seconds', 0.0)}s")
        k2.metric("Mean Observed", f"{stats['mean_observed_count']:.1f} persons")
        k3.metric("Minimum Observed", f"{stats['min_observed_count']} persons")
        k4.metric("Frames Analyzed", f"{stats['total_processed_frames']}")
        k5.metric("Sampling Interval", f"{'1:1 (Full)' if stats['sampling_interval'] == 1 else f'1 in {stats['sampling_interval']}'}")

        # Note on Aerial Estimation
        st.info(
            "💡 **Interpretation Note:** In aerial or UAV imagery, individuals may appear small or partially occluded. "
            "Counts are automated estimates based on detected bounding boxes. Recorded count strictly matches person detections retained for that frame."
        )

        # Plot Crowd Count Over Time
        csv_path = res["frame_csv_path"]
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)

            st.markdown("#### Crowd Count Over Time")
            fig, ax = plt.subplots(figsize=(10, 3.8))
            ax.plot(df["timestamp_seconds"], df["observed_person_count"], color="#2563EB", linewidth=2, label="Observed Person Count")
            if "tracked_person_count" in df.columns:
                ax.plot(df["timestamp_seconds"], df["tracked_person_count"], color="#10B981", linestyle="--", linewidth=1.5, label="Tracked Individuals")

            # Peak Marker
            peak_val = stats["peak_observed_count"]
            peak_time = stats["peak_timestamp_seconds"]
            if peak_time is not None:
                ax.scatter([peak_time], [peak_val], color="#DC2626", s=60, zorder=5, label=f"Peak Count ({peak_val})")

            ax.set_xlabel("Video Timestamp (seconds)", fontsize=10)
            ax.set_ylabel("Observed Person Count", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            fig.tight_layout()
            st.pyplot(fig)

            # Per-frame tabular inspection
            with st.expander("Inspect Frame-by-Frame Data Table"):
                display_cols = ["frame_number", "timestamp_seconds", "observed_person_count", "tracked_person_count", "dominant_direction", "avg_speed_px_per_sec", "processing_time_seconds"]
                st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True)

# ==========================================
# TAB 3: DENSITY HEATMAP
# ==========================================
with tab_density:
    if res is None:
        st.info("⚠️ Please upload and process a video in the Overview tab to view heatmap.")
    else:
        st.subheader("Accumulated Crowd Density Heatmap")
        st.markdown(
            "This visualization shows the accumulated spatial concentration of people over all processed frames. "
            "Gaussian smoothing is applied to detected ground footpoints."
        )

        hm_col1, hm_col2 = st.columns([1.2, 0.8])
        with hm_col1:
            hm_path = res.get("heatmap_image_path")
            if hm_path and Path(hm_path).exists():
                hm_img = cv2.imread(hm_path)
                hm_rgb = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
                st.image(hm_rgb, caption="Cumulative Spatial Density Heatmap (Standalone)", use_container_width=True)
            else:
                st.warning("No heatmap generated or empty detections.")

        with hm_col2:
            st.markdown("#### Spatial Quadrant Distribution")
            q_dist = res.get("quadrant_distribution", {})
            st.write("Percentage of accumulated crowd presence by image quadrant:")

            q_c1, q_c2 = st.columns(2)
            with q_c1:
                st.metric("Top-Left (NW)", f"{q_dist.get('NW', 0.0)}%")
                st.metric("Bottom-Left (SW)", f"{q_dist.get('SW', 0.0)}%")
            with q_c2:
                st.metric("Top-Right (NE)", f"{q_dist.get('NE', 0.0)}%")
                st.metric("Bottom-Right (SE)", f"{q_dist.get('SE', 0.0)}%")

            st.caption(
                "**Interpretation:** Represents historical accumulation of observed detections across the video sequence, "
                "not the exact number of individuals physically present at any single instant."
            )

# ==========================================
# TAB 4: MOVEMENT DYNAMICS
# ==========================================
with tab_movement:
    if res is None:
        st.info("⚠️ Please upload and process a video in the Overview tab to view movement analysis.")
    else:
        st.subheader("Crowd Flow & Movement Direction Analysis")
        mov = res.get("cumulative_movement", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dominant Heading", mov.get("dominant_movement_direction", "Stationary"))
        m2.metric("Average Speed", f"{mov.get('overall_avg_speed_px_per_sec', 0.0)} px/s")
        m3.metric("95th Percentile Speed", f"{mov.get('speed_95th_percentile_px_per_sec', 0.0)} px/s")
        m4.metric("Total Observations", f"{mov.get('total_movement_observations', 0)}")

        st.warning(
            "⚠️ **Image-Space Speed Disclaimer:** Speed values are expressed strictly in 2D image-space pixels per second. "
            "Without camera intrinsic calibration and a ground-plane homography transformation matrix, physical distance and speed in meters cannot be reliably determined."
        )

        dir_pcts = mov.get("cumulative_direction_percentages", {})
        if dir_pcts:
            st.markdown("#### Directional Heading Distribution")
            # Bar chart of cardinal directions
            cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Stationary"]
            vals = [dir_pcts.get(k, 0.0) for k in cardinals]

            fig, ax = plt.subplots(figsize=(9, 3.5))
            bars = ax.bar(cardinals, vals, color="#3B82F6", edgecolor="#1D4ED8")
            ax.set_ylabel("Percentage of Observations (%)", fontsize=10)
            ax.set_xlabel("Compass Heading", fontsize=10)
            ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 100)
            ax.grid(axis="y", linestyle=":", alpha=0.7)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

            fig.tight_layout()
            st.pyplot(fig)

# ==========================================
# TAB 5: EVENTS & CAMERA ADVICE
# ==========================================
with tab_events:
    if res is None:
        st.info("⚠️ Please upload and process a video in the Overview tab to view events.")
    else:
        st.subheader("Explainable Anomaly Alerts & Camera Advisory")

        col_ev, col_cam = st.columns([1.1, 0.9])

        with col_ev:
            st.markdown("#### Potential Crowd Dynamic Anomalies")
            events_csv = res.get("events_csv_path")
            if events_csv and Path(events_csv).exists():
                ev_df = pd.read_csv(events_csv)
                if not ev_df.empty:
                    st.dataframe(ev_df[["event_type", "severity", "timestamp_seconds", "title", "description"]], use_container_width=True)
                else:
                    st.success("No anomalous surges or velocity spikes were detected in this video.")
            else:
                st.success("Nominal conditions observed across all frames.")

            st.caption(
                "🛡️ **Responsible AI Policy:** Flagged events are statistical anomalies designed for operator verification, "
                "not confirmed emergencies. Automatic emergency escalation is not performed."
            )

            # Virtual Line Crossing Stats
            lc = res.get("line_crossing_summary", {})
            if lc:
                st.markdown("#### Virtual Line Crossing (Gate Flow)")
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Total Entry (IN)", f"{lc.get('total_in', 0)}")
                lc2.metric("Total Exit (OUT)", f"{lc.get('total_out', 0)}")
                lc3.metric("Net Inflow", f"{lc.get('net_flow', 0)}")

        with col_cam:
            st.markdown("#### Camera Operation Recommendations")
            suggs = res.get("camera_suggestions", [])
            for s in suggs:
                priority_color = {
                    "ACTION_RECOMMENDED": "#DC2626",
                    "ATTENTION": "#F59E0B",
                    "INFO": "#2563EB",
                }.get(s.get("priority"), "#64748B")

                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {priority_color}; padding: 10px 14px; background-color: #F8FAFC; margin-bottom: 12px; border-radius: 4px;">
                        <strong style="color: {priority_color};">[{s.get('priority')}] {s.get('target_area')}</strong><br/>
                        <span>{s.get('recommendation')}</span><br/>
                        <small style="color: #64748B;"><em>Rationale:</em> {s.get('rationale')}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "📋 **Advisory Note:** Recommendations are contextual guidance for the drone pilot or PTZ operator. "
                "The system does not execute physical flight or camera actuator commands."
            )

# ==========================================
# TAB 6: ACCURACY VALIDATION
# ==========================================
with tab_validation:
    st.subheader("Manual Ground-Truth Count Validation")
    st.markdown(
        "Validate model predictions against manual human counts on selected frames. "
        "Calculates Mean Absolute Error (MAE), RMSE, and Count Bias."
    )

    if res is None:
        st.info("Please process a video first to populate candidate frames for validation.")
    else:
        csv_path = res["frame_csv_path"]
        df_frames = pd.read_csv(csv_path)

        val_col1, val_col2 = st.columns([1, 1.2])

        with val_col1:
            st.markdown("#### Enter Ground-Truth Count")
            avail_frames = df_frames["frame_number"].tolist()
            selected_f = st.selectbox("Select Frame Number to Validate:", options=avail_frames)

            # Look up model prediction for this frame
            matched_row = df_frames[df_frames["frame_number"] == selected_f].iloc[0]
            pred_count = int(matched_row["observed_person_count"])
            t_sec = float(matched_row["timestamp_seconds"])

            st.write(f"**Timestamp:** {t_sec:.2f}s | **Model Predicted Count:** `{pred_count}`")

            gt_input = st.number_input("Manual Ground-Truth Count (Human Counted):", min_value=0, max_value=500, value=pred_count, step=1)

            if st.button("Record Validation Sample"):
                st.session_state.validator.add_sample(
                    frame_number=selected_f,
                    timestamp_seconds=t_sec,
                    predicted_count=pred_count,
                    ground_truth_count=gt_input,
                )
                st.success(f"Recorded Frame {selected_f}: Predicted={pred_count}, Ground Truth={gt_input}")

        with val_col2:
            st.markdown("#### Validation Metrics")
            metrics = st.session_state.validator.calculate_metrics()

            if metrics["num_samples"] > 0:
                vm1, vm2, vm3 = st.columns(3)
                vm1.metric("Samples", f"{metrics['num_samples']}")
                vm2.metric("MAE", f"{metrics['mae']} persons")
                vm3.metric("Bias", f"{metrics['bias']:+.2f}")

                st.write(f"**Interpretation:** {metrics['bias_interpretation']}")
                st.write(f"**RMSE:** {metrics['rmse']} | **MAPE:** {metrics['mape_percent']}%")

                val_df = st.session_state.validator.to_dataframe()
                st.dataframe(val_df, use_container_width=True)

                val_csv = val_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Validation CSV", data=val_csv, file_name="validation_results.csv", mime="text/csv")
            else:
                st.info("No validation samples recorded yet. Enter counts above to calculate MAE and bias.")

        st.caption(
            "⚠️ **Validation Limitation Note:** Aerial video validation is subject to human counting error on small, "
            "occluded, or shadowed subjects (+/- 10% variance typical in dense clusters)."
        )

# ==========================================
# TAB 7: DOWNLOADS & REPORTS
# ==========================================
with tab_downloads:
    st.subheader("Generated Deliverables & Exports")

    if res is None:
        st.info("⚠️ Process a video to generate download packages.")
    else:
        st.write("Download all structured outputs, annotated video, and reports:")

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.markdown("#### 1. Visual Deliverables")
            # Annotated Video
            vid_path = res.get("annotated_video_path")
            if vid_path and Path(vid_path).exists():
                with open(vid_path, "rb") as f:
                    vid_bytes = f.read()
                st.download_button(
                    "🎬 Download Annotated Output Video (MP4)",
                    data=vid_bytes,
                    file_name=Path(vid_path).name,
                    mime="video/mp4",
                    use_container_width=True,
                )
                st.caption(f"Verification: {res.get('annotated_video_status')}")

            # Standalone Heatmap
            hm_path = res.get("heatmap_image_path")
            if hm_path and Path(hm_path).exists():
                with open(hm_path, "rb") as f:
                    hm_bytes = f.read()
                st.download_button(
                    "🔥 Download Density Heatmap Image (PNG)",
                    data=hm_bytes,
                    file_name=Path(hm_path).name,
                    mime="image/png",
                    use_container_width=True,
                )

        with d_col2:
            st.markdown("#### 2. Structured Data Deliverables")
            # Frame CSV
            f_csv = res.get("frame_csv_path")
            if f_csv and Path(f_csv).exists():
                with open(f_csv, "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    "📊 Download Per-Frame CSV Data",
                    data=csv_bytes,
                    file_name=Path(f_csv).name,
                    mime="text/csv",
                    use_container_width=True,
                )

            # Complete JSON
            c_json = res.get("complete_json_path")
            if c_json and Path(c_json).exists():
                with open(c_json, "rb") as f:
                    json_bytes = f.read()
                st.download_button(
                    "📦 Download Complete Analytics JSON",
                    data=json_bytes,
                    file_name=Path(c_json).name,
                    mime="application/json",
                    use_container_width=True,
                )

            # Operational Report Markdown
            s_rep = res.get("summary_report_path")
            if s_rep and Path(s_rep).exists():
                with open(s_rep, "rb") as f:
                    rep_bytes = f.read()
                st.download_button(
                    "📑 Download Operational Summary Report (MD)",
                    data=rep_bytes,
                    file_name=Path(s_rep).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
