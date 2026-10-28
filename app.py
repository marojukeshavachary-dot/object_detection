"""
app.py
------
Streamlit UI for the Real-Time Object Detection System.

Run with:
    streamlit run app.py

The UI is split into five pages:
    Dashboard        – Project overview and model info
    Webcam Detection – Live detection via the user's webcam (WebRTC)
    Image Detection  – Upload JPG/PNG for detection
    Video Detection  – Upload MP4/AVI for frame-by-frame detection
    About Project    – Architecture, credits, and limitations
"""

import io
import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Local imports ─────────────────────────────────────────────────────────────
from detector import ObjectDetector, AVAILABLE_MODELS

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Real-Time Object Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Global CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stRadio label {
    color: #c9d1d9 !important;
    font-size: 0.95rem;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 50%, #58a6ff 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(31,111,235,0.35);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px;
}
.hero p {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
}

/* ── Metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #388bfd; }
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #58a6ff;
    border-left: 4px solid #1f6feb;
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* ── Status badge ── */
.badge-success { color: #3fb950; font-weight: 600; }
.badge-warning { color: #d29922; font-weight: 600; }
.badge-error   { color: #f85149; font-weight: 600; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 12px;
    background: #0d1117;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Info boxes ── */
.info-box {
    background: rgba(31,111,235,0.1);
    border: 1px solid rgba(56,139,253,0.4);
    border-radius: 10px;
    padding: 14px 18px;
    color: #c9d1d9;
    font-size: 0.9rem;
    margin: 12px 0;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "detector": None,
        "loaded_model_name": None,
        "webcam_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 24px;">
            <div style="font-size:2.8rem;">🎯</div>
            <div style="font-size:1.05rem; font-weight:700; color:#58a6ff;">Object Detection</div>
            <div style="font-size:0.78rem; color:#8b949e;">OpenCV · YOLOv8 · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠  Dashboard",
            "📷  Webcam Detection",
            "🖼️  Image Detection",
            "🎬  Video Detection",
            "ℹ️  About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Global settings ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Detection Settings")

    selected_model_label = st.selectbox(
        "Model",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Lighter models are faster but slightly less accurate.",
    )
    selected_model_file = AVAILABLE_MODELS[selected_model_label]

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Detections below this score are discarded.",
    )

    # ── Load / reload model ──────────────────────────────────────────────
    if st.session_state.loaded_model_name != selected_model_file:
        with st.spinner(f"Loading {selected_model_file} …"):
            try:
                st.session_state.detector = ObjectDetector(selected_model_file)
                st.session_state.loaded_model_name = selected_model_file
                st.success("Model ready ✓")
            except Exception as exc:
                st.error(f"Model load failed: {exc}")
                st.session_state.detector = None
                st.session_state.loaded_model_name = None

    # Quick status indicator
    if st.session_state.detector:
        st.markdown('<p class="badge-success">● Model loaded</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="badge-error">● Model not loaded</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div style="color:#8b949e; font-size:0.78rem; text-align:center;">'
        "COCO 80-class pretrained model<br>No GPU required</div>",
        unsafe_allow_html=True,
    )

# Convenience alias used throughout pages
detector: ObjectDetector | None = st.session_state.detector

# ══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def render_hero(title: str, subtitle: str):
    """Render the blue hero banner at the top of each page."""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_metrics(total: int, unique: int, mode: str, extra: str = ""):
    """Render the four summary metric cards."""
    cols = st.columns(4)
    for col, value, label in zip(
        cols,
        [total, unique, mode, extra or "—"],
        ["Total Detected", "Unique Classes", "Detection Mode", "Avg Confidence"],
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_counts_table_and_chart(counts: dict):
    """Show a table and bar chart of per-class counts."""
    if not counts:
        return
    df = (
        pd.DataFrame(list(counts.items()), columns=["Object", "Count"])
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.markdown('<p class="section-header">Object Counts</p>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col_chart:
        st.markdown('<p class="section-header">Distribution</p>', unsafe_allow_html=True)
        fig = px.bar(
            df,
            x="Object",
            y="Count",
            color="Count",
            color_continuous_scale=["#1f6feb", "#58a6ff", "#79c0ff"],
            template="plotly_dark",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def bgr_to_rgb_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image for Streamlit display."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def image_to_download_bytes(frame_bgr: np.ndarray) -> bytes:
    """Encode a BGR frame as PNG bytes ready for st.download_button."""
    success, buf = cv2.imencode(".png", frame_bgr)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return buf.tobytes()


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    render_hero(
        "🎯 Real-Time Object Detection System",
        "Computer Vision Using OpenCV and YOLOv8 · Detect 80 object classes in real time",
    )

    # Feature cards
    st.markdown('<p class="section-header">Features</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📷", "Webcam", "Live detection from your camera"),
        (c2, "🖼️", "Images", "Upload JPG/PNG for instant analysis"),
        (c3, "🎬", "Video", "Frame-by-frame MP4/AVI processing"),
        (c4, "📊", "Dashboard", "Counts, charts, and confidence data"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2rem;">{icon}</div>'
            f'<div style="font-weight:600; color:#c9d1d9; margin-top:8px;">{title}</div>'
            f'<div style="color:#8b949e; font-size:0.83rem; margin-top:4px;">{desc}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Model info
    st.markdown('<p class="section-header">Model Information</p>', unsafe_allow_html=True)
    if detector:
        info_cols = st.columns(3)
        info_cols[0].info(f"**Model:** {st.session_state.loaded_model_name}")
        info_cols[1].info("**Dataset:** COCO (80 classes)")
        info_cols[2].info("**Device:** CPU (no GPU required)")

        # Show all 80 COCO class names
        with st.expander("📋  View all 80 detectable object classes"):
            names = list(detector.model.names.values())
            # Display in a 5-column grid
            cols = st.columns(5)
            for i, name in enumerate(names):
                cols[i % 5].markdown(f"• {name}")
    else:
        st.warning("Model is not loaded yet. Check sidebar for errors.")

    st.divider()
    st.markdown(
        '<div class="info-box">💡 <b>Getting started:</b> '
        "Select a detection mode from the sidebar, then upload an image or video, "
        "or start your webcam to begin detecting objects in real time.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Webcam Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📷  Webcam Detection":
    render_hero(
        "📷 Webcam Detection",
        "Real-time object detection using your device camera",
    )

    if not detector:
        st.error("❌ Model not loaded. Check the sidebar.")
        st.stop()

    st.markdown(
        '<div class="info-box">'
        "⚠️ <b>Note:</b> Browser security policies restrict direct webcam access in "
        "Streamlit's Python backend. Use the <b>OpenCV Desktop Mode</b> below to run "
        "live detection in a native window, or use the <b>snapshot</b> method to "
        "capture a frame from your webcam and detect objects in it."
        "</div>",
        unsafe_allow_html=True,
    )

    tab_snapshot, tab_desktop = st.tabs(
        ["📸 Webcam Snapshot (Browser)", "🖥️ Desktop Window (OpenCV)"]
    )

    # ── Tab 1: Browser snapshot using st.camera_input ────────────────────
    with tab_snapshot:
        st.markdown('<p class="section-header">Capture & Detect</p>', unsafe_allow_html=True)
        st.write(
            "Click **Take Photo** to capture a frame from your webcam. "
            "The image will be processed instantly by YOLOv8."
        )

        img_file = st.camera_input("Take Photo", label_visibility="collapsed")
        if img_file is not None:
            # Decode captured JPEG into BGR
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Detecting objects …"):
                result = detector.detect_image(frame_bgr, conf_threshold)

            st.success(f"✅ Detected **{result['total']}** object(s)")

            col_orig, col_ann = st.columns(2)
            with col_orig:
                st.markdown("**Original Frame**")
                st.image(bgr_to_rgb_pil(frame_bgr), use_container_width=True)
            with col_ann:
                st.markdown("**Annotated Frame**")
                st.image(bgr_to_rgb_pil(result["annotated_frame"]), use_container_width=True)

            render_metrics(
                result["total"],
                len(result["counts"]),
                "Webcam Snapshot",
                f"{result['inference_ms']:.0f} ms",
            )
            render_counts_table_and_chart(result["counts"])

            # Download annotated snapshot
            st.download_button(
                "⬇️ Download Annotated Image",
                data=image_to_download_bytes(result["annotated_frame"]),
                file_name="webcam_detection.png",
                mime="image/png",
            )

    # ── Tab 2: OpenCV desktop window ──────────────────────────────────────
    with tab_desktop:
        st.markdown('<p class="section-header">OpenCV Live Window</p>', unsafe_allow_html=True)
        st.write(
            "Click **Start Detection** to open an OpenCV window on your desktop "
            "with live bounding boxes. Press **Q** inside that window to stop."
        )

        col_start, col_stop = st.columns(2)

        if col_start.button("▶ Start Detection", key="cam_start"):
            st.session_state.webcam_running = True

        if col_stop.button("⏹ Stop Detection", key="cam_stop"):
            st.session_state.webcam_running = False

        if st.session_state.webcam_running:
            cam_status = st.empty()
            cam_status.info("⏳ Opening webcam …")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cam_status.error(
                    "❌ Webcam not found or already in use. "
                    "Ensure no other application is using the camera."
                )
                st.session_state.webcam_running = False
            else:
                cam_status.success("✅ Webcam active — OpenCV window opened. Press Q to quit.")
                prev_time = time.time()

                try:
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret:
                            cam_status.error("❌ Could not read frame from webcam.")
                            break

                        result = detector.detect_image(frame, conf_threshold)

                        # FPS overlay
                        curr_time = time.time()
                        fps = 1.0 / max(curr_time - prev_time, 1e-6)
                        prev_time = curr_time
                        cv2.putText(
                            result["annotated_frame"],
                            f"FPS: {fps:.1f}",
                            (frame.shape[1] - 120, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.imshow("Object Detection — Press Q to quit", result["annotated_frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            st.session_state.webcam_running = False
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    cam_status.info("Camera released.")


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Image Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️  Image Detection":
    render_hero(
        "🖼️ Image Detection",
        "Upload an image and detect objects using YOLOv8",
    )

    if not detector:
        st.error("❌ Model not loaded. Check the sidebar.")
        st.stop()

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
    )

    if uploaded is not None:
        # Convert uploaded file → BGR frame
        pil_img = Image.open(uploaded).convert("RGB")
        frame_bgr = pil_to_bgr(pil_img)

        with st.spinner("Running YOLOv8 inference …"):
            t0 = time.perf_counter()
            result = detector.detect_image(frame_bgr, conf_threshold)
            wall_ms = (time.perf_counter() - t0) * 1000

        st.success(
            f"✅ Detection complete — **{result['total']}** object(s) found "
            f"in **{wall_ms:.0f} ms**"
        )

        # Side-by-side comparison
        st.markdown('<p class="section-header">Detection Results</p>', unsafe_allow_html=True)
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(pil_img, use_container_width=True)
        with col_ann:
            st.markdown("**Annotated Image**")
            st.image(bgr_to_rgb_pil(result["annotated_frame"]), use_container_width=True)

        # Metrics
        avg_conf = (
            f"{sum(d['confidence'] for d in result['detections']) / len(result['detections']):.0%}"
            if result["detections"]
            else "—"
        )
        render_metrics(result["total"], len(result["counts"]), "Image Upload", avg_conf)

        # Table + chart
        render_counts_table_and_chart(result["counts"])

        # Detailed per-detection table
        if result["detections"]:
            st.markdown(
                '<p class="section-header">Detection Details</p>', unsafe_allow_html=True
            )
            df_det = pd.DataFrame(
                [
                    {
                        "Object": d["class_name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "BBox (x1,y1,x2,y2)": str(d["bbox"]),
                    }
                    for d in result["detections"]
                ]
            )
            st.dataframe(df_det, use_container_width=True, hide_index=True)

        # Download button
        st.download_button(
            "⬇️ Download Annotated Image",
            data=image_to_download_bytes(result["annotated_frame"]),
            file_name="detected_image.png",
            mime="image/png",
        )
    else:
        st.markdown(
            '<div class="info-box">👆 Upload an image above to begin. '
            "YOLOv8 can detect 80 common object classes including people, "
            "vehicles, animals, furniture, and electronics.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Video Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬  Video Detection":
    render_hero(
        "🎬 Video Detection",
        "Upload a video and process it frame by frame with YOLOv8",
    )

    if not detector:
        st.error("❌ Model not loaded. Check the sidebar.")
        st.stop()

    st.markdown(
        '<div class="info-box">'
        "📌 <b>Tip:</b> Processing speed depends on your CPU and the video resolution. "
        "Short clips (under 30 s) at 720p are recommended for best performance. "
        "The processed video will be available for download when complete."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported: MP4, AVI, MOV, MKV",
    )

    if uploaded_video is not None:
        # Save upload to a temp file that OpenCV can read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

        # Destination for the processed video
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        tmp_out_path = str(output_dir / f"processed_{int(time.time())}.mp4")

        # Show video info before processing
        cap_info = cv2.VideoCapture(tmp_in_path)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap_info.get(cv2.CAP_PROP_FPS)
        vid_w   = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h   = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        st.markdown('<p class="section-header">Video Information</p>', unsafe_allow_html=True)
        info_cols = st.columns(4)
        info_cols[0].metric("Frames", total_frames)
        info_cols[1].metric("FPS", f"{vid_fps:.1f}")
        info_cols[2].metric("Width", f"{vid_w}px")
        info_cols[3].metric("Height", f"{vid_h}px")

        if st.button("▶ Process Video", type="primary"):
            progress_bar  = st.progress(0, text="Processing frames …")
            status_text   = st.empty()

            def update_progress(p: float):
                pct = int(p * 100)
                progress_bar.progress(p, text=f"Processing … {pct}%")
                status_text.text(f"Processed {int(p * total_frames)}/{total_frames} frames")

            with st.spinner("Running detection on video …"):
                t_start = time.perf_counter()
                try:
                    summary = detector.process_video(
                        tmp_in_path,
                        tmp_out_path,
                        conf_threshold,
                        progress_callback=update_progress,
                    )
                    elapsed = time.perf_counter() - t_start
                    progress_bar.progress(1.0, text="✅ Processing complete!")
                    status_text.empty()

                    st.success(
                        f"✅ Processed **{summary['frames_processed']}** frames "
                        f"in **{elapsed:.1f} s** "
                        f"({summary['avg_fps']:.1f} fps throughput)"
                    )

                    # Summary metrics
                    render_metrics(
                        summary["total_detections"],
                        summary["unique_classes"],
                        "Video Upload",
                        f"{summary['avg_fps']:.1f} fps",
                    )
                    render_counts_table_and_chart(summary["counts"])

                    # Download processed video
                    if Path(tmp_out_path).exists():
                        with open(tmp_out_path, "rb") as f:
                            video_bytes = f.read()
                        st.download_button(
                            "⬇️ Download Processed Video",
                            data=video_bytes,
                            file_name="detected_video.mp4",
                            mime="video/mp4",
                        )
                except Exception as exc:
                    st.error(f"❌ Video processing failed: {exc}")
                finally:
                    # Clean up the temp input file
                    try:
                        os.unlink(tmp_in_path)
                    except OSError:
                        pass
    else:
        st.markdown(
            '<div class="info-box">👆 Upload a video file above to begin processing. '
            "MP4 files work best for download compatibility.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Page: About Project
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About Project":
    render_hero(
        "ℹ️ About This Project",
        "Architecture, technologies, and limitations",
    )

    st.markdown(
        """
<p class="section-header">Architecture</p>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit (app.py)                   │
│  Sidebar: Settings & Navigation                         │
│  Pages:   Dashboard | Webcam | Image | Video | About   │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│              Detection Engine (detector.py)             │
│  ObjectDetector.detect_image()  → single frame          │
│  ObjectDetector.process_video() → full video            │
│  _draw_detections()             → bounding boxes        │
└──────────────────────┬──────────────────────────────────┘
                       │ uses
┌──────────────────────▼──────────────────────────────────┐
│            Ultralytics YOLOv8n (yolov8n.pt)             │
│  Pretrained on COCO (80 object classes)                 │
│  Runs on CPU — no GPU required                          │
└─────────────────────────────────────────────────────────┘
```
""",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<p class="section-header">Technology Stack</p>', unsafe_allow_html=True
        )
        st.markdown(
            """
| Component | Library |
|---|---|
| Web UI | Streamlit |
| Object Detection | Ultralytics YOLOv8 |
| Video / Image IO | OpenCV |
| Array Ops | NumPy |
| Image Handling | Pillow |
| Charts | Plotly |
| Data Tables | Pandas |
"""
        )

    with col_b:
        st.markdown(
            '<p class="section-header">Detectable Object Classes (sample)</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
- 🧑 person · bicycle · car · motorcycle · airplane
- 🐶 dog · cat · horse · bird · cow
- 🪑 chair · sofa · dining table · bed
- 📺 tv · laptop · phone · keyboard · mouse
- 🍎 apple · banana · orange · pizza · cake
- *(80 total COCO classes)*
"""
        )

    st.markdown(
        '<p class="section-header">Known Limitations</p>', unsafe_allow_html=True
    )
    st.markdown(
        """
- **CPU speed:** Processing is slower than GPU. Expect ~3–8 fps for real-time webcam.
- **80 classes only:** The pretrained COCO model cannot detect objects outside its training set.
- **Webcam in browser:** Full frame-by-frame streaming requires an OpenCV desktop window (Tab 2 on Webcam page).
- **Large videos:** Memory grows with resolution. Prefer 720p or lower for stability.
- **First run:** Model weights (~6 MB) are downloaded from the internet on first use.
"""
    )

    st.markdown(
        '<p class="section-header">Future Enhancements</p>', unsafe_allow_html=True
    )
    st.markdown(
        """
- GPU acceleration (CUDA/MPS) for real-time performance
- Custom model training on user-supplied datasets
- Object tracking (SORT / DeepSORT) across video frames
- REST API endpoint for headless inference
- Export results as JSON or CSV
"""
    )

    st.divider()
    st.markdown(
        '<div class="info-box" style="text-align:center;">'
        "Built with ❤️ using Python · OpenCV · YOLOv8 · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )
