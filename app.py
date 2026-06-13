import streamlit as st
import cv2
import math
import tempfile
import numpy as np
from ultralytics import YOLO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PPE Compliance Monitor", layout="wide", page_icon="🦺")

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Mask"}


# ── Load model (cached so it only loads once) ────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")


# ── Annotation helper ────────────────────────────────────────────────────────
def annotate_frame(frame, model, conf_threshold, selected_classes, class_names):
    """Run YOLO on *frame*, draw boxes, return (rgb_frame, violations, compliant, total)."""
    violations = 0
    compliant = 0
    total = 0

    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            conf = math.ceil(box.conf[0].item() * 100) / 100
            cls_id = int(box.cls[0].item())
            label = class_names[cls_id]

            if conf < conf_threshold or label not in selected_classes:
                continue

            total += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if label in VIOLATION_CLASSES:
                color = (0, 0, 255)          # red (BGR)
                txt_color = (255, 255, 255)
                violations += 1
            elif label in COMPLIANT_CLASSES:
                color = (0, 255, 0)          # green
                txt_color = (0, 0, 0)
                compliant += 1
            else:
                color = (255, 0, 0)          # blue
                txt_color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            tag = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, max(th + 4, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 1, cv2.LINE_AA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, violations, compliant, total


# ── Session state defaults ────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:1.2rem 0 0.6rem">
        <span style="font-size:2.6rem">🦺</span>
        <h1 style="margin:0;font-size:2rem">PPE Compliance Monitor</h1>
        <p style="margin:0;color:gray;font-size:1rem">
            Real-time Personal Protective Equipment Detection powered by YOLOv8
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    input_source = st.radio("Input Source", ["📹 Upload Video", "🎥 Live Webcam"])

    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
    )

    selected_classes = st.multiselect(
        "Classes to Detect",
        options=CLASS_NAMES,
        default=CLASS_NAMES,
    )

    st.divider()
    st.info(
        "**Color coding**\n\n"
        "🔴 **Red** — Safety violation\n\n"
        "🟢 **Green** — Compliant equipment\n\n"
        "🔵 **Blue** — Other objects"
    )

model = load_model()


# ── Helper: render live stats row ─────────────────────────────────────────────
def render_stats(cols_placeholder, violations, compliant, total, progress_placeholder):
    denom = compliant + violations
    rate = (compliant / denom * 100) if denom else 100.0

    c1, c2, c3, c4 = cols_placeholder.columns(4)
    c1.metric("Total Detections", total)
    c2.metric("🔴 Violations", violations)
    c3.metric("🟢 Compliant", compliant)
    c4.metric("✅ Compliance Rate", f"{rate:.1f}%")

    progress_placeholder.progress(rate / 100)


# ── Video upload mode ─────────────────────────────────────────────────────────
if input_source == "📹 Upload Video":
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        col_left, col_right = st.columns(2)
        start = col_left.button("▶ Start Detection")
        stop = col_right.button("⏹ Stop")

        frame_ph = st.empty()
        stats_ph = st.container()
        progress_ph = st.empty()

        if stop:
            st.session_state.stop_requested = True

        if start:
            st.session_state.running = True
            st.session_state.stop_requested = False

            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Could not open the video file.")
            else:
                while cap.isOpened():
                    if st.session_state.stop_requested:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb, v, c, t = annotate_frame(
                        frame, model, conf_threshold, selected_classes, CLASS_NAMES
                    )
                    frame_ph.image(rgb, channels="RGB", use_container_width=True)
                    render_stats(stats_ph, v, c, t, progress_ph)

                cap.release()
            st.session_state.running = False
            st.success("Processing complete.")
    else:
        st.markdown(
            "<p style='text-align:center;color:gray;padding:4rem 0'>"
            "Upload a video to get started ↑</p>",
            unsafe_allow_html=True,
        )

# ── Webcam mode ───────────────────────────────────────────────────────────────
else:
    st.warning("⚠️ Ensure your webcam is connected and not used by another application.")

    col_left, col_right = st.columns(2)
    start_cam = col_left.button("🎥 Start Webcam")
    stop_cam = col_right.button("⏹ Stop Webcam")

    frame_ph = st.empty()
    stats_ph = st.container()
    progress_ph = st.empty()

    if stop_cam:
        st.session_state.stop_requested = True

    if start_cam:
        st.session_state.running = True
        st.session_state.stop_requested = False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam. Check that it is connected and free.")
        else:
            while cap.isOpened():
                if st.session_state.stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                rgb, v, c, t = annotate_frame(
                    frame, model, conf_threshold, selected_classes, CLASS_NAMES
                )
                frame_ph.image(rgb, channels="RGB", use_container_width=True)
                render_stats(stats_ph, v, c, t, progress_ph)

            cap.release()
        st.session_state.running = False
