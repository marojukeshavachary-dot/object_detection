"""
detector.py
-----------
Core object detection engine using YOLOv8 (Ultralytics) and OpenCV.

This module is completely separated from Streamlit UI code so that the
detection logic can be reused, tested, or swapped out independently.

First run will download ~6 MB of YOLOv8n weights from the internet.
Subsequent runs use the cached weights stored by Ultralytics.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Available YOLO model variants (ordered lightest → heaviest).
# All are pretrained on COCO (80 common object classes).
AVAILABLE_MODELS = {
    "YOLOv8 Nano  (fastest,  ~3 MB)": "yolov8n.pt",
    "YOLOv8 Small (~11 MB)": "yolov8s.pt",
    "YOLOv8 Medium (~26 MB)": "yolov8m.pt",
}

# Default colour palette for bounding boxes (BGR format for OpenCV)
_COLOUR_PALETTE = [
    (86, 190, 255),   # Amber-yellow
    (60, 220, 100),   # Green
    (255, 100, 80),   # Blue
    (200, 80, 255),   # Purple
    (80, 200, 255),   # Orange
    (100, 255, 200),  # Teal
    (255, 80, 160),   # Pink
    (120, 120, 255),  # Red
]


def get_colour_for_class(class_id: int) -> tuple:
    """Return a consistent BGR colour for a given COCO class id."""
    return _COLOUR_PALETTE[class_id % len(_COLOUR_PALETTE)]


# ── Model loader ──────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Wrapper around a YOLOv8 model.

    Usage
    -----
    detector = ObjectDetector("yolov8n.pt")
    results  = detector.detect_image(frame_bgr, conf_threshold=0.45)
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Load (or download) the requested YOLOv8 model.

        Parameters
        ----------
        model_name : str
            Filename of the YOLO weights, e.g. "yolov8n.pt".
            Ultralytics automatically downloads it on first use.
        """
        self.model_name = model_name
        logger.info("Loading model: %s", model_name)
        try:
            self.model = YOLO(model_name)
            # Warm up the model with a tiny blank frame so the first real
            # inference is not artificially slow.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            raise RuntimeError(
                f"Could not load model '{model_name}'. "
                "Check your internet connection for first-time downloads."
            ) from exc

    # ── Core inference ────────────────────────────────────────────────────────

    def detect_image(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float = 0.45,
    ) -> dict:
        """
        Run YOLO inference on a single BGR frame (as returned by OpenCV).

        Parameters
        ----------
        frame_bgr    : np.ndarray  – BGR image array.
        conf_threshold : float     – Minimum confidence to keep a detection.

        Returns
        -------
        dict with keys:
            annotated_frame : np.ndarray  – Frame with boxes/labels drawn.
            detections      : list[dict]  – Per-object metadata.
            counts          : dict[str,int] – Class → count mapping.
            total           : int          – Total objects detected.
            inference_ms    : float        – Model inference time in ms.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Received an empty frame.")

        t0 = time.perf_counter()
        results = self.model(frame_bgr, conf=conf_threshold, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        counts: dict[str, int] = {}

        # results[0] contains all detections for this single frame
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )
            counts[class_name] = counts.get(class_name, 0) + 1

        # Draw boxes on a copy of the frame
        annotated = self._draw_detections(frame_bgr.copy(), detections)

        return {
            "annotated_frame": annotated,
            "detections": detections,
            "counts": counts,
            "total": len(detections),
            "inference_ms": inference_ms,
        }

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_detections(
        self, frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and confidence scores on the frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            colour = get_colour_for_class(det["class_id"])
            label = f"{det['class_name']}  {det['confidence']:.0%}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background pill
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(y1 - 6, text_h + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - 4),
                (x1 + text_w + 6, label_y + baseline),
                colour,
                cv2.FILLED,
            )
            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Object count overlay (top-left corner)
        count_text = f"Objects: {len(detections)}"
        cv2.rectangle(frame, (6, 6), (200, 34), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            frame,
            count_text,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    # ── Convenience: process a whole video file ───────────────────────────────

    def process_video(
        self,
        input_path: str,
        output_path: str,
        conf_threshold: float = 0.45,
        progress_callback=None,
    ) -> dict:
        """
        Process every frame of a video file and write the annotated output.

        Parameters
        ----------
        input_path       : str   – Path to the source video.
        output_path      : str   – Path for the annotated output video.
        conf_threshold   : float – Confidence filter.
        progress_callback: callable(float) | None
                            Called with progress 0.0→1.0 after each frame.

        Returns
        -------
        dict with summary statistics for the whole video.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use mp4v codec; produces a valid .mp4 for browsers to play
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_counts: dict[str, int] = {}
        frame_idx = 0
        total_detections = 0
        t_start = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect_image(frame, conf_threshold)
                writer.write(result["annotated_frame"])
                total_detections += result["total"]

                # Merge per-frame counts into totals
                for cls, cnt in result["counts"].items():
                    all_counts[cls] = all_counts.get(cls, 0) + cnt

                frame_idx += 1
                if progress_callback and total_frames > 0:
                    progress_callback(frame_idx / total_frames)
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - t_start
        return {
            "frames_processed": frame_idx,
            "total_detections": total_detections,
            "unique_classes": len(all_counts),
            "counts": all_counts,
            "elapsed_seconds": elapsed,
            "avg_fps": frame_idx / elapsed if elapsed > 0 else 0,
        }
