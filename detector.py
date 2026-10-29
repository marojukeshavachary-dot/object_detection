"""
detector.py
-----------
Core object detection engine using YOLOv8 (Ultralytics) and OpenCV.

This module is completely separated from Streamlit UI code so that the
detection logic can be reused, tested, or swapped out independently.

First run will download ~6 MB of YOLOv8n weights from the internet.
Subsequent runs use the cached weights stored by Ultralytics or local project directory.
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

# COCO 80 Category Presets for quick UI filtering
COCO_CATEGORIES = {
    "All (80 classes)": None,
    "People & Faces": ["person"],
    "Vehicles & Traffic": [
        "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter"
    ],
    "Animals & Pets": [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe"
    ],
    "Accessories & Personal": [
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "clock", "watch / clock"
    ],
    "Electronics & Tech": [
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "clock", "watch / clock"
    ],
    "Kitchen & Food": [
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake"
    ],
    "Furniture & Indoor": [
        "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ],
}

# Color palettes (BGR format for OpenCV)
PALETTES = {
    "default": [
        (86, 190, 255),   # Amber-yellow
        (60, 220, 100),   # Green
        (255, 100, 80),   # Blue
        (200, 80, 255),   # Purple
        (80, 200, 255),   # Orange
        (100, 255, 200),  # Teal
        (255, 80, 160),   # Pink
        (120, 120, 255),  # Coral Red
    ],
    "neon": [
        (255, 255, 0),    # Cyan
        (0, 255, 128),    # Neon Lime
        (255, 0, 255),    # Magenta
        (0, 230, 255),    # Electric Yellow
        (255, 128, 0),    # Sky Blue
        (128, 0, 255),    # Hot Pink
    ],
    "emerald": [
        (160, 230, 60),   # Emerald
        (200, 240, 100),  # Mint
        (120, 200, 50),   # Leaf
        (220, 250, 150),  # Bright Jade
        (180, 210, 40),   # Forest
    ],
    "amber": [
        (50, 160, 255),   # Warm Amber
        (60, 200, 255),   # Bright Gold
        (40, 120, 240),   # Sunset Orange
        (80, 180, 250),   # Honey
    ],
}


def get_colour_for_class(class_id: int, palette_name: str = "default") -> tuple:
    """Return a consistent BGR colour for a given COCO class id and palette."""
    palette = PALETTES.get(palette_name, PALETTES["default"])
    return palette[class_id % len(palette)]


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

        # Check local path first to prevent unnecessary network queries
        p = Path(model_name)
        if not p.is_file():
            candidate = Path(__file__).parent / model_name
            if candidate.is_file():
                model_name = str(candidate)

        try:
            self.model = YOLO(model_name)
            # Warm up the model with a tiny blank frame so the first real
            # inference is not artificially slow.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.info("Model loaded successfully: %s", model_name)
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
        iou_threshold: float = 0.45,
        target_classes: list[str] | None = None,
        box_thickness: int = 2,
        show_labels: bool = True,
        show_conf: bool = True,
        color_palette_name: str = "default",
        extract_crops: bool = False,
    ) -> dict:
        """
        Run YOLO inference on a single BGR frame (as returned by OpenCV).

        Parameters
        ----------
        frame_bgr          : np.ndarray – BGR image array.
        conf_threshold     : float      – Minimum confidence to keep a detection.
        iou_threshold      : float      – NMS IoU threshold.
        target_classes     : list[str]  – Optional filter to keep only selected classes.
        box_thickness      : int        – Line thickness of bounding boxes.
        show_labels        : bool       – Whether to display class label.
        show_conf          : bool       – Whether to display confidence percentage.
        color_palette_name : str        – "default", "neon", "emerald", or "amber".
        extract_crops      : bool       – Whether to include RGB crop of each detection.

        Returns
        -------
        dict with keys:
            annotated_frame : np.ndarray  – Frame with boxes/labels drawn.
            detections      : list[dict]  – Per-object metadata (including crop if requested).
            counts          : dict[str,int] – Class → count mapping.
            total           : int          – Total objects detected.
            inference_ms    : float        – Model inference time in ms.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Received an empty frame.")

        t0 = time.perf_counter()
        results = self.model(frame_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        counts: dict[str, int] = {}
        target_classes_set = set(target_classes) if target_classes else None

        img_h, img_w = frame_bgr.shape[:2]

        # results[0] contains all detections for this single frame
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            raw_class_name = self.model.names[class_id]

            # COCO labels wristwatches and clocks as 'clock'
            if raw_class_name == "clock":
                class_name = "watch / clock"
            else:
                class_name = raw_class_name

            # Filter if target_classes specified
            if target_classes_set is not None and class_name not in target_classes_set and raw_class_name not in target_classes_set:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            det_item = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            }

            if extract_crops:
                cx1 = max(0, min(x1, img_w - 1))
                cy1 = max(0, min(y1, img_h - 1))
                cx2 = max(0, min(x2, img_w))
                cy2 = max(0, min(y2, img_h))
                if cx2 > cx1 and cy2 > cy1:
                    crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    det_item["crop_rgb"] = crop_rgb

            detections.append(det_item)
            counts[class_name] = counts.get(class_name, 0) + 1

        # Draw boxes on a copy of the frame
        annotated = self._draw_detections(
            frame_bgr.copy(),
            detections,
            box_thickness=box_thickness,
            show_labels=show_labels,
            show_conf=show_conf,
            palette_name=color_palette_name,
        )

        return {
            "annotated_frame": annotated,
            "detections": detections,
            "counts": counts,
            "total": len(detections),
            "inference_ms": inference_ms,
        }

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        box_thickness: int = 2,
        show_labels: bool = True,
        show_conf: bool = True,
        palette_name: str = "default",
    ) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and confidence scores on the frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            colour = get_colour_for_class(det["class_id"], palette_name)

            # Build label text based on toggles
            if show_labels and show_conf:
                label = f"{det['class_name']}  {det['confidence']:.0%}"
            elif show_labels:
                label = det['class_name']
            elif show_conf:
                label = f"{det['confidence']:.0%}"
            else:
                label = ""

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thickness)

            if label:
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

        # Sleek HUD overlay (top-left badge)
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
        iou_threshold: float = 0.45,
        target_classes: list[str] | None = None,
        progress_callback=None,
        frame_callback=None,
        box_thickness: int = 2,
        show_labels: bool = True,
        show_conf: bool = True,
        color_palette_name: str = "default",
    ) -> dict:
        """
        Process every frame of a video file and write the annotated output.

        Parameters
        ----------
        input_path        : str   – Path to the source video.
        output_path       : str   – Path for the annotated output video.
        conf_threshold    : float – Confidence filter.
        iou_threshold     : float – IoU threshold.
        target_classes    : list[str] | None
        progress_callback : callable(float, int, int) | None
                            Called with (progress_0_to_1, current_frame, total_frames).
        frame_callback    : callable(np.ndarray, int, int) | None
                            Called periodically with latest annotated frame for live UI updates.

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
        max_detections_in_frame = 0
        t_start = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect_image(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    target_classes=target_classes,
                    box_thickness=box_thickness,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    color_palette_name=color_palette_name,
                    extract_crops=False,
                )
                writer.write(result["annotated_frame"])
                current_total = result["total"]
                total_detections += current_total
                if current_total > max_detections_in_frame:
                    max_detections_in_frame = current_total

                # Merge per-frame counts into totals
                for cls, cnt in result["counts"].items():
                    all_counts[cls] = all_counts.get(cls, 0) + cnt

                frame_idx += 1

                # Progress callback
                if progress_callback and total_frames > 0:
                    progress_callback(frame_idx / total_frames, frame_idx, total_frames)

                # Periodic frame callback for live UI preview (every 3 frames or first frame)
                if frame_callback and (frame_idx % 3 == 0 or frame_idx == 1):
                    frame_callback(result["annotated_frame"], frame_idx, total_frames, current_total)
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - t_start
        return {
            "frames_processed": frame_idx,
            "total_detections": total_detections,
            "max_detections_in_single_frame": max_detections_in_frame,
            "unique_classes": len(all_counts),
            "counts": all_counts,
            "elapsed_seconds": elapsed,
            "avg_fps": frame_idx / elapsed if elapsed > 0 else 0,
        }
