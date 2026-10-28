# 🎯 Real-Time Object Detection System

**Computer Vision Using OpenCV and YOLOv8**

A complete, beginner-friendly object detection application that detects and identifies
multiple objects in real time using a webcam, uploaded images, or video files.
Powered by **YOLOv8** (Ultralytics) and presented through a modern **Streamlit** dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 Webcam Detection | Live snapshot detection via browser camera; OpenCV desktop window for real-time streaming |
| 🖼️ Image Detection | Upload JPG/PNG, get instant bounding boxes, labels, and confidence scores |
| 🎬 Video Detection | Frame-by-frame processing of MP4/AVI files with download option |
| 📊 Results Dashboard | Per-class counts, bar charts, confidence tables |
| ⚙️ Settings | Configurable confidence threshold and model selection |

---

## 🛠️ Technologies Used

| Technology | Role |
|---|---|
| Python 3.9+ | Programming language |
| OpenCV (`opencv-python-headless`) | Frame capture, drawing, video I/O |
| Ultralytics YOLOv8 | Pretrained object detection model (COCO 80 classes) |
| Streamlit | Web UI |
| NumPy | Array operations |
| Pillow | Image format handling |
| Plotly | Interactive bar charts |
| Pandas | Detection results tables |

---

## 💻 System Requirements

- **OS:** Windows 10 / 11 (also works on macOS and Linux)
- **Python:** 3.9, 3.10, or 3.11
- **RAM:** 4 GB minimum (8 GB recommended for video)
- **Disk:** ~500 MB (for packages + model weights)
- **Internet:** Required on first run to download YOLOv8n weights (~6 MB)
- **Webcam:** Optional (only needed for webcam detection)

---

## 📦 Installation

### 1. Clone or download the project

```powershell
cd F:\OPEN__CV\object-detection-opencv
```

### 2. Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

> You should see `(venv)` at the start of your prompt after activation.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- `streamlit` · `opencv-python-headless` · `ultralytics`
- `numpy` · `Pillow` · `plotly` · `pandas`

---

## ▶️ Running the Application

```powershell
streamlit run app.py
```

Streamlit will open your browser automatically at **http://localhost:8501**.

> **First run:** YOLOv8n weights (~6 MB) will be downloaded from the internet
> and cached by Ultralytics. This only happens once.

---

## 📖 How to Use

### 🏠 Dashboard
The landing page shows:
- Feature overview
- Currently loaded model name and device info
- All 80 detectable COCO object classes (expandable)

### 📷 Webcam Detection

**Snapshot mode (browser):**
1. Navigate to **Webcam Detection** in the sidebar.
2. Click the **Take Photo** button on the camera widget.
3. Annotated results with bounding boxes appear instantly.
4. Download the annotated image if desired.

**Desktop window mode (OpenCV):**
1. Click **Start Detection**.
2. An OpenCV window will open on your desktop with live bounding boxes.
3. Press **Q** inside the OpenCV window to stop.
4. Click **Stop Detection** in Streamlit to release the camera.

### 🖼️ Image Detection
1. Navigate to **Image Detection**.
2. Click **Browse files** and upload a JPG, JPEG, or PNG.
3. Detection runs automatically.
4. View original vs. annotated images side by side.
5. Scroll down for per-class counts, confidence scores, and the download button.

### 🎬 Video Detection
1. Navigate to **Video Detection**.
2. Upload an MP4, AVI, MOV, or MKV file.
3. Review video info (frames, FPS, resolution).
4. Click **Process Video**.
5. Watch the progress bar; results summary appears on completion.
6. Download the annotated video.

---

## 🔧 Project Structure

```
object-detection-opencv/
│
├── app.py              ← Streamlit UI (all pages)
├── detector.py         ← Detection engine (YOLO + OpenCV, no UI code)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── .gitignore          ← Excludes outputs, weights, venv
│
├── models/             ← (empty) place custom .pt files here if needed
├── outputs/            ← Processed videos are saved here temporarily
├── sample_images/      ← Drop test images here
└── sample_videos/      ← Drop test videos here
```

---

## ❗ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python-headless` |
| `ModuleNotFoundError: No module named 'ultralytics'` | Run `pip install ultralytics` |
| `streamlit: command not found` | Activate your venv: `venv\Scripts\activate` |
| Model download fails | Check internet connection; Ultralytics downloads from GitHub |
| Webcam not found | Ensure no other app is using the camera; try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in app.py |
| Video processing very slow | Use a shorter clip or lower resolution; CPU inference is limited |
| `OSError: [Errno 28] No space left` | Free disk space; processed videos are written to `outputs/` |

---

## ⚙️ Configurable Settings (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Model | YOLOv8 Nano | Nano is fastest; Medium is most accurate |
| Confidence Threshold | 0.45 | Lower = more detections (including false positives) |

---

## 📐 Architecture Overview

```
Streamlit UI (app.py)
        │
        ▼
ObjectDetector (detector.py)
        │
        ├─ detect_image()   → single BGR frame → annotated frame + metadata
        └─ process_video()  → reads video → writes annotated video
                │
                ▼
        Ultralytics YOLO (yolov8n.pt)
        Pretrained on COCO — 80 object classes
        CPU inference (no GPU required)
```

---

## ⚠️ Limitations

1. **CPU only** — Real-time webcam runs at ~3–8 fps depending on hardware. GPU acceleration (CUDA) would give 30+ fps.
2. **80 COCO classes** — Cannot detect objects outside the training set (e.g., custom logos, rare animals).
3. **Browser webcam streaming** — Full frame-by-frame streaming in the browser requires the OpenCV desktop window mode.
4. **Large videos** — Very high-resolution videos (4K) may be slow or cause memory pressure.
5. **First-run internet** — Weights must be downloaded once; offline environments need manual weight placement.

---

## 🚀 Future Enhancements

- GPU acceleration via CUDA / Apple MPS
- Object tracking across frames (SORT / DeepSORT)
- Custom YOLOv8 model training on user datasets
- REST API for headless inference
- JSON / CSV export of detection results
- WebRTC-based real-time browser streaming

---

## 📜 License

This project uses YOLOv8 under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---

*Built with Python · OpenCV · Ultralytics YOLOv8 · Streamlit*
