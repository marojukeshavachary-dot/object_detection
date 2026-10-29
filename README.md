# 🎯 VisionMatrix – Real-Time Object Detection System

**Advanced Computer Vision Using OpenCV, YOLOv8, and Vercel Edge**

A production-grade, multi-platform object detection application that detects, localizes, and classifies multiple objects in real time using webcam feeds, uploaded images, or video files. 

### 🌐 Live Web Application (Deployed on Vercel Edge)
- **Primary Live Website:** [https://object-detection-web.vercel.app](https://object-detection-web.vercel.app)
- **Vision Mirror:** [https://object-detection-vision.vercel.app](https://object-detection-vision.vercel.app)
- **Edge Deployment:** [https://object-detection-brown-five.vercel.app](https://object-detection-brown-five.vercel.app)

---

## ✨ Features & UI Capabilities

| Feature | Description |
|---|---|
| 🎨 **High-Tech Glassmorphic UI** | Premium dark theme with glowing telemetry, responsive cards, and clean typography |
| 🖼️ **Instant Sample Benchmarks** | One-click test samples for city transit and sports scenes — no download needed |
| 🔍 **Detections Crop Gallery** | Visual grid displaying cropped object thumbnails with confidence pill badges |
| 🎯 **Target Class Filtering** | Whitelist specific object classes or choose category presets (Vehicles, People, Animals, etc.) |
| 🎨 **Annotation Customizer** | Adjust box thickness (1-5px), toggles for labels & confidences, and 4 color palettes |
| 🎬 **Live Video Stream Pipeline** | Frame-by-frame processing with real-time live preview and MP4 video export |
| 📷 **Dual Webcam Architecture** | Browser snapshot mode for quick analysis + native desktop OpenCV window for maximum FPS |
| 📊 **Interactive Analytics** | Dark Plotly class distribution charts, confidence histograms, and telemetry cards |
| 💾 **Full Export Suite** | One-click export for Annotated PNGs, structured CSV reports, and JSON metadata |

---

## 🛠️ Technologies Used

| Technology | Role |
|---|---|
| Python 3.9+ | Core programming runtime |
| OpenCV (`opencv-python` / headless) | Frame capture, video I/O, anti-aliased HUD rendering |
| Ultralytics YOLOv8 | Anchor-free deep learning object detection model |
| Streamlit | Modern reactive Web UI framework |
| PyTorch | Underlying deep learning tensor backend |
| Plotly | Interactive GPU/browser charts |
| Pandas | Results registry and CSV export |
| Pillow | Image format handling |
| NumPy | High-speed numerical matrix operations |

---

## 💻 System Requirements

- **OS:** Windows 10 / 11, macOS, or Linux
- **Python:** 3.9, 3.10, 3.11, 3.12, or 3.13
- **RAM:** 4 GB minimum (8 GB recommended for video)
- **Disk:** ~500 MB (for dependencies + model weights)
- **Webcam:** Optional (for real-time camera testing)

---

## 📦 Installation & Setup

### 1. Open Terminal in the project directory

```powershell
cd F:\OPEN__CV\object-detection-opencv
```

### 2. Activate the virtual environment

```powershell
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies (if not already installed)

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```powershell
streamlit run app.py
```

Streamlit will open your browser at **http://localhost:8501**.

---

## 📖 How to Use

### 🏠 Dashboard
- Quick-launch workflow cards for image, video, webcam, and technical specs.
- Live YOLOv8 model benchmark comparison table.
- Interactive COCO 80-class taxonomy browser.

### 🖼️ Image Detection
1. Select a built-in sample button (e.g. *City Bus & Pedestrians*) or upload your own image.
2. View side-by-side original and annotated comparison frames.
3. Browse the **Detections Crop Gallery** to inspect cropped thumbnails of every localized object.
4. Export annotated images (PNG), detection tables (CSV), or metadata (JSON).

### 🎬 Video Detection
1. Choose the built-in sample traffic video or upload an MP4/AVI file.
2. Review video telemetry (resolution, native FPS, frame count, estimated duration).
3. Click **Start Video Processing Pipeline**.
4. Watch the **real-time live preview** as frames are being processed by YOLOv8.
5. Play or download the finalized annotated MP4 video.

### 📷 Webcam Detection
- **Browser Snapshot:** Capture photos directly from your browser webcam and run instant inference.
- **Native Desktop OpenCV Window:** Launch a dedicated low-latency desktop window with live FPS overlay. Press **Q** to exit.

---

## ⚙️ Configurable Sidebar Settings

| Setting | Options / Range | Description |
|---|---|---|
| **Model Architecture** | Nano, Small, Medium | Choose between highest speed (Nano) or highest precision (Medium) |
| **Confidence Threshold** | 0.10 – 0.95 | Lower = more detections; Higher = strict detections |
| **NMS IoU Threshold** | 0.10 – 0.90 | Controls box overlap suppression |
| **Target Class Filter** | Category presets or custom multi-select | Whitelist specific classes to detect (e.g. only Vehicles) |
| **Box Thickness** | 1px – 5px | Line thickness of bounding boxes |
| **Color Palette** | Dynamic, Neon, Emerald, Amber | Aesthetic color palette for bounding box annotations |
| **Label & Conf Toggles** | Checkboxes | Show or hide labels and percentage confidence badges |

---

## 📜 License

This project utilizes YOLOv8 under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
