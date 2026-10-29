/**
 * VisionMatrix AI - Client-Side Real-Time Object Detection Engine
 * Powered by TensorFlow.js & COCO-SSD
 */

(function () {
  'use strict';

  // --- STATE ---
  let model = null;
  let isModelLoading = false;
  let currentMode = 'webcam'; // 'webcam' | 'image' | 'video'
  let isWebcamRunning = false;
  let webcamStream = null;
  let facingMode = 'user'; // 'user' | 'environment'
  let animationFrameId = null;

  // Performance tracking
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 0;
  let currentLatency = 0;

  // Settings
  let confThreshold = 0.20; // 20% default so small items like watch, mouse, cup are captured
  let activeFilter = 'all'; // 'all' | 'tech' | 'people' | 'vehicles'
  let classSearchQuery = '';
  let showBoxes = true;
  let showLabels = true;
  let showCenters = true;
  let mirrorStream = true;

  // Current detections
  let lastDetections = [];
  let sourceWidth = 640;
  let sourceHeight = 480;

  // --- DOM ELEMENTS ---
  const tabWebcam = document.getElementById('tab-webcam');
  const tabImage = document.getElementById('tab-image');
  const tabVideo = document.getElementById('tab-video');

  const modelStatusIndicator = document.getElementById('model-status-indicator');
  const modelStatusText = document.getElementById('model-status-text');

  const confSlider = document.getElementById('conf-slider');
  const confVal = document.getElementById('conf-val');
  const presetBtns = document.querySelectorAll('.preset-btn');
  const classSearchInput = document.getElementById('class-search');
  const btnClearFilter = document.getElementById('btn-clear-filter');

  const toggleBoxes = document.getElementById('toggle-boxes');
  const toggleLabels = document.getElementById('toggle-labels');
  const toggleCenters = document.getElementById('toggle-centers');
  const toggleMirror = document.getElementById('toggle-mirror');

  const sampleDesk = document.getElementById('sample-desk');
  const sampleStreet = document.getElementById('sample-street');
  const sampleRoom = document.getElementById('sample-room');

  const hudFps = document.getElementById('hud-fps');
  const hudLatency = document.getElementById('hud-latency');
  const hudCount = document.getElementById('hud-count');
  const hudBackend = document.getElementById('hud-backend');

  const viewportStage = document.getElementById('viewport-stage');
  const webcamVideo = document.getElementById('webcam-video');
  const fileVideo = document.getElementById('file-video');
  const sourceImage = document.getElementById('source-image');
  const outputCanvas = document.getElementById('output-canvas');
  const ctx = outputCanvas.getContext('2d');

  const viewportPlaceholder = document.getElementById('viewport-placeholder');
  const placeholderTitle = document.getElementById('placeholder-title');
  const placeholderSubtitle = document.getElementById('placeholder-subtitle');
  const btnStartPrimary = document.getElementById('btn-start-primary');

  const fileDropzone = document.getElementById('file-dropzone');
  const fileInput = document.getElementById('file-input');
  const dropzoneTitle = document.getElementById('dropzone-title');

  const btnCamToggle = document.getElementById('btn-cam-toggle');
  const btnCamText = document.getElementById('btn-cam-text');
  const btnFlipCam = document.getElementById('btn-flip-cam');
  const btnSnapshot = document.getElementById('btn-snapshot');
  const btnDownloadImg = document.getElementById('btn-download-img');

  const statTotal = document.getElementById('stat-total');
  const statUnique = document.getElementById('stat-unique');
  const statPrimary = document.getElementById('stat-primary');
  const statConf = document.getElementById('stat-conf');

  const breakdownBadge = document.getElementById('breakdown-badge');
  const breakdownList = document.getElementById('breakdown-list');
  const cropsCount = document.getElementById('crops-count');
  const cropsGrid = document.getElementById('crops-grid');

  const btnExportJson = document.getElementById('btn-export-json');
  const btnExportCsv = document.getElementById('btn-export-csv');

  // --- COLOR PALETTE & LABELS ---
  const COLOR_PALETTE = [
    '#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899',
    '#8b5cf6', '#14b8a6', '#f97316', '#3b82f6', '#84cc16'
  ];

  // Friendly names and icons for COCO classes
  const CLASS_ALIASES = {
    'clock': 'watch / clock ⌚',
    'mouse': 'mouse 🖱️',
    'cell phone': 'cell phone 📱',
    'laptop': 'laptop 💻',
    'keyboard': 'keyboard ⌨️',
    'cup': 'cup ☕',
    'bottle': 'bottle 🍾',
    'person': 'person 👤',
    'backpack': 'backpack 🎒',
    'handbag': 'handbag 👜',
    'book': 'book 📖',
    'chair': 'chair 🪑',
    'tv': 'monitor / tv 🖥️',
    'car': 'car 🚗',
    'motorcycle': 'motorcycle 🏍️',
    'bicycle': 'bicycle 🚲',
    'dog': 'dog 🐶',
    'cat': 'cat 🐱'
  };

  const TECH_CLASSES = ['clock', 'mouse', 'cell phone', 'laptop', 'keyboard', 'cup', 'bottle', 'book', 'tv', 'remote'];
  const PEOPLE_CLASSES = ['person', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase'];
  const VEHICLE_CLASSES = ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bicycle'];

  function getColorForClass(clsName) {
    let hash = 0;
    for (let i = 0; i < clsName.length; i++) {
      hash = clsName.charCodeAt(i) + ((hash << 5) - hash);
    }
    const idx = Math.abs(hash) % COLOR_PALETTE.length;
    return COLOR_PALETTE[idx];
  }

  function getDisplayName(clsName) {
    return CLASS_ALIASES[clsName.toLowerCase()] || clsName;
  }

  // --- INITIALIZATION ---
  async function init() {
    setupEventListeners();
    await loadModel();
  }

  async function loadModel() {
    isModelLoading = true;
    modelStatusIndicator.classList.add('loading');
    modelStatusText.textContent = 'Loading COCO-SSD...';

    try {
      if (window.tf) {
        await tf.setBackend('webgl');
        await tf.ready();
        hudBackend.textContent = tf.getBackend().toUpperCase();
      }
      model = await cocoSsd.load({ base: 'mobilenet_v2' });
      isModelLoading = false;
      modelStatusIndicator.classList.remove('loading');
      modelStatusText.textContent = 'Engine Ready';
      console.log('VisionMatrix AI: COCO-SSD Model Loaded successfully.');
    } catch (err) {
      console.warn('WebGL backend error, falling back to CPU:', err);
      try {
        if (window.tf) {
          await tf.setBackend('cpu');
          await tf.ready();
          hudBackend.textContent = 'CPU';
        }
        model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        isModelLoading = false;
        modelStatusIndicator.classList.remove('loading');
        modelStatusText.textContent = 'Ready (CPU)';
      } catch (e2) {
        isModelLoading = false;
        modelStatusText.textContent = 'Error Loading Model';
        console.error('Fatal model load error:', e2);
      }
    }
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Mode Switchers
    tabWebcam.addEventListener('click', () => switchMode('webcam'));
    tabImage.addEventListener('click', () => switchMode('image'));
    tabVideo.addEventListener('click', () => switchMode('video'));

    // Confidence Slider
    confSlider.addEventListener('input', (e) => {
      confThreshold = parseInt(e.target.value, 10) / 100;
      confVal.textContent = `${e.target.value}%`;
      if (currentMode === 'image' && sourceImage.src) {
        detectStaticImage(sourceImage);
      }
    });

    // Preset Buttons
    presetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.filter;
        if (currentMode === 'image' && sourceImage.src) {
          detectStaticImage(sourceImage);
        }
      });
    });

    // Class Search Filter
    classSearchInput.addEventListener('input', (e) => {
      classSearchQuery = e.target.value.trim().toLowerCase();
      if (currentMode === 'image' && sourceImage.src) {
        detectStaticImage(sourceImage);
      }
    });

    btnClearFilter.addEventListener('click', () => {
      classSearchInput.value = '';
      classSearchQuery = '';
      presetBtns.forEach(b => b.classList.remove('active'));
      document.querySelector('.preset-btn[data-filter="all"]').classList.add('active');
      activeFilter = 'all';
      if (currentMode === 'image' && sourceImage.src) {
        detectStaticImage(sourceImage);
      }
    });

    // Toggles
    toggleBoxes.addEventListener('change', (e) => { showBoxes = e.target.checked; renderCurrentFrame(); });
    toggleLabels.addEventListener('change', (e) => { showLabels = e.target.checked; renderCurrentFrame(); });
    toggleCenters.addEventListener('change', (e) => { showCenters = e.target.checked; renderCurrentFrame(); });
    toggleMirror.addEventListener('change', (e) => {
      mirrorStream = e.target.checked;
      webcamVideo.style.transform = mirrorStream ? 'scaleX(-1)' : 'none';
    });

    // Webcam Controls
    btnStartPrimary.addEventListener('click', startWebcam);
    btnCamToggle.addEventListener('click', () => {
      if (isWebcamRunning) stopWebcam();
      else startWebcam();
    });

    btnFlipCam.addEventListener('click', async () => {
      facingMode = (facingMode === 'user') ? 'environment' : 'user';
      if (isWebcamRunning) {
        stopWebcam();
        await startWebcam();
      }
    });

    btnSnapshot.addEventListener('click', takeSnapshot);
    btnDownloadImg.addEventListener('click', downloadAnnotatedImage);

    // Export Buttons
    btnExportJson.addEventListener('click', exportJson);
    btnExportCsv.addEventListener('click', exportCsv);

    // File Input & Drag and Drop
    fileInput.addEventListener('change', handleFileInput);
    setupDropzone();

    // Sample Benchmarks
    sampleDesk.addEventListener('click', () => loadSampleImage('desk'));
    sampleStreet.addEventListener('click', () => loadSampleImage('street'));
    sampleRoom.addEventListener('click', () => loadSampleImage('room'));

    // Video play/pause listener
    fileVideo.addEventListener('play', () => {
      if (!animationFrameId) requestAnimationFrame(processVideoFrame);
    });
  }

  // --- MODE SWITCHING ---
  function switchMode(mode) {
    if (currentMode === mode) return;

    // Teardown previous
    if (isWebcamRunning) stopWebcam();
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;

    currentMode = mode;
    tabWebcam.classList.toggle('active', mode === 'webcam');
    tabImage.classList.toggle('active', mode === 'image');
    tabVideo.classList.toggle('active', mode === 'video');

    // Hide elements
    webcamVideo.style.display = 'none';
    fileVideo.style.display = 'none';
    sourceImage.style.display = 'none';
    clearCanvas();

    if (mode === 'webcam') {
      fileDropzone.style.display = 'none';
      viewportPlaceholder.style.display = 'flex';
      placeholderTitle.textContent = 'Webcam Ready';
      placeholderSubtitle.textContent = 'Click "Start Camera" below to begin live real-time detection';
      btnCamToggle.style.display = 'inline-flex';
      btnFlipCam.style.display = 'inline-flex';
      btnSnapshot.style.display = 'inline-flex';
      document.getElementById('samples-group').style.display = 'flex';
    } else if (mode === 'image') {
      viewportPlaceholder.style.display = 'none';
      fileDropzone.style.display = 'flex';
      fileInput.accept = 'image/*';
      dropzoneTitle.textContent = 'Drop an image here or click to browse';
      btnCamToggle.style.display = 'none';
      btnFlipCam.style.display = 'none';
      btnSnapshot.style.display = 'none';
      document.getElementById('samples-group').style.display = 'flex';
    } else if (mode === 'video') {
      viewportPlaceholder.style.display = 'none';
      fileDropzone.style.display = 'flex';
      fileInput.accept = 'video/*';
      dropzoneTitle.textContent = 'Drop a video file here or click to browse (MP4, WebM)';
      btnCamToggle.style.display = 'none';
      btnFlipCam.style.display = 'none';
      btnSnapshot.style.display = 'none';
      document.getElementById('samples-group').style.display = 'none';
    }
  }

  // --- WEBCAM MANAGEMENT ---
  async function startWebcam() {
    if (!model) {
      alert('The AI model is still loading. Please wait a moment and try again.');
      return;
    }

    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: facingMode
        },
        audio: false
      };

      webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamVideo.srcObject = webcamStream;
      webcamVideo.style.display = 'block';
      webcamVideo.style.transform = mirrorStream ? 'scaleX(-1)' : 'none';
      viewportPlaceholder.style.display = 'none';

      await new Promise(resolve => {
        webcamVideo.onloadedmetadata = () => {
          webcamVideo.play();
          resolve();
        };
      });

      resizeCanvasToSource(webcamVideo.videoWidth, webcamVideo.videoHeight);
      isWebcamRunning = true;
      btnCamToggle.classList.add('active-cam');
      btnCamText.textContent = 'Stop Camera';

      lastFrameTime = performance.now();
      requestAnimationFrame(processWebcamFrame);
    } catch (err) {
      console.error('Camera access failed:', err);
      alert('Could not access camera. Please make sure your camera is connected and browser permission is granted.');
    }
  }

  function stopWebcam() {
    isWebcamRunning = false;
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    webcamVideo.pause();
    webcamVideo.srcObject = null;
    webcamVideo.style.display = 'none';
    viewportPlaceholder.style.display = 'flex';
    btnCamToggle.classList.remove('active-cam');
    btnCamText.textContent = 'Start Camera';
    cancelAnimationFrame(animationFrameId);
    clearCanvas();
    updateHUD(0, 0, 0);
  }

  // --- REAL-TIME WEBCAM PROCESSING LOOP ---
  async function processWebcamFrame() {
    if (!isWebcamRunning || !model) return;

    const tStart = performance.now();

    try {
      // Run detection
      const rawPredictions = await model.detect(webcamVideo);
      const filtered = filterPredictions(rawPredictions);
      lastDetections = filtered;

      // Calculate latency
      currentLatency = Math.round(performance.now() - tStart);

      // Calculate FPS
      frameCount++;
      const now = performance.now();
      if (now - lastFrameTime >= 500) {
        currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime) * 10) / 10;
        frameCount = 0;
        lastFrameTime = now;
      }

      // Draw bounding boxes on canvas
      drawDetections(filtered, webcamVideo.videoWidth, webcamVideo.videoHeight, mirrorStream);

      // Update analytics & crops every few frames
      updateHUD(currentFps, currentLatency, filtered.length);
      updateAnalytics(filtered);

      if (frameCount % 4 === 0) {
        updateCropGallery(webcamVideo, filtered);
      }
    } catch (e) {
      console.warn('Frame detection skipped:', e);
    }

    if (isWebcamRunning) {
      animationFrameId = requestAnimationFrame(processWebcamFrame);
    }
  }

  // --- STATIC IMAGE DETECTION ---
  async function detectStaticImage(imgElement) {
    if (!model) return;

    const tStart = performance.now();
    resizeCanvasToSource(imgElement.naturalWidth || imgElement.width, imgElement.naturalHeight || imgElement.height);

    // Draw background image
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    ctx.drawImage(imgElement, 0, 0, outputCanvas.width, outputCanvas.height);

    const rawPredictions = await model.detect(imgElement);
    const filtered = filterPredictions(rawPredictions);
    lastDetections = filtered;

    currentLatency = Math.round(performance.now() - tStart);
    currentFps = 0;

    // Draw overlay on top of image
    drawBoundingBoxes(filtered, outputCanvas.width, outputCanvas.height, false);

    updateHUD(currentFps, currentLatency, filtered.length);
    updateAnalytics(filtered);
    updateCropGallery(imgElement, filtered);
  }

  // --- VIDEO FILE DETECTION ---
  async function processVideoFrame() {
    if (currentMode !== 'video' || fileVideo.paused || fileVideo.ended || !model) {
      animationFrameId = null;
      return;
    }

    const tStart = performance.now();
    resizeCanvasToSource(fileVideo.videoWidth, fileVideo.videoHeight);

    const rawPredictions = await model.detect(fileVideo);
    const filtered = filterPredictions(rawPredictions);
    lastDetections = filtered;

    currentLatency = Math.round(performance.now() - tStart);

    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 500) {
      currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime) * 10) / 10;
      frameCount = 0;
      lastFrameTime = now;
    }

    drawDetections(filtered, fileVideo.videoWidth, fileVideo.videoHeight, false);
    updateHUD(currentFps, currentLatency, filtered.length);
    updateAnalytics(filtered);

    if (frameCount % 6 === 0) {
      updateCropGallery(fileVideo, filtered);
    }

    animationFrameId = requestAnimationFrame(processVideoFrame);
  }

  // --- FILTER PREDICTIONS ---
  function filterPredictions(predictions) {
    return predictions.filter(pred => {
      // Confidence check
      if (pred.score < confThreshold) return false;

      const cls = pred.class.toLowerCase();

      // Category Preset Filter
      if (activeFilter === 'tech' && !TECH_CLASSES.includes(cls)) return false;
      if (activeFilter === 'people' && !PEOPLE_CLASSES.includes(cls)) return false;
      if (activeFilter === 'vehicles' && !VEHICLE_CLASSES.includes(cls)) return false;

      // Search Query Filter
      if (classSearchQuery && !cls.includes(classSearchQuery) && !getDisplayName(cls).toLowerCase().includes(classSearchQuery)) {
        return false;
      }

      return true;
    });
  }

  // --- DRAWING PIPELINE ---
  function resizeCanvasToSource(w, h) {
    if (w && h && (outputCanvas.width !== w || outputCanvas.height !== h)) {
      outputCanvas.width = w;
      outputCanvas.height = h;
      sourceWidth = w;
      sourceHeight = h;
    }
  }

  function clearCanvas() {
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  }

  function drawDetections(predictions, width, height, isMirrored) {
    clearCanvas();
    drawBoundingBoxes(predictions, width, height, isMirrored);
  }

  function drawBoundingBoxes(predictions, width, height, isMirrored) {
    predictions.forEach(pred => {
      let [x, y, w, h] = pred.bbox;

      // If mirrored (front-facing webcam), invert x coordinate
      if (isMirrored) {
        x = width - x - w;
      }

      const color = getColorForClass(pred.class);
      const displayName = getDisplayName(pred.class);
      const scorePct = Math.round(pred.score * 100);

      // Bounding Box
      if (showBoxes) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.round(width / 350));
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
      }

      // Center Target Crosshair
      if (showCenters) {
        const cx = x + w / 2;
        const cy = y + h / 2;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
        ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
        ctx.stroke();
        ctx.restore();
      }

      // Labels & Confidence Chip
      if (showLabels) {
        const labelText = `${displayName} ${scorePct}%`;
        const fontSize = Math.max(12, Math.min(18, Math.round(width / 50)));
        ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;

        const textMetrics = ctx.measureText(labelText);
        const padding = 6;
        const chipW = textMetrics.width + padding * 2;
        const chipH = fontSize + padding * 2;
        const chipY = Math.max(0, y - chipH);

        // Chip Background
        ctx.fillStyle = 'rgba(10, 14, 23, 0.85)';
        ctx.fillRect(x, chipY, chipW, chipH);

        // Chip Left Accent Strip
        ctx.fillStyle = color;
        ctx.fillRect(x, chipY, 3, chipH);

        // Text
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'top';
        ctx.fillText(labelText, x + padding + 2, chipY + padding);
      }
    });
  }

  function renderCurrentFrame() {
    if (currentMode === 'image' && sourceImage.src) {
      detectStaticImage(sourceImage);
    }
  }

  // --- ANALYTICS & HUD ---
  function updateHUD(fps, latency, count) {
    hudFps.textContent = fps.toFixed(1);
    hudLatency.textContent = `${latency} ms`;
    hudCount.textContent = count;
  }

  function updateAnalytics(predictions) {
    statTotal.textContent = predictions.length;

    if (predictions.length === 0) {
      statUnique.textContent = '0';
      statPrimary.textContent = '-';
      statConf.textContent = '0%';
      breakdownBadge.textContent = '0 detected';
      breakdownList.innerHTML = `
        <div class="empty-state">
          <p>No objects detected</p>
          <span>Adjust sensitivity slider or change view</span>
        </div>`;
      return;
    }

    // Counts by class
    const counts = {};
    let confSum = 0;

    predictions.forEach(p => {
      counts[p.class] = (counts[p.class] || 0) + 1;
      confSum += p.score;
    });

    const uniqueClasses = Object.keys(counts);
    statUnique.textContent = uniqueClasses.length;

    // Top class
    let topClass = uniqueClasses[0];
    let topCount = counts[topClass];
    uniqueClasses.forEach(c => {
      if (counts[c] > topCount) {
        topClass = c;
        topCount = counts[c];
      }
    });
    statPrimary.textContent = getDisplayName(topClass);

    // Avg confidence
    const avgConf = Math.round((confSum / predictions.length) * 100);
    statConf.textContent = `${avgConf}%`;

    // Breakdown list
    breakdownBadge.textContent = `${predictions.length} found`;
    breakdownList.innerHTML = uniqueClasses.map(cls => {
      const count = counts[cls];
      const color = getColorForClass(cls);
      const name = getDisplayName(cls);
      return `
        <div class="breakdown-item" style="border-left-color: ${color}">
          <div class="item-left">
            <span class="item-name">${name}</span>
            <span class="item-count">(${count})</span>
          </div>
          <span class="item-conf" style="color: ${color}">${count}x</span>
        </div>
      `;
    }).join('');
  }

  // --- CROP EXTRACTION GALLERY ---
  function updateCropGallery(sourceMedia, predictions) {
    if (predictions.length === 0) {
      cropsCount.textContent = '0 items';
      cropsGrid.innerHTML = `<div class="empty-state"><span>No objects currently detected</span></div>`;
      return;
    }

    cropsCount.textContent = `${predictions.length} items`;

    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d');

    const cardsHtml = predictions.slice(0, 12).map((pred, idx) => {
      let [x, y, w, h] = pred.bbox;

      // Handle video source boundaries
      const sw = sourceMedia.videoWidth || sourceMedia.naturalWidth || sourceMedia.width;
      const sh = sourceMedia.videoHeight || sourceMedia.naturalHeight || sourceMedia.height;

      // Padding around crop
      const pad = 10;
      const sx = Math.max(0, x - pad);
      const sy = Math.max(0, y - pad);
      const sWidth = Math.min(sw - sx, w + pad * 2);
      const sHeight = Math.min(sh - sy, h + pad * 2);

      if (sWidth <= 0 || sHeight <= 0) return '';

      cropCanvas.width = sWidth;
      cropCanvas.height = sHeight;
      cropCtx.drawImage(sourceMedia, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight);

      const dataUrl = cropCanvas.toDataURL('image/jpeg', 0.8);
      const name = getDisplayName(pred.class);
      const score = Math.round(pred.score * 100);

      return `
        <div class="crop-card">
          <img src="${dataUrl}" alt="${name}">
          <div class="crop-info">
            <span class="crop-name" title="${name}">${name}</span>
            <span class="crop-score">${score}%</span>
          </div>
        </div>
      `;
    }).join('');

    cropsGrid.innerHTML = cardsHtml;
  }

  // --- DROPZONE & FILE INPUT ---
  function setupDropzone() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      fileDropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    fileDropzone.addEventListener('dragover', () => fileDropzone.classList.add('dragover'));
    fileDropzone.addEventListener('dragleave', () => fileDropzone.classList.remove('dragover'));
    fileDropzone.addEventListener('drop', (e) => {
      fileDropzone.classList.remove('dragover');
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length) handleFile(files[0]);
    });
  }

  function handleFileInput(e) {
    if (e.target.files.length) {
      handleFile(e.target.files[0]);
    }
  }

  function handleFile(file) {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        sourceImage.onload = () => {
          fileDropzone.style.display = 'none';
          sourceImage.style.display = 'block';
          detectStaticImage(sourceImage);
        };
        sourceImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      fileVideo.src = url;
      fileDropzone.style.display = 'none';
      fileVideo.style.display = 'block';
      fileVideo.play();
    }
  }

  // --- SAMPLE BENCHMARKS ---
  function loadSampleImage(type) {
    switchMode('image');
    fileDropzone.style.display = 'none';
    sourceImage.style.display = 'block';

    // SVG synthetic benchmark image generating high accuracy COCO objects
    let svg = '';
    if (type === 'desk') {
      svg = `
      <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">
        <rect width="1000" height="700" fill="#1e293b"/>
        <rect x="50" y="100" width="900" height="500" rx="20" fill="#334155"/>
        <!-- Laptop -->
        <rect x="250" y="200" width="350" height="220" rx="8" fill="#475569" stroke="#94a3b8" stroke-width="4"/>
        <rect x="270" y="220" width="310" height="180" fill="#0f172a"/>
        <rect x="220" y="420" width="410" height="25" rx="6" fill="#64748b"/>
        <!-- Mouse -->
        <ellipse cx="680" cy="380" rx="35" ry="55" fill="#e2e8f0" stroke="#0f172a" stroke-width="3"/>
        <line x1="680" y1="330" x2="680" y2="380" stroke="#64748b" stroke-width="2"/>
        <!-- Watch / Clock -->
        <circle cx="150" cy="360" r="45" fill="#f8fafc" stroke="#f59e0b" stroke-width="8"/>
        <circle cx="150" cy="360" r="4" fill="#0f172a"/>
        <line x1="150" y1="360" x2="150" y2="330" stroke="#0f172a" stroke-width="4"/>
        <line x1="150" y1="360" x2="175" y2="360" stroke="#0f172a" stroke-width="3"/>
        <rect x="135" y="270" width="30" height="45" fill="#78350f"/>
        <rect x="135" y="405" width="30" height="45" fill="#78350f"/>
        <!-- Coffee Cup -->
        <circle cx="750" cy="240" r="35" fill="#ef4444"/>
        <circle cx="750" cy="240" r="28" fill="#451a03"/>
        <path d="M785 230 C800 230 800 250 785 250" stroke="#ef4444" stroke-width="8" fill="none"/>
        <!-- Cell Phone -->
        <rect x="120" y="460" width="60" height="110" rx="10" fill="#020617" stroke="#38bdf8" stroke-width="3"/>
        <rect x="125" y="470" width="50" height="90" fill="#1e293b"/>
      </svg>`;
    } else if (type === 'street') {
      svg = `
      <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">
        <rect width="1000" height="700" fill="#0f172a"/>
        <rect x="0" y="400" width="1000" height="300" fill="#334155"/>
        <line x1="0" y1="550" x2="1000" y2="550" stroke="#f59e0b" stroke-width="8" stroke-dasharray="40 30"/>
        <!-- Car 1 -->
        <rect x="150" y="440" width="280" height="90" rx="15" fill="#3b82f6"/>
        <rect x="200" y="380" width="160" height="70" rx="10" fill="#60a5fa"/>
        <circle cx="200" cy="530" r="30" fill="#020617" stroke="#94a3b8" stroke-width="6"/>
        <circle cx="380" cy="530" r="30" fill="#020617" stroke="#94a3b8" stroke-width="6"/>
        <!-- Person -->
        <circle cx="650" cy="380" r="25" fill="#fed7aa"/>
        <rect x="635" y="405" width="30" height="80" rx="8" fill="#ef4444"/>
        <line x1="640" y1="485" x2="640" y2="570" stroke="#1e293b" stroke-width="12"/>
        <line x1="660" y1="485" x2="660" y2="570" stroke="#1e293b" stroke-width="12"/>
        <!-- Bicycle -->
        <circle cx="750" cy="510" r="35" fill="none" stroke="#22d3ee" stroke-width="5"/>
        <circle cx="860" cy="510" r="35" fill="none" stroke="#22d3ee" stroke-width="5"/>
        <path d="M750 510 L805 510 L850 460 L805 460 Z" fill="none" stroke="#22d3ee" stroke-width="6"/>
      </svg>`;
    } else {
      svg = `
      <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">
        <rect width="1000" height="700" fill="#1e1b4b"/>
        <!-- Couch / Sofa -->
        <rect x="200" y="320" width="600" height="220" rx="20" fill="#4338ca"/>
        <rect x="240" y="240" width="520" height="160" rx="15" fill="#4f46e5"/>
        <!-- Potted plant -->
        <polygon points="100,550 140,550 130,470 110,470" fill="#92400e"/>
        <circle cx="120" cy="430" r="40" fill="#15803d"/>
        <circle cx="95" cy="410" r="30" fill="#22c55e"/>
        <circle cx="145" cy="410" r="30" fill="#16a34a"/>
        <!-- Clock on wall -->
        <circle cx="500" cy="120" r="45" fill="#ffffff" stroke="#f59e0b" stroke-width="6"/>
        <line x1="500" y1="120" x2="500" y2="95" stroke="#000" stroke-width="3"/>
        <line x1="500" y1="120" x2="520" y2="120" stroke="#000" stroke-width="3"/>
      </svg>`;
    }

    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    sourceImage.onload = () => {
      detectStaticImage(sourceImage);
    };
    sourceImage.src = url;
  }

  // --- SNAPSHOT & EXPORT ---
  function takeSnapshot() {
    if (!isWebcamRunning) return;
    const saveCanvas = document.createElement('canvas');
    saveCanvas.width = outputCanvas.width;
    saveCanvas.height = outputCanvas.height;
    const sCtx = saveCanvas.getContext('2d');

    if (mirrorStream) {
      sCtx.save();
      sCtx.scale(-1, 1);
      sCtx.drawImage(webcamVideo, -saveCanvas.width, 0, saveCanvas.width, saveCanvas.height);
      sCtx.restore();
    } else {
      sCtx.drawImage(webcamVideo, 0, 0, saveCanvas.width, saveCanvas.height);
    }
    sCtx.drawImage(outputCanvas, 0, 0);

    const a = document.createElement('a');
    a.download = `snapshot_${Date.now()}.png`;
    a.href = saveCanvas.toDataURL('image/png');
    a.click();
  }

  function downloadAnnotatedImage() {
    let saveCanvas = document.createElement('canvas');
    saveCanvas.width = outputCanvas.width;
    saveCanvas.height = outputCanvas.height;
    const sCtx = saveCanvas.getContext('2d');

    if (currentMode === 'webcam' && isWebcamRunning) {
      takeSnapshot();
      return;
    } else if (currentMode === 'image' && sourceImage.src) {
      sCtx.drawImage(sourceImage, 0, 0, saveCanvas.width, saveCanvas.height);
      sCtx.drawImage(outputCanvas, 0, 0);
    } else if (currentMode === 'video' && fileVideo.src) {
      sCtx.drawImage(fileVideo, 0, 0, saveCanvas.width, saveCanvas.height);
      sCtx.drawImage(outputCanvas, 0, 0);
    } else {
      sCtx.drawImage(outputCanvas, 0, 0);
    }

    const a = document.createElement('a');
    a.download = `annotated_${Date.now()}.png`;
    a.href = saveCanvas.toDataURL('image/png');
    a.click();
  }

  function exportJson() {
    const data = {
      timestamp: new Date().toISOString(),
      model: 'COCO-SSD v2 MobileNet',
      confidenceThreshold: confThreshold,
      totalDetections: lastDetections.length,
      detections: lastDetections.map(d => ({
        class: d.class,
        displayName: getDisplayName(d.class),
        confidence: Math.round(d.score * 1000) / 1000,
        bbox: {
          x: Math.round(d.bbox[0]),
          y: Math.round(d.bbox[1]),
          width: Math.round(d.bbox[2]),
          height: Math.round(d.bbox[3])
        }
      }))
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.download = `detections_${Date.now()}.json`;
    a.href = URL.createObjectURL(blob);
    a.click();
  }

  function exportCsv() {
    const headers = ['id', 'class', 'display_name', 'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'];
    const rows = lastDetections.map((d, i) => [
      i + 1,
      `"${d.class}"`,
      `"${getDisplayName(d.class)}"`,
      (d.score).toFixed(3),
      Math.round(d.bbox[0]),
      Math.round(d.bbox[1]),
      Math.round(d.bbox[2]),
      Math.round(d.bbox[3])
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const a = document.createElement('a');
    a.download = `detections_${Date.now()}.csv`;
    a.href = URL.createObjectURL(blob);
    a.click();
  }

  // --- START APP ---
  window.addEventListener('DOMContentLoaded', init);

})();
