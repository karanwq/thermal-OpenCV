import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from base64 import b64encode

import cv2
import numpy as np
import torch
from flask import Flask, render_template_string, request
from ultralytics import YOLO

app = Flask(__name__)
if not torch.cuda.is_available():
    torch.set_num_threads(1)

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Thermal OpenCV // Detection Unit</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
      :root {
        color-scheme: dark;
        --bg: #050a0a;
        --bg-grid: #081112;
        --panel: #0b1415;
        --panel-2: #0e191b;
        --border: #1c3235;
        --border-bright: #2b4d51;
        --text: #d8ece9;
        --muted: #6f9591;
        --faint: #3f5c59;
        --hot: #ff6a1a;
        --hot-dim: #7a3717;
        --cold: #22d3ee;
        --danger: #ff4444;
        --warn: #ffb020;
        --safe: #2fe08a;
        --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
        --display: "Space Grotesk", Arial, sans-serif;
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        min-height: 100vh;
        background:
          radial-gradient(circle at 15% 0%, rgba(255, 106, 26, 0.08), transparent 40%),
          radial-gradient(circle at 85% 15%, rgba(34, 211, 238, 0.06), transparent 45%),
          repeating-linear-gradient(0deg, var(--bg-grid) 0px, var(--bg-grid) 1px, var(--bg) 1px, var(--bg) 28px),
          var(--bg);
        color: var(--text);
        font-family: var(--display);
      }

      .wrap {
        max-width: 1080px;
        margin: 0 auto;
        padding: 28px 20px 56px;
      }

      /* ---------- Header ---------- */
      .hud-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        color: var(--faint);
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 26px;
        flex-wrap: wrap;
        gap: 8px;
      }

      .hud-bar .dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--safe);
        margin-right: 8px;
        box-shadow: 0 0 8px var(--safe);
      }

      .hero {
        padding: 6px 0 28px;
      }

      .eyebrow {
        font-family: var(--mono);
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        color: var(--hot);
        text-transform: uppercase;
        margin: 0 0 12px;
      }

      h1 {
        margin: 0 0 12px;
        font-size: clamp(2rem, 5vw, 2.9rem);
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.01em;
      }

      h1 span {
        color: var(--hot);
      }

      .hero p {
        margin: 0;
        max-width: 56ch;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.55;
      }

      /* ---------- Layout ---------- */
      .grid {
        display: grid;
        gap: 20px;
        grid-template-columns: 1fr;
      }

      @media (min-width: 900px) {
        .grid {
          grid-template-columns: 320px 1fr;
          align-items: start;
        }
      }

      .panel {
        background: linear-gradient(180deg, var(--panel-2), var(--panel));
        border: 1px solid var(--border);
        border-radius: 4px;
        overflow: hidden;
      }

      .panel-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .panel-head .tag {
        color: var(--cold);
      }

      .panel-body {
        padding: 18px 16px;
      }

      /* ---------- Upload form ---------- */
      .error-banner {
        margin: 0 0 14px;
        padding: 10px 14px;
        border: 1px solid var(--danger);
        background: rgba(255, 68, 68, 0.08);
        color: #ffb3b3;
        font-family: var(--mono);
        font-size: 0.78rem;
        border-radius: 4px;
      }

      .dropzone {
        position: relative;
        border: 1px dashed var(--border-bright);
        border-radius: 4px;
        padding: 28px 14px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.15s ease, background 0.15s ease;
        background: rgba(255, 106, 26, 0.02);
      }

      .dropzone:hover,
      .dropzone.drag {
        border-color: var(--hot);
        background: rgba(255, 106, 26, 0.06);
      }

      .dropzone input[type="file"] {
        position: absolute;
        inset: 0;
        opacity: 0;
        cursor: pointer;
      }

      .dropzone .glyph {
        font-family: var(--mono);
        font-size: 1.6rem;
        color: var(--hot);
        margin-bottom: 8px;
        line-height: 1;
      }

      .dropzone .primary {
        font-size: 0.92rem;
        color: var(--text);
        margin-bottom: 4px;
      }

      .dropzone .filename {
        font-family: var(--mono);
        font-size: 0.75rem;
        color: var(--cold);
        margin-top: 8px;
        word-break: break-all;
        min-height: 1em;
      }

      .dropzone .hint {
        font-family: var(--mono);
        font-size: 0.68rem;
        color: var(--faint);
        letter-spacing: 0.04em;
      }

      button.submit {
        appearance: none;
        border: 0;
        width: 100%;
        margin-top: 14px;
        border-radius: 4px;
        padding: 12px 16px;
        background: var(--hot);
        color: #150900;
        font-family: var(--mono);
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        cursor: pointer;
        transition: background 0.15s ease, transform 0.05s ease;
      }

      button.submit:hover { background: #ff7d38; }
      button.submit:active { transform: scale(0.99); }
      button.submit:disabled { background: var(--hot-dim); color: var(--muted); cursor: progress; }

      button.submit:focus-visible,
      .dropzone:focus-within {
        outline: 2px solid var(--cold);
        outline-offset: 2px;
      }

      .local-note {
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid var(--border);
        font-size: 0.78rem;
        color: var(--faint);
        line-height: 1.5;
      }

      .local-note code {
        font-family: var(--mono);
        color: var(--muted);
        background: rgba(255,255,255,0.04);
        padding: 1px 5px;
        border-radius: 3px;
      }

      /* ---------- Viewfinder / output ---------- */
      .viewfinder {
        position: relative;
        background: #020505;
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }

      .viewfinder::before,
      .viewfinder::after,
      .vf-corner-a,
      .vf-corner-b {
        content: "";
        position: absolute;
        width: 22px;
        height: 22px;
        border: 2px solid var(--hot);
        opacity: 0.85;
        z-index: 2;
      }

      .viewfinder::before { top: 10px; left: 10px; border-right: 0; border-bottom: 0; }
      .viewfinder::after { top: 10px; right: 10px; border-left: 0; border-bottom: 0; }
      .vf-corner-a { bottom: 10px; left: 10px; border-right: 0; border-top: 0; }
      .vf-corner-b { bottom: 10px; right: 10px; border-left: 0; border-top: 0; }

      .vf-scan {
        position: absolute;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cold), transparent);
        opacity: 0.65;
        animation: scan 3.4s linear infinite;
        z-index: 3;
        pointer-events: none;
      }

      @keyframes scan {
        0% { top: 4%; }
        50% { top: 94%; }
        100% { top: 4%; }
      }

      .image-box {
        width: 100%;
      }

      .image-box img {
        display: block;
        width: 100%;
        height: auto;
      }

      .empty-state {
        text-align: center;
        padding: 40px 24px;
        color: var(--faint);
        font-family: var(--mono);
        font-size: 0.8rem;
        letter-spacing: 0.06em;
      }

      .empty-state .big {
        font-size: 2rem;
        margin-bottom: 10px;
        color: var(--border-bright);
      }

      /* ---------- Readout / detections table ---------- */
      .readout {
        border-top: 1px solid var(--border);
        font-family: var(--mono);
      }

      .readout-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        background: rgba(255,255,255,0.02);
      }

      .readout-head .count {
        color: var(--hot);
      }

      table.det-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
      }

      table.det-table th {
        text-align: left;
        font-weight: 500;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--faint);
        padding: 8px 16px;
        border-bottom: 1px solid var(--border);
      }

      table.det-table td {
        padding: 9px 16px;
        border-bottom: 1px solid var(--border);
        color: var(--text);
      }

      table.det-table tr:last-child td { border-bottom: 0; }

      .swatch {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 2px;
        margin-right: 8px;
        vertical-align: middle;
      }

      .swatch.close { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
      .swatch.mid { background: var(--warn); box-shadow: 0 0 6px var(--warn); }
      .swatch.far { background: var(--safe); box-shadow: 0 0 6px var(--safe); }
      .swatch.unknown { background: var(--faint); }

      .no-detections {
        padding: 18px 16px;
        color: var(--faint);
        font-size: 0.8rem;
      }

      @media (prefers-reduced-motion: reduce) {
        .vf-scan { animation: none; top: 50%; }
      }
    </style>
  </head>
  <body>
    <main class="wrap">
      <div class="hud-bar">
        <span><span class="dot"></span>SYS.STATUS &mdash; MODEL LOADED &mdash; INFERENCE READY</span>
        <span>THERM-01 // OBJECT DETECTION UNIT</span>
      </div>

      <section class="hero">
        <p class="eyebrow">Thermal vision &middot; YOLO detection &middot; distance estimation</p>
        <h1>Upload a frame.<br>See what it <span>sees</span>.</h1>
        <p>Run any image through a pseudo-thermal render with live object detection overlaid &mdash; class, confidence, and estimated distance banded by proximity.</p>
      </section>

      <section class="grid">
        <div class="panel">
          <div class="panel-head">
            <span>Source</span>
            <span class="tag">01</span>
          </div>
          <div class="panel-body">
            {% if error %}
            <div class="error-banner">ERR &mdash; {{ error }}</div>
            {% endif %}
            <form method="post" action="/process" enctype="multipart/form-data" id="scan-form">
              <label class="dropzone" id="dropzone">
                <input type="file" name="image" accept="image/*" required id="file-input">
                <div class="glyph">&#9670;</div>
                <div class="primary">Drop image or click to browse</div>
                <div class="hint">JPG &middot; PNG &middot; WEBP</div>
                <div class="filename" id="filename"></div>
              </label>
              <button type="submit" class="submit" id="submit-btn">Render thermal scan</button>
            </form>
            <div class="local-note">
              For live webcam tracking with persistent object IDs, run <code>thermal_cam.py</code> locally instead of this web uploader.
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <span>Output</span>
            <span class="tag">02</span>
          </div>
          <div class="viewfinder">
            {% if image_data %}
              <div class="vf-scan"></div>
              <div class="image-box">
                <img src="data:image/png;base64,{{ image_data }}" alt="Thermal processed image with detections">
              </div>
            {% else %}
              <div class="empty-state">
                <div class="big">&#9671;</div>
                AWAITING INPUT &mdash; SELECT AN IMAGE TO BEGIN SCAN
              </div>
            {% endif %}
          </div>

          {% if image_data %}
          <div class="readout">
            <div class="readout-head">
              <span>Detections</span>
              <span class="count">{{ detections|length }} object{{ 's' if detections|length != 1 else '' }}</span>
            </div>
            {% if detections %}
            <table class="det-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Range</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {% for d in detections %}
                <tr>
                  <td>
                    {% if d.distance is none %}
                      <span class="swatch unknown"></span>
                    {% elif d.distance < 4 %}
                      <span class="swatch close"></span>
                    {% elif d.distance < 12 %}
                      <span class="swatch mid"></span>
                    {% else %}
                      <span class="swatch far"></span>
                    {% endif %}
                    {{ d.label|upper }}
                  </td>
                  <td>{{ '%.2f'|format(d.distance) + 'm' if d.distance is not none else 'N/A' }}</td>
                  <td>{{ '%.0f'|format(d.confidence * 100) }}%</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
            {% else %}
            <div class="no-detections">No objects cleared the confidence threshold on this frame.</div>
            {% endif %}
          </div>
          {% endif %}
        </div>
      </section>
    </main>

    <script>
      const dropzone = document.getElementById('dropzone');
      const fileInput = document.getElementById('file-input');
      const filenameEl = document.getElementById('filename');
      const form = document.getElementById('scan-form');
      const submitBtn = document.getElementById('submit-btn');

      function showFilename() {
        if (fileInput.files && fileInput.files[0]) {
          filenameEl.textContent = fileInput.files[0].name;
        }
      }

      fileInput.addEventListener('change', showFilename);

      ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
          e.preventDefault();
          dropzone.classList.add('drag');
        });
      });

      ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
          e.preventDefault();
          dropzone.classList.remove('drag');
        });
      });

      dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          fileInput.files = e.dataTransfer.files;
          showFilename();
        }
      });

      form.addEventListener('submit', () => {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Analyzing frame...';
      });
    </script>
  </body>
</html>
"""

# ---------------------------------------------------------------------------
# Distance estimation (ported from thermal_cam.py's DistanceEstimator)
# ---------------------------------------------------------------------------

OBJECT_DIMENSIONS = {
    "person":         {"height": 1.70, "width": 0.45, "reliability": 0.95},
    "bicycle":        {"height": 1.10, "width": 0.60, "reliability": 0.75},
    "car":            {"height": 1.50, "width": 1.80, "reliability": 0.80},
    "motorcycle":     {"height": 1.20, "width": 0.80, "reliability": 0.75},
    "airplane":       {"height": 6.00, "width": 30.0, "reliability": 0.40},
    "bus":            {"height": 3.00, "width": 2.50, "reliability": 0.75},
    "train":          {"height": 3.50, "width": 3.00, "reliability": 0.55},
    "truck":          {"height": 2.50, "width": 2.50, "reliability": 0.65},
    "boat":           {"height": 1.50, "width": 2.50, "reliability": 0.40},
    "traffic light":  {"height": 0.80, "width": 0.30, "reliability": 0.75},
    "fire hydrant":   {"height": 0.75, "width": 0.30, "reliability": 0.85},
    "stop sign":      {"height": 0.75, "width": 0.75, "reliability": 0.90},
    "parking meter":  {"height": 1.20, "width": 0.20, "reliability": 0.80},
    "bench":          {"height": 0.90, "width": 1.50, "reliability": 0.65},
    "bird":           {"height": 0.15, "width": 0.20, "reliability": 0.35},
    "cat":            {"height": 0.25, "width": 0.25, "reliability": 0.60},
    "dog":            {"height": 0.50, "width": 0.40, "reliability": 0.50},
    "horse":          {"height": 1.60, "width": 1.00, "reliability": 0.70},
    "sheep":          {"height": 0.80, "width": 0.60, "reliability": 0.65},
    "cow":            {"height": 1.40, "width": 0.70, "reliability": 0.65},
    "elephant":       {"height": 3.00, "width": 2.00, "reliability": 0.70},
    "bear":           {"height": 1.20, "width": 0.90, "reliability": 0.55},
    "zebra":          {"height": 1.40, "width": 1.00, "reliability": 0.75},
    "giraffe":        {"height": 4.50, "width": 1.00, "reliability": 0.75},
    "backpack":       {"height": 0.45, "width": 0.30, "reliability": 0.55},
    "umbrella":       {"height": 0.90, "width": 1.00, "reliability": 0.45},
    "handbag":        {"height": 0.30, "width": 0.35, "reliability": 0.45},
    "tie":            {"height": 0.50, "width": 0.08, "reliability": 0.55},
    "suitcase":       {"height": 0.60, "width": 0.40, "reliability": 0.65},
    "frisbee":        {"height": 0.03, "width": 0.27, "reliability": 0.75},
    "skis":           {"height": 0.10, "width": 1.70, "reliability": 0.70},
    "snowboard":      {"height": 0.10, "width": 1.50, "reliability": 0.75},
    "sports ball":    {"height": 0.22, "width": 0.22, "reliability": 0.45},
    "kite":           {"height": 0.60, "width": 0.60, "reliability": 0.35},
    "baseball bat":   {"height": 0.07, "width": 0.70, "reliability": 0.80},
    "baseball glove": {"height": 0.25, "width": 0.20, "reliability": 0.65},
    "skateboard":     {"height": 0.13, "width": 0.80, "reliability": 0.80},
    "surfboard":      {"height": 0.05, "width": 1.80, "reliability": 0.55},
    "tennis racket":  {"height": 0.68, "width": 0.27, "reliability": 0.80},
    "bottle":         {"height": 0.25, "width": 0.08, "reliability": 0.75},
    "wine glass":     {"height": 0.20, "width": 0.08, "reliability": 0.75},
    "cup":            {"height": 0.10, "width": 0.08, "reliability": 0.70},
    "fork":           {"height": 0.02, "width": 0.19, "reliability": 0.65},
    "knife":          {"height": 0.02, "width": 0.20, "reliability": 0.60},
    "spoon":          {"height": 0.02, "width": 0.17, "reliability": 0.65},
    "bowl":           {"height": 0.08, "width": 0.15, "reliability": 0.65},
    "banana":         {"height": 0.03, "width": 0.18, "reliability": 0.60},
    "apple":          {"height": 0.08, "width": 0.08, "reliability": 0.75},
    "sandwich":       {"height": 0.05, "width": 0.12, "reliability": 0.45},
    "orange":         {"height": 0.08, "width": 0.08, "reliability": 0.75},
    "broccoli":       {"height": 0.12, "width": 0.10, "reliability": 0.45},
    "carrot":         {"height": 0.03, "width": 0.18, "reliability": 0.55},
    "hot dog":        {"height": 0.03, "width": 0.15, "reliability": 0.55},
    "pizza":          {"height": 0.02, "width": 0.30, "reliability": 0.55},
    "donut":          {"height": 0.04, "width": 0.10, "reliability": 0.70},
    "cake":           {"height": 0.12, "width": 0.25, "reliability": 0.40},
    "chair":          {"height": 0.90, "width": 0.50, "reliability": 0.70},
    "couch":          {"height": 0.90, "width": 1.80, "reliability": 0.55},
    "potted plant":   {"height": 0.60, "width": 0.35, "reliability": 0.35},
    "bed":            {"height": 0.60, "width": 1.50, "reliability": 0.55},
    "dining table":   {"height": 0.75, "width": 1.20, "reliability": 0.55},
    "toilet":         {"height": 0.40, "width": 0.37, "reliability": 0.85},
    "tv":             {"height": 0.50, "width": 1.00, "reliability": 0.60},
    "laptop":         {"height": 0.02, "width": 0.35, "reliability": 0.85},
    "mouse":          {"height": 0.04, "width": 0.06, "reliability": 0.75},
    "remote":         {"height": 0.03, "width": 0.05, "reliability": 0.65},
    "keyboard":       {"height": 0.03, "width": 0.45, "reliability": 0.80},
    "cell phone":     {"height": 0.15, "width": 0.07, "reliability": 0.85},
    "microwave":      {"height": 0.30, "width": 0.50, "reliability": 0.80},
    "oven":           {"height": 0.90, "width": 0.60, "reliability": 0.75},
    "toaster":        {"height": 0.20, "width": 0.30, "reliability": 0.70},
    "sink":           {"height": 0.20, "width": 0.50, "reliability": 0.60},
    "refrigerator":   {"height": 1.70, "width": 0.70, "reliability": 0.80},
    "book":           {"height": 0.02, "width": 0.15, "reliability": 0.40},
    "clock":          {"height": 0.30, "width": 0.30, "reliability": 0.55},
    "vase":           {"height": 0.30, "width": 0.15, "reliability": 0.45},
    "scissors":       {"height": 0.02, "width": 0.15, "reliability": 0.55},
    "teddy bear":     {"height": 0.30, "width": 0.20, "reliability": 0.45},
    "hair drier":     {"height": 0.22, "width": 0.10, "reliability": 0.70},
    "toothbrush":     {"height": 0.02, "width": 0.18, "reliability": 0.55},
}

# Same as thermal_cam.py's auto_calibrate_camera default (no per-user calibration
# is possible for one-off web uploads, so we fall back to the same generic value).
FOCAL_LENGTH_PX = 550.0
CONF_THRESHOLD = 0.30


def estimate_distance_hybrid(bbox_width: int, bbox_height: int, object_name: str):
    dims = OBJECT_DIMENSIONS.get(object_name)
    if dims is None or bbox_width <= 0 or bbox_height <= 0:
        return None

    real_width = dims["width"]
    real_height = dims["height"]

    dist_width = (real_width * FOCAL_LENGTH_PX) / bbox_width
    dist_height = (real_height * FOCAL_LENGTH_PX) / bbox_height

    aspect = bbox_width / bbox_height
    expected_aspect = real_width / real_height
    if aspect > expected_aspect * 1.6:
        return 0.35 * dist_height + 0.65 * dist_width
    return 0.6 * dist_height + 0.4 * dist_width


# ---------------------------------------------------------------------------
# YOLO model (loaded once at process start)
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "yolo11n.pt"  # small/fast model, appropriate for a CPU web dyno
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[INFO] Loading {MODEL_NAME} on {DEVICE}...")
        _model = YOLO(MODEL_NAME)
        _model.to(DEVICE)
    return _model


def thermalize_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("No image data provided.")

    if image.ndim == 2:
        gray = image
    else:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thermal_base = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    edges = cv2.Canny(gray, 40, 120)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    thermal = cv2.addWeighted(thermal_base, 0.85, edges_bgr, 0.15, 0)
    return thermal


def detect_and_annotate(thermal: np.ndarray, image: np.ndarray):
    """Run YOLO detection on the original image, draw boxes/labels on the
    thermal-rendered image, and return (annotated_image, detections_list)."""
    model = get_model()
    results = model.predict(image, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

        for i in range(len(cls_ids)):
            xmin, ymin, xmax, ymax = xyxy[i]
            confidence = float(confs[i])
            object_name = model.names[int(cls_ids[i])]

            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= 0 or box_height <= 0:
                continue

            distance = estimate_distance_hybrid(box_width, box_height, object_name)

            if distance is None:
                box_color = (160, 160, 160)  # gray: unknown distance class
                text_bg = (90, 90, 90)
            elif distance < 4.0:
                box_color = (0, 0, 255)      # red: close
                text_bg = (0, 0, 200)
            elif distance < 12.0:
                box_color = (0, 165, 255)    # orange: mid-range
                text_bg = (0, 120, 200)
            else:
                box_color = (0, 255, 0)      # green: far
                text_bg = (0, 180, 0)

            cv2.rectangle(thermal, (xmin, ymin), (xmax, ymax), box_color, 2)

            if distance is not None:
                label_text = f"{object_name.upper()} {distance:.2f}m ({confidence:.2f})"
            else:
                label_text = f"{object_name.upper()} ({confidence:.2f})"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(ymin - 20, 0)
            cv2.rectangle(thermal, (xmin, ty), (xmin + tw + 10, ty + th + 8), text_bg, -1)
            cv2.putText(thermal, label_text, (xmin + 5, ty + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "label": object_name,
                "confidence": confidence,
                "distance": distance,
            })

    return thermal, detections


def encode_png(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image.")
    return b64encode(buffer.tobytes()).decode("ascii")


@app.get("/")
def index():
    return render_template_string(INDEX_HTML, image_data=None, detections=None, error=None)


@app.post("/process")
def process():
    uploaded = request.files.get("image")
    if not uploaded or uploaded.filename == "":
        return render_template_string(
            INDEX_HTML, image_data=None, detections=None,
            error="No file was uploaded. Please choose an image first."
        )

    try:
        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        if raw.size == 0:
            raise ValueError("The uploaded file is empty.")

        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "Could not read that file as an image. Please upload a JPG, PNG, or WEBP."
            )

        thermal = thermalize_image(image)
        thermal, detections = detect_and_annotate(thermal, image)
        image_data = encode_png(thermal)
        return render_template_string(
            INDEX_HTML, image_data=image_data, detections=detections, error=None
        )
    except Exception as exc:
        # Any decode/inference failure now surfaces as a friendly banner
        # instead of crashing the request with a raw 500 error.
        return render_template_string(
            INDEX_HTML, image_data=None, detections=None, error=str(exc)
        ), 400


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Warm up the model at startup in dev mode so the first request isn't slow.
    get_model()
    # use_reloader=False avoids loading the YOLO model twice (once in the
    # main process, once in Flask's debug auto-reloader subprocess).
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)