import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from base64 import b64encode
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template_string, request
from ultralytics import YOLO

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Thermal OpenCV</title>
    <style>
      :root {
        color-scheme: dark;
        --bg: #0b1020;
        --panel: #121a2e;
        --border: #25304b;
        --text: #e8edf7;
        --muted: #9fb0d0;
        --accent: #67d1ff;
        --accent-2: #ff9f43;
      }
      body {
        margin: 0;
        min-height: 100vh;
        background: linear-gradient(160deg, #08101d 0%, #10172b 55%, #1b1020 100%);
        color: var(--text);
        font-family: Arial, sans-serif;
      }
      .wrap {
        max-width: 960px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }
      .hero {
        padding: 24px 0 20px;
      }
      h1 {
        margin: 0 0 10px;
        font-size: 2rem;
        line-height: 1.1;
      }
      p {
        margin: 0 0 16px;
        color: var(--muted);
      }
      .panel {
        background: rgba(18, 26, 46, 0.92);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
      }
      .grid {
        display: grid;
        gap: 20px;
        grid-template-columns: 1fr;
      }
      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
      }
      input[type="file"] {
        color: var(--muted);
      }
      button {
        appearance: none;
        border: 0;
        border-radius: 8px;
        padding: 10px 16px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #04101b;
        font-weight: 700;
        cursor: pointer;
      }
      .note {
        font-size: 0.95rem;
        color: var(--muted);
      }
      .image-box {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        background: #050913;
      }
      .image-box img {
        display: block;
        width: 100%;
        height: auto;
      }
      .label {
        padding: 10px 14px;
        font-size: 0.9rem;
        color: var(--muted);
        border-bottom: 1px solid var(--border);
      }
      .detections {
        padding: 10px 14px;
        font-size: 0.85rem;
        color: var(--muted);
        border-top: 1px solid var(--border);
      }
      .detections ul {
        margin: 6px 0 0;
        padding-left: 18px;
      }
      @media (min-width: 840px) {
        .grid {
          grid-template-columns: 360px 1fr;
          align-items: start;
        }
      }
    </style>
  </head>
  <body>
    <main class="wrap">
      <section class="hero">
        <h1>Thermal OpenCV</h1>
        <p>Upload an image and Render will return a thermal-style OpenCV preview with YOLO object detection and estimated distance overlays.</p>
      </section>

      <section class="grid">
        <form class="panel" method="post" action="/process" enctype="multipart/form-data">
          <div class="controls">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Render thermal</button>
          </div>
          <p class="note">For live webcam tracking, run <code>thermal_cam.py</code> locally on your machine instead.</p>
        </form>

        <div class="panel">
          {% if image_data %}
            <div class="label">Thermal output</div>
            <div class="image-box">
              <img src="data:image/png;base64,{{ image_data }}" alt="Thermal processed image with detections">
            </div>
            <div class="detections">
              {% if detections %}
                Detected {{ detections|length }} object(s):
                <ul>
                {% for d in detections %}
                  <li>{{ d.label }} &mdash; {{ '%.2f'|format(d.distance) + 'm' if d.distance is not none else 'distance n/a' }} (conf {{ '%.2f'|format(d.confidence) }})</li>
                {% endfor %}
                </ul>
              {% else %}
                No objects detected above the confidence threshold.
              {% endif %}
            </div>
          {% else %}
            <p class="note">No image processed yet.</p>
          {% endif %}
        </div>
      </section>
    </main>
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

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

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

            if confidence <= CONF_THRESHOLD:
                continue

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
    return render_template_string(INDEX_HTML, image_data=None, detections=None)


@app.post("/process")
def process():
    uploaded = request.files.get("image")
    if not uploaded or uploaded.filename == "":
        return render_template_string(INDEX_HTML, image_data=None, detections=None)

    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    thermal = thermalize_image(image)
    thermal, detections = detect_and_annotate(thermal, image)
    image_data = encode_png(thermal)
    return render_template_string(INDEX_HTML, image_data=image_data, detections=detections)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Warm up the model at startup in dev mode so the first request isn't slow.
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=True)