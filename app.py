from base64 import b64encode

import cv2
import numpy as np
from flask import Flask, render_template_string, request

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
        <p>Upload an image and Render will return a thermal-style OpenCV preview. This is the deployment-friendly web version of the project.</p>
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
              <img src="data:image/png;base64,{{ image_data }}" alt="Thermal processed image">
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


def encode_png(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image.")
    return b64encode(buffer.tobytes()).decode("ascii")


@app.get("/")
def index():
    return render_template_string(INDEX_HTML, image_data=None)


@app.post("/process")
def process():
    uploaded = request.files.get("image")
    if not uploaded or uploaded.filename == "":
        return render_template_string(INDEX_HTML, image_data=None)

    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    thermal = thermalize_image(image)
    image_data = encode_png(thermal)
    return render_template_string(INDEX_HTML, image_data=image_data)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
