import cv2
import numpy as np
import torch
import time
import threading
from collections import deque
from ultralytics import YOLO

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.lock  = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame is not None, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        self._thread.join(timeout=2)
        self.cap.release()


def run_local_yolo_thermal_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Booting PyTorch Backend on Device: {device.upper()}")

    model = YOLO("yolo11n.pt")
    model.to(device)
    if device == "cuda":
        model.half()

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, verbose=False)

    cap = CameraStream(src=0, width=640, height=480)
    if not cap.cap.isOpened():
        print("[ERROR] Local camera stream could not be accessed.")
        return

    FOCAL_LENGTH       = 525.0
    DEFAULT_REAL_WIDTH = 0.5
    REAL_WIDTHS = {
        "person":     0.5,
        "car":        1.8,
        "motorcycle": 0.8,
        "bicycle":    0.6,
        "truck":      2.5,
        "dog":        0.4,
        "cat":        0.25,
    }

    BUFFER_SIZE      = 5
    distance_history = {}

    _lut = np.arange(256, dtype=np.uint8).reshape(256, 1)
    JET_LUT = cv2.applyColorMap(_lut, cv2.COLORMAP_JET)

    ALPHA, BETA = 0.85, 0.15

    prev_time = time.perf_counter()
    fps        = 0.0

    INFER_EVERY = 2
    frame_idx   = 0
    last_results = []

    print("[STATUS] Tracking Pipeline running. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        now       = time.perf_counter()
        dt        = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_base = JET_LUT[gray].squeeze(axis=2)
        edges        = cv2.Canny(gray, 40, 120)
        edges_bgr    = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        thermal_sim  = np.empty_like(thermal_base)
        cv2.addWeighted(thermal_base, ALPHA, edges_bgr, BETA, 0, thermal_sim)

        frame_idx += 1
        if frame_idx % INFER_EVERY == 0:
            last_results = model.track(
                frame,
                device=device,
                imgsz=640,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.30,
                half=(device == "cuda"),
            )

        active_ids_this_frame = set()

        for result in last_results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            xyxy       = boxes.xyxy.cpu().numpy().astype(int)
            confs      = boxes.conf.cpu().numpy().astype(float)
            cls_ids    = boxes.cls.cpu().numpy().astype(int)
            track_ids  = boxes.id.cpu().numpy().astype(int)

            for i in range(len(track_ids)):
                xmin, ymin, xmax, ymax = xyxy[i]
                confidence  = confs[i]
                object_name = model.names[cls_ids[i]]
                obj_id      = track_ids[i]

                active_ids_this_frame.add(obj_id)

                if confidence <= 0.30:
                    continue

                box_width_pixels = xmax - xmin
                if box_width_pixels <= 0:
                    continue

                real_width  = REAL_WIDTHS.get(object_name, DEFAULT_REAL_WIDTH)
                raw_distance = (real_width * FOCAL_LENGTH) / box_width_pixels

                if obj_id not in distance_history:
                    distance_history[obj_id] = deque(maxlen=BUFFER_SIZE)
                distance_history[obj_id].append(raw_distance)
                distance = float(np.mean(distance_history[obj_id]))

                if distance < 4.0:
                    box_color = (0,   0, 255)
                    text_bg   = (0,   0, 255)
                elif distance < 12.0:
                    box_color = (0, 255, 255)
                    text_bg   = (0, 165, 255)
                else:
                    box_color = (0, 255,   0)
                    text_bg   = (0, 128,   0)

                cv2.rectangle(thermal_sim, (xmin, ymin), (xmax, ymax), box_color, 2)

                text_label = f"ID:{obj_id} {object_name.upper()} {distance:.1f}m"
                (tw, th), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                ty = max(ymin - 20, 0)
                cv2.rectangle(thermal_sim, (xmin, ty), (xmin + tw + 10, ty + th + 8), text_bg, -1)
                cv2.putText(thermal_sim, text_label, (xmin + 5, ty + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for old_id in list(distance_history.keys()):
            if old_id not in active_ids_this_frame:
                del distance_history[old_id]

        fps_text = f"FPS: {fps:.1f}"
        cv2.rectangle(thermal_sim, (10, 10), (140, 45), (0, 0, 0), -1)
        cv2.putText(thermal_sim, fps_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Live Local YOLO Thermal Vision Matrix", thermal_sim)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_local_yolo_thermal_pipeline()