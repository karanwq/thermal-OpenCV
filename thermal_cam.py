import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
import numpy as np
import torch
import time
import threading
import json
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from ultralytics import YOLO
from typing import Dict, Tuple, Optional, List
import sys

@dataclass
class CameraCalibration:
    focal_length: float
    principal_point_x: float
    principal_point_y: float
    image_width: int
    image_height: int
    distortion_coeffs: np.ndarray = None
    
    def __post_init__(self):
        if self.distortion_coeffs is None:
            self.distortion_coeffs = np.zeros(5, dtype=np.float32)
    
    def to_dict(self):
        data = asdict(self)
        if isinstance(data['distortion_coeffs'], np.ndarray):
            data['distortion_coeffs'] = data['distortion_coeffs'].tolist()
        return data
    
    @classmethod
    def from_dict(cls, data):
        data_copy = data.copy()
        if 'distortion_coeffs' in data_copy:
            data_copy['distortion_coeffs'] = np.array(data_copy['distortion_coeffs'], dtype=np.float32)
        return cls(**data_copy)


def _candidate_backends():
    """Ordered list of backends to try for the current OS. DSHOW is fast but
    some webcam drivers (especially thermal/IR USB cams) crash or fail under
    it on Windows - MSMF is the more modern fallback. CAP_ANY is always last
    as a final catch-all."""
    if sys.platform.startswith("win"):
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]


class CameraStream:
    WARMUP_SECONDS = 3.0
    RECONNECT_AFTER = 2.0

    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.cap = self._open(src)

        if self.cap is None:
            print("[ERROR] Could not find or open any available camera index (0, 1, 2).")
            print("        Common causes: another app (Zoom/Teams/Camera app) is holding")
            print("        the camera, Windows camera privacy permissions are off, or the")
            print("        wrong index/backend for this device.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self, preferred_idx):
        camera_indices = list(dict.fromkeys([preferred_idx, 0, 1, 2]))
        for test_idx in camera_indices:
            for backend in _candidate_backends():
                print(f"[INFO] Attempting to open camera index {test_idx} "
                      f"(backend={backend})...")
                cap = cv2.VideoCapture(test_idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"[SUCCESS] Connected to camera {test_idx} at "
                          f"{actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
                    return cap
                cap.release()
        return None

    def _reader(self):
        last_good_read = time.perf_counter()
        opened_at = time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if ok:
                last_good_read = now
                with self.lock:
                    self.frame = frame
            else:
                still_warming_up = (now - opened_at) < self.WARMUP_SECONDS
                if not still_warming_up and (now - last_good_read) > self.RECONNECT_AFTER:
                    print("[WARN] Camera stream lost, attempting to reconnect...")
                    self.cap.release()
                    new_cap = self._open(self.src)
                    if new_cap is not None:
                        self.cap = new_cap
                        last_good_read = time.perf_counter()
                        opened_at = last_good_read
                    else:
                        print("[ERROR] Reconnect failed, giving up.")
                        self.running = False
                        break
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame is not None, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2)
        if self.cap and self.cap.isOpened():
            self.cap.release()


class DistanceEstimator:
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
    
    def __init__(self, calibration: CameraCalibration, buffer_size=3):
        self.calibration = calibration
        self.distance_history: Dict[int, deque] = {}
        self.buffer_size = buffer_size
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}
        self._kf_initialized: Dict[int, bool] = {}
        self.focal = calibration.focal_length
        
    def _init_kalman_filter(self, obj_id: int):

        kf = cv2.KalmanFilter(2, 1)
        kf.transitionMatrix = np.array([[1.0, 1.0],
                                         [0.0, 1.0]], dtype=np.float32)
        kf.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
        kf.processNoiseCov = np.array([[1e-3, 0.0],
                                        [0.0, 1e-2]], dtype=np.float32)
        kf.measurementNoiseCov = np.array([[0.05]], dtype=np.float32)
        kf.errorCovPost = np.eye(2, dtype=np.float32)
        kf.statePost = np.zeros((2, 1), dtype=np.float32)
        self.kalman_filters[obj_id] = kf
        self._kf_initialized[obj_id] = False

    def get_class_reliability(self, object_name: str) -> float:
        dims = self.OBJECT_DIMENSIONS.get(object_name)
        return dims["reliability"] if dims else 0.3
        
    def estimate_distance_hybrid(self, bbox_width: int, bbox_height: int, object_name: str) -> Optional[float]:
        dims = self.OBJECT_DIMENSIONS.get(object_name)
        if dims is None:
            return None
        real_width = dims["width"]
        real_height = dims["height"]
        
        if bbox_width <= 0 or bbox_height <= 0:
            return None
        
        dist_width = (real_width * self.focal) / bbox_width
        dist_height = (real_height * self.focal) / bbox_height

        aspect = bbox_width / bbox_height
        expected_aspect = real_width / real_height
        if aspect > expected_aspect * 1.6:
            return 0.35 * dist_height + 0.65 * dist_width
        return 0.6 * dist_height + 0.4 * dist_width
    
    def filter_distance(self, obj_id: int, raw_distance: float) -> float:
        if obj_id not in self.kalman_filters:
            self._init_kalman_filter(obj_id)
        
        kf = self.kalman_filters[obj_id]


        if not self._kf_initialized[obj_id]:
            kf.statePost = np.array([[raw_distance], [0.0]], dtype=np.float32)
            self._kf_initialized[obj_id] = True

        kf.predict()
        corrected = kf.correct(np.array([[raw_distance]], dtype=np.float32))
        return float(corrected[0][0])
    
    def smooth_distance(self, obj_id: int, raw_distance: float) -> float:
        if obj_id not in self.distance_history:
            self.distance_history[obj_id] = deque(maxlen=self.buffer_size)
        self.distance_history[obj_id].append(raw_distance)
        return self.filter_distance(obj_id, raw_distance)
    
    def get_distance_confidence(self, distance: float, reliability: float = 1.0) -> float:
        if 1.0 <= distance <= 20.0:
            range_conf = 1.0
        elif 0.5 <= distance < 1.0 or 20.0 < distance <= 50.0:
            range_conf = 0.8
        else:
            range_conf = 0.5

        return range_conf * reliability
    
    def cleanup_object(self, obj_id: int):
        self.distance_history.pop(obj_id, None)
        self.kalman_filters.pop(obj_id, None)
        self._kf_initialized.pop(obj_id, None)


class DistanceAlert:
    def __init__(self, close_threshold=3.0, far_threshold=20.0):
        self.close_threshold = close_threshold
        self.far_threshold = far_threshold
        self.alert_cooldown = {}
    
    def check(self, obj_id: int, distance: float, object_name: str, confidence: float) -> Optional[str]:
        if confidence < 0.8:
            return None
        
        current_time = time.time()
        if obj_id in self.alert_cooldown:
            if current_time - self.alert_cooldown[obj_id] < 1.0:
                return None
        
        if distance < self.close_threshold and object_name == "person":
            self.alert_cooldown[obj_id] = current_time
            return 'close'
        
        if distance > self.far_threshold:
            return 'far'
        
        return 'ok'


class CalibrationHelper:
    def __init__(self, camera_index=0, width=640, height=480):
        self.cap = None
        camera_indices = list(dict.fromkeys([camera_index, 0, 1, 2]))
        for test_idx in camera_indices:
            for backend in _candidate_backends():
                cap = cv2.VideoCapture(test_idx, backend)
                if cap.isOpened():
                    self.cap = cap
                    break
                cap.release()
            if self.cap is not None:
                break

        if self.cap is None:
            raise RuntimeError(
                "Could not open any camera (tried indices 0, 1, 2 with all "
                "available backends). Check that no other app is using the "
                "camera and that camera permissions are granted."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        
        self.measurements = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
    
    def measure_single_object(self) -> bool:
        print(f"\n{'='*60}\nMEASUREMENT {len(self.measurements) + 1}\n{'='*60}")
        
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            return False
        
        window_name = "Draw bbox (click-drag-release), SPACE=OK, ESC=Retry"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw_rectangle)
        
        print("[1] Draw bounding box around object")
        
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        while True:
            display = frame.copy()
            
            if self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.putText(display, "SPACE=OK  ESC=Retry", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and self.start_point and self.end_point:
                break
            elif key == 27:
                self.start_point = None
                self.end_point = None
        
        cv2.destroyWindow(window_name)
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        print(f"[2] Bbox: {bbox_width}×{bbox_height} px")
        print("[3] Enter REAL-WORLD dimensions:")
        
        try:
            real_height = float(input("    Height (m): "))
            real_width = float(input("    Width (m): "))
            measured_distance = float(input("    Distance (m): "))
            
            if real_height <= 0 or real_width <= 0 or measured_distance <= 0:
                raise ValueError("All values must be positive")
        except ValueError as e:
            print(f"[ERROR] {e}")
            return False
        
        focal_length_h = (real_height * bbox_height) / measured_distance
        focal_length_w = (real_width * bbox_width) / measured_distance
        focal_length_avg = (focal_length_h + focal_length_w) / 2
        
        print(f"[4] Focal lengths: H={focal_length_h:.1f}px, W={focal_length_w:.1f}px, Avg={focal_length_avg:.1f}px")
        
        self.measurements.append({
            "distance_m": measured_distance,
            "real_height_m": real_height,
            "real_width_m": real_width,
            "bbox_height_px": bbox_height,
            "bbox_width_px": bbox_width,
            "focal_length_avg": focal_length_avg,
        })
        return True
    
    def run_interactive(self):
        print(f"\n{'='*60}\nINTERACTIVE CAMERA CALIBRATION\n{'='*60}")
        print("\nCalibrate at 5-6 different distances")
        
        while True:
            success = self.measure_single_object()
            if not success:
                continue
            
            print("\nOptions: [C]alibrate more  [S]ummary  [E]xit")
            choice = input("Choice: ").upper()
            
            if choice == 'S':
                self.show_summary()
            elif choice == 'E':
                break
        
        if self.measurements:
            return self.finalize()
        return None
    
    def show_summary(self):
        if not self.measurements:
            print("\n[WARNING] No measurements yet")
            return
        
        print(f"\n{'='*60}\nCALIBRATION SUMMARY\n{'='*60}")
        
        focal_lengths = []
        for i, m in enumerate(self.measurements, 1):
            fl = m["focal_length_avg"]
            focal_lengths.append(fl)
            print(f"[{i}] {m['distance_m']:.1f}m → {fl:.1f}px")
        
        focal_mean = np.mean(focal_lengths)
        focal_std = np.std(focal_lengths)
        
        print(f"\n{'-'*60}")
        print(f"Mean: {focal_mean:.1f} ± {focal_std:.1f}px")
        print(f"Variation: {(focal_std/focal_mean)*100:.1f}%")
        
        if focal_std / focal_mean > 0.15:
            print("[WARNING] High variation")
        else:
            print(f"[OK] Recommended: focal_length={focal_mean:.1f}")
    
    def finalize(self) -> CameraCalibration:
        focal_lengths = [m["focal_length_avg"] for m in self.measurements]
        focal_mean = np.mean(focal_lengths)
        
        return CameraCalibration(
            focal_length=focal_mean,
            principal_point_x=self.width / 2,
            principal_point_y=self.height / 2,
            image_width=self.width,
            image_height=self.height,
        )
    
    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def ensure_model_downloaded(model_name="yolo11n.pt"):
    possible_paths = [
        model_name,
        Path.home() / ".yolo" / "weights" / model_name,
        Path.home() / ".cache" / "ultralytics" / "weights" / model_name,
        Path.cwd() / model_name,
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"[INFO] Model found: {path}")
            return str(path)
    
    download_dir = Path.home() / ".yolo" / "weights"
    download_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_dir / model_name
    
    print(f"[INFO] Downloading model...")
    
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_name)
        
        if Path(model_name).exists():
            import shutil
            shutil.move(model_name, str(model_path))
            print(f"[INFO] Model downloaded")
            return str(model_path)
        else:
            print(f"[ERROR] Download failed")
            return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def auto_calibrate_camera(width: int = 640, height: int = 480) -> CameraCalibration:
    return CameraCalibration(
        focal_length=550.0,
        principal_point_x=width / 2.0,
        principal_point_y=height / 2.0,
        image_width=width,
        image_height=height,
    )


def load_calibration(filepath: str) -> Optional[CameraCalibration]:
    try:
        with open(filepath) as f:
            data = json.load(f)
        return CameraCalibration.from_dict(data)
    except FileNotFoundError:
        print(f"[WARNING] Calibration not found: {filepath}")
        return None


def save_calibration(calibration: CameraCalibration, filepath: str = "camera_calibration.json"):
    with open(filepath, 'w') as f:
        json.dump(calibration.to_dict(), f, indent=2)
    print(f"[INFO] Calibration saved to {filepath}")


class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.smoothing_times = deque(maxlen=window_size)
        self.last_report = time.time()
    
    def log_times(self, frame_time, inference_time, smoothing_time):
        self.frame_times.append(frame_time)
        self.inference_times.append(inference_time)
        self.smoothing_times.append(smoothing_time)
    
    def should_report(self):
        return time.time() - self.last_report > 1.0
    
    def report(self):
        self.last_report = time.time()
        if not self.frame_times:
            return None
        
        return {
            "fps": 1.0 / np.mean(self.frame_times),
            "inference_ms": np.mean(self.inference_times) * 1000,
            "smoothing_ms": np.mean(self.smoothing_times) * 1000,
        }


def run_thermal_vision_pipeline(calibration_file: Optional[str] = None,
                                 model_name: Optional[str] = None,
                                 open_vocab_classes: Optional[List[str]] = None,
                                 custom_dims_file: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device.upper()}")

    if open_vocab_classes:

        if model_name is None:
            model_name = "yolov8s-worldv2.pt"
        print(f"[INFO] Open-vocabulary mode - prompted classes: {open_vocab_classes}")
    else:

        if model_name is None:
            model_name = "yolo11s.pt" if device == "cuda" else "yolo11n.pt"

    model_path = ensure_model_downloaded(model_name)
    if model_path is None:
        print("[ERROR] Model unavailable")
        return

    try:
        model = YOLO(model_path)
        model.to(device)
        if open_vocab_classes:
            model.set_classes(open_vocab_classes)
        if device == "cuda" and not open_vocab_classes:
            model.half()
        print(f"[INFO] YOLO loaded ({model_name})")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=640, verbose=False)
    except Exception as e:
        print(f"[WARN] Model warm-up pass failed (non-fatal, continuing): {e}")
    if custom_dims_file:
        try:
            with open(custom_dims_file) as f:
                custom_dims = json.load(f)
            DistanceEstimator.OBJECT_DIMENSIONS.update(custom_dims)
            print(f"[INFO] Loaded {len(custom_dims)} custom object dimensions "
                  f"from {custom_dims_file}")
        except FileNotFoundError:
            print(f"[WARN] Custom dimensions file not found: {custom_dims_file}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Could not parse custom dimensions file: {e}")

    tracked_class_ids = None
    unknown_classes = [name for name in model.names.values()
                        if name not in DistanceEstimator.OBJECT_DIMENSIONS]
    if unknown_classes:
        print(f"[WARN] No known size for these classes, distance will be "
              f"skipped for them: {unknown_classes}")
        if open_vocab_classes:
            print("       -> Add them via --custom-dims to get distance for these too.")
    else:
        print("[INFO] Distance available for all detected classes.")

    if calibration_file and Path(calibration_file).exists():
        calibration = load_calibration(calibration_file)
        print(f"[INFO] Loaded calibration")
    else:
        calibration = auto_calibrate_camera(640, 480)
        print(f"[INFO] Auto-calibration (focal={calibration.focal_length:.1f}px) "
              f"- run --mode calibrate for real accuracy")
    
    distance_estimator = DistanceEstimator(calibration, buffer_size=3)
    alert_system = DistanceAlert(close_threshold=3.0)
    perf_monitor = PerformanceMonitor()

    cap = CameraStream(src=0, width=640, height=480)
    if cap.cap is None or not cap.cap.isOpened() or not cap.running:
        print("[ERROR] Camera initialization loop failed.")
        return

    _lut = np.arange(256, dtype=np.uint8).reshape(256, 1)
    JET_LUT = cv2.applyColorMap(_lut, cv2.COLORMAP_JET)

    ALPHA, BETA = 0.85, 0.15

    INFER_EVERY = 1 if device == "cuda" else 2

    MIN_HITS_TO_CONFIRM = 3
    hit_counts: Dict[int, int] = {}
    
    prev_time = time.perf_counter()
    fps = 0.0
    frame_idx = 0
    last_results = []
    show_calibration_info = False

    print("\n[STATUS] Running. Press 'q' to exit, 'c' for calibration info, 's' for stats")
    print("="*70)

    frame_count = 0
    t_inference_total = 0
    t_smoothing_total = 0

    while True:
        t_frame_start = time.perf_counter()
        
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_base = JET_LUT[gray].squeeze(axis=2)
        edges = cv2.Canny(gray, 40, 120)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        thermal_sim = cv2.addWeighted(thermal_base, ALPHA, edges_bgr, BETA, 0)

        frame_idx += 1
        t_infer_start = time.perf_counter()
        
        if frame_idx % INFER_EVERY == 0:
            try:
                last_results = model.track(
                    frame,
                    device=device,
                    imgsz=640,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    conf=0.30,
                    classes=tracked_class_ids,
                    half=(device == "cuda" and not open_vocab_classes),
                )
            except Exception as e:
                print(f"[WARN] Inference failed on frame {frame_idx}: {e}")
                continue
        
        t_infer_end = time.perf_counter()
        t_inference_total += (t_infer_end - t_infer_start)

        active_ids = set()
        t_smooth_start = time.perf_counter()

        for result in last_results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
            track_ids = boxes.id.cpu().numpy().astype(np.int32)

            for i in range(len(track_ids)):
                xmin, ymin, xmax, ymax = xyxy[i]
                confidence = confs[i]
                object_name = model.names[cls_ids[i]]
                obj_id = track_ids[i]

                active_ids.add(obj_id)

                if confidence <= 0.30:
                    continue

                box_width = xmax - xmin
                box_height = ymax - ymin
                
                if box_width <= 0 or box_height <= 0:
                    continue


                hit_counts[obj_id] = hit_counts.get(obj_id, 0) + 1
                if hit_counts[obj_id] < MIN_HITS_TO_CONFIRM:
                    continue

                raw_distance = distance_estimator.estimate_distance_hybrid(
                    box_width, box_height, object_name
                )

                if raw_distance is None or raw_distance <= 0:

                    box_color = (160, 160, 160)
                    cv2.rectangle(thermal_sim, (xmin, ymin), (xmax, ymax), box_color, 2)
                    text_label = f"ID:{obj_id} {object_name.upper()}"
                    (tw, th), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    ty = max(ymin - 20, 0)
                    cv2.rectangle(thermal_sim, (xmin, ty), (xmin + tw + 10, ty + th + 8), (90, 90, 90), -1)
                    cv2.putText(thermal_sim, text_label, (xmin + 5, ty + th + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    continue

                distance = distance_estimator.smooth_distance(obj_id, raw_distance)
                reliability = distance_estimator.get_class_reliability(object_name)
                conf_metric = distance_estimator.get_distance_confidence(distance, reliability)

                alert_status = alert_system.check(obj_id, distance, object_name, conf_metric)

                if distance < 4.0:
                    box_color = (0, 0, 255)
                    text_bg = (0, 0, 200)
                elif distance < 12.0:
                    box_color = (0, 165, 255)
                    text_bg = (0, 120, 200)
                else:
                    box_color = (0, 255, 0)
                    text_bg = (0, 180, 0)

                cv2.rectangle(thermal_sim, (xmin, ymin), (xmax, ymax), box_color, 2)

                conf_marker = "✓" if conf_metric >= 0.9 else "◐" if conf_metric >= 0.7 else "○"
                text_label = f"ID:{obj_id} {object_name.upper()} {distance:.2f}m {conf_marker}"
                
                (tw, th), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                ty = max(ymin - 20, 0)
                cv2.rectangle(thermal_sim, (xmin, ty), (xmin + tw + 10, ty + th + 8), text_bg, -1)
                cv2.putText(thermal_sim, text_label, (xmin + 5, ty + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                if alert_status == 'close':
                    print(f"⚠️  ALERT: {object_name.upper()} (ID:{obj_id}) {distance:.2f}m")

        t_smooth_end = time.perf_counter()
        t_smoothing_total += (t_smooth_end - t_smooth_start)

        for old_id in list(distance_estimator.distance_history.keys()):
            if old_id not in active_ids:
                distance_estimator.cleanup_object(old_id)
        for old_id in list(hit_counts.keys()):
            if old_id not in active_ids:
                hit_counts.pop(old_id, None)

        info_h = 45 if not show_calibration_info else 90
        cv2.rectangle(thermal_sim, (10, 10), (450, info_h), (0, 0, 0), -1)
        
        fps_text = f"FPS: {fps:.1f} | Hybrid | q=exit c=cal s=stats"
        cv2.putText(thermal_sim, fps_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        if show_calibration_info:
            cal_text = f"Focal: {calibration.focal_length:.1f}px | {calibration.image_width}x{calibration.image_height}"
            cv2.putText(thermal_sim, cal_text, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Thermal Vision Matrix", thermal_sim)

        frame_count += 1
        t_frame_end = time.perf_counter()
        perf_monitor.log_times(t_frame_end - t_frame_start, t_infer_end - t_infer_start, 
                              t_smooth_end - t_smooth_start)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Exiting...")
            break
        elif key == ord('c'):
            show_calibration_info = not show_calibration_info
        elif key == ord('s'):
            if perf_monitor.should_report():
                stats = perf_monitor.report()
                if stats:
                    print(f"[STATS] FPS: {stats['fps']:.1f} | Inference: {stats['inference_ms']:.1f}ms | Smoothing: {stats['smoothing_ms']:.2f}ms")

    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        print(f"\n[FINAL] Avg inference: {(t_inference_total/frame_count)*1000:.1f}ms | Avg smoothing: {(t_smoothing_total/frame_count)*1000:.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Thermal Camera System - YOLO + Distance Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python thermal_cam_final.py --mode run
  python thermal_cam_final.py --mode run --calibration camera_calibration.json
  python thermal_cam_final.py --mode calibrate
  python thermal_cam_final.py --mode run --open-vocab-classes "drone,stroller,forklift"
  python thermal_cam_final.py --mode run --open-vocab-classes "drone,forklift" --custom-dims my_dims.json
  python thermal_cam_final.py --mode help
        """
    )
    
    parser.add_argument('--mode', choices=['run', 'calibrate', 'help'],
                       default='help',
                       help='Mode (default: help)')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to calibration JSON')
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO model to use (e.g. yolo11n.pt, yolo11s.pt, '
                            'yolo11m.pt). Bigger = more accurate but slower. '
                            'Default: yolo11s.pt on GPU, yolo11n.pt on CPU. '
                            'Ignored if --open-vocab-classes is set unless '
                            'you explicitly pick a yolov8*-world*.pt model.')
    parser.add_argument('--open-vocab-classes', type=str, default=None,
                       help='Comma-separated list of object names to detect '
                            'via open-vocabulary YOLO-World instead of the '
                            'fixed 80-class model, e.g. "drone,stroller,forklift". '
                            'Runs fully local, no fixed class list.')
    parser.add_argument('--custom-dims', type=str, default=None,
                       help='Path to a JSON file of real-world dimensions for '
                            'extra classes (needed for open-vocab prompts, '
                            'since those have no built-in known size). '
                            'Format: {"drone": {"height":0.3,"width":0.5,'
                            '"reliability":0.5}}')
    
    args = parser.parse_args()

    open_vocab_classes = None
    if args.open_vocab_classes:
        open_vocab_classes = [c.strip() for c in args.open_vocab_classes.split(',') if c.strip()]

    if args.mode == 'help':
        print("\n Thermal Cam")
    elif args.mode == 'run':
        try:
            run_thermal_vision_pipeline(args.calibration, args.model,
                                         open_vocab_classes, args.custom_dims)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'calibrate':
        calibrator = None
        try:
            calibrator = CalibrationHelper(camera_index=0, width=640, height=480)
            calibration = calibrator.run_interactive()
            
            if calibration:
                save_calibration(calibration, "camera_calibration.json")
                print("\n[SUCCESS] Calibration complete!")
                print("\nRun with:")
                print("  python thermal_cam_final.py --mode run --calibration camera_calibration.json")
            
            calibrator.cleanup()
        except KeyboardInterrupt:
            print("\n[INFO] Cancelled")
            if calibrator is not None:
                calibrator.cleanup()
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            if calibrator is not None:
                calibrator.cleanup()


if __name__ == "__main__":
    main()