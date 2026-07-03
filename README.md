# Thermal Camera YOLO Tracker

Live webcam object tracking with a thermal-style OpenCV visualization, YOLO object detection, ByteTrack tracking, and approximate distance labels.

## Requirements

- Python 3.10 or newer
- A local webcam
- Windows, macOS, or Linux with camera access
- Optional: NVIDIA GPU with CUDA for faster inference

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```powershell
python thermal_cam.py
```

Press `q` in the video window to exit.

The first run may download the YOLO model weights file `yolo11n.pt`.

## Notes

- The distance estimate is approximate and depends on the assumed real-world width of each object class.
- If your webcam does not open, check that no other application is using it and that camera permissions are enabled.
