# Thermal OpenCV

This repo now includes two modes:

- `app.py`: a Flask app for Render that applies a thermal effect to uploaded images
- `thermal_cam.py`: the local webcam-based YOLO tracker

## Render deployment

Set the start command to:

```bash
gunicorn app:app
```

Or keep the included `Procfile` and let Render detect it.

## Local web app

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Local webcam tracker

```powershell
pip install -r requirements-local.txt
python thermal_cam.py
```

The first run of the tracker may download `yolo11n.pt`.
