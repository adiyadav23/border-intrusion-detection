# 🚨 AI-Based Border Intrusion Detection System

A Python desktop application that detects unauthorized vehicle intrusions in surveillance video using **YOLOv8 Nano** and **OpenCV**. When a vehicle crosses a virtual boundary line, the system captures evidence, logs the event to an **SQLite** database, and lets you review the history from a login-protected GUI.

---

## ✨ Features

- **Vehicle detection** with YOLOv8n (cars, buses, trucks)
- **Virtual boundary crossing detection**: flags vehicles whose bounding box passes a configurable Y-line
- **Colour filter**: ignores green vehicles (e.g. friendly/authorized) using HSV masking
- **Automatic evidence capture**: saves an annotated frame with a 2-second cooldown to avoid duplicates
- **Event logging** to SQLite (timestamp, image path, event type)
- **Desktop GUI** (CustomTkinter) with login/registration, live video view and a detection history table (double-click a row to open the evidence image)
- **CPU-friendly**: frame skipping and downscaled input for real-time-ish performance without a GPU
- **Lightweight CLI version** (`app.py`) for quick testing with a video file or webcam

## 📊 Results

| Metric | Value |
|---|---|
| Vehicle detection precision | **92.0%** |
| Recall | **92.8%** |
| F1-score | **92.4%** |
| Virtual-boundary crossing accuracy | **93.2%** |
| Colour-filter accuracy | **[add your figure]%** |
| Evidence capture / DB logging | 100% success (500 test records) |
| Unit tests | 50 / 50 passing |
| Speed | **12.4 FPS** on 720p video, CPU only |

Evaluated on 8 test videos covering multiple resolutions, night-time and multi-vehicle scenarios.

## 🧰 Tech Stack

Python · YOLOv8 Nano (Ultralytics) · OpenCV · NumPy · SQLite · CustomTkinter · Pillow

## 🧠 How It Works

```
Video frame ─► Resize ─► YOLOv8n (car/bus/truck) ─► Confidence filter
          ─► Green-colour filter (HSV) ─► Crosses BORDER_Y?
          ─► Draw alert ─► Save evidence image ─► Log to SQLite ─► Show in GUI
```

1. Frames are read with OpenCV, skipping frames (`FRAME_SKIP`) for speed.
2. YOLOv8n detects vehicles above the confidence threshold.
3. Detections whose box is mostly green (HSV mask > 30%) are ignored.
4. If the bottom edge of a box (`y2`) goes past `BORDER_Y`, it is an intrusion.
5. An annotated frame is saved to `evidence/` and a row is inserted into `detections.db`.

## 📁 Project Structure

```
.
├── main.py            # Full GUI application (login + detection + history)
├── app.py             # Lightweight OpenCV-window version (no login/DB)
├── requirements.txt
├── evidence/          # Saved intrusion screenshots (auto-created)
├── detections.db      # SQLite DB (auto-created)
└── video.mp4          # Your sample video (not included in repo)
```

## 🚀 Installation

```bash
git clone https://github.com/adiyadav23/<repo-name>.git
cd <repo-name>

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

The YOLOv8n weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on first run.

## ▶️ Usage

**GUI version**

```bash
python main.py
```

1. Register a user, then log in.
2. Click **Upload Video** and choose an `.mp4`, `.avi` or `.mov` file.
3. Click **Start Detection**.
4. Click **History** to see logged intrusions; double-click a row to open its evidence image.

**CLI version**

```bash
python app.py       # uses VIDEO_SOURCE in app.py (set to 0 for webcam); press q to quit
```

## ⚙️ Configuration

Edit the constants at the top of `main.py` / `app.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolov8n.pt` | YOLO weights |
| `CONFIDENCE` | `0.5` | Minimum detection confidence |
| `BORDER_Y` | `200` (GUI) / `300` (CLI) | Y-pixel of the virtual boundary |
| `FRAME_SKIP` | `4` / `3` | Process every Nth frame (higher = faster) |
| `SAVE_DIR` | `evidence` | Folder for saved evidence |

> Set `BORDER_Y` to match your camera view and resized frame height.

## ⚠️ Known Limitations

- False positives from **shadows** and false negatives from **occlusion** and **small, distant vehicles**
- The green-colour filter can misfire on vehicles near vegetation
- `os.startfile` (open evidence image) is Windows-only

## 🔮 Future Improvements

- Object tracking (ByteTrack/DeepSORT) to count unique vehicles and use direction of crossing
- Polygon-shaped restricted zones instead of a single horizontal line
- Larger model (YOLOv8s/m) or fine-tuning on night-time data
- Salted password hashing (bcrypt/PBKDF2) and cross-platform image opening
- Email/SMS alerts and live RTSP camera support

## 📄 License

Released under the [MIT License](LICENSE).

## 👤 Author

**Aditya Yadav**: [GitHub](https://github.com/adiyadav23) · [LinkedIn](https://www.linkedin.com/in/aditya-yadav-187b8528a)
