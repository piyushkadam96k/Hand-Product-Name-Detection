# Hand Product Name Detection (MediaPipe + OpenCV + Tesseract)

follow on instagram @piyush_kadam96k

Detects hands from a webcam feed, checks if a product is being held, and extracts a product name from the cropped hand region using OCR. Overlays detections and FPS on the video, logs events, and saves the output to a video file.

## ✨ Features
- Detect hands using MediaPipe 🖐️
- Heuristic product-in-hand check via contour area in the hand crop 📦
- OCR with Tesseract to extract a likely product name 🔤
- Optional camera inversion (none, horizontal, vertical, both) 🔁
- On-screen overlays with transparency and FPS 🎛️
- Logs to `gesture_recognition.log` 📝
- Saves processed video to `output.mp4` 💾

## 📦 Requirements
- Python 3.8+
- Packages: `opencv-python`, `mediapipe`, `numpy`, `pytesseract`
- Tesseract OCR installed on your system

Install Python dependencies:
```bash
pip install opencv-python mediapipe numpy pytesseract
```

Install Tesseract OCR:
- Windows (recommended builds): https://github.com/UB-Mannheim/tesseract/wiki

After installing Tesseract, ensure its executable path matches what `app.py` expects:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```
If Tesseract is installed elsewhere, update the path in `app.py` accordingly.

## ▶️ Usage
From the project directory, run:
```bash
python app.py --camera 0 --width 640 --height 480 --invert none
```

Arguments:
- `--camera` (int): Webcam index (default: `0`) 🎥
- `--width` (int): Frame width (default: `640`) ↔️
- `--height` (int): Frame height (default: `480`) ↕️
- `--invert` (str): One of `none`, `horizontal`, `vertical`, `both` (default: `none`) 🔁

🎮 Controls:
- Press `q` to quit.

📤 Outputs:
- Processed video: `output.mp4` 🎞️
- Logs: `gesture_recognition.log` 📝
- Display window titled: "Hand Product Name Detection" 🪟

## ⚙️ How it Works (High Level)
1. Captures frames from the selected camera.
2. Detects hands with MediaPipe and draws landmarks/boxes.
3. Crops around each detected hand, runs a contour-based heuristic to decide if a product might be present.
4. If a product is likely, runs Tesseract OCR over the hand crop and displays the first meaningful word as the product name.
5. Overlays results and FPS, writes frames to `output.mp4`.

## 🧰 Troubleshooting
- Webcam not opening: ensure the correct `--camera` index and that no other app is using the camera.
- Tesseract errors or empty text:
  - Verify Tesseract is installed and the path in `app.py` is correct.
  - Lighting/contrast strongly affects OCR quality; try better lighting or higher resolution.
- Low FPS: reduce frame size (`--width`, `--height`) or close other CPU-intensive apps.

## 📁 Project Structure
```
.
├── app.py                  # Main application
├── README.md               # This file
├── images/                 # (Optional) Assets folder
├── known_faces/            # (Present but unused by app.py)
└── .venv/                  # (Optional) Virtual environment
```

## 📝 Notes
- The product detection step is heuristic and may need tuning (e.g., contour area threshold) for your use case.
- OCR is sensitive to motion blur and text orientation. Try holding products steady and front-facing to the camera.

