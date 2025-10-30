# ✋ Hand Recognition + Product Name Detection using MediaPipe, OpenCV & Tesseract

## 📘 Overview
This Python project detects **hands** using MediaPipe, identifies whether a product is held, and applies **OCR (Tesseract)** to extract the product name from the cropped hand region.

---

## 🚀 Features
- 🖐️ Detects hands in real-time using **MediaPipe**
- 📦 Detects whether a **product** is being held
- 🔠 Extracts **product name** using Tesseract OCR
- 🔄 Supports camera inversion (`none`, `horizontal`, `vertical`, or `both`)
- 🎥 Overlays hand detection, product name, and **FPS counter**
- 🪞 Transparent overlay for better visualization
- 🧾 Logs detections to `gesture_recognition.log`
- 💾 Saves output video to `output.mp4`

---

## 🧩 Dependencies

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy pytesseract
```

Install **Tesseract OCR** (choose your OS):
🔗 [Tesseract Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki)

---

## ⚙️ Run the Project

```bash
python hand_product_name_detection.py --camera 0 --width 640 --height 480 --invert horizontal
```

**Arguments:**
| Argument | Type | Description |
|-----------|------|-------------|
| `--camera` | int | Camera index (default: 0) |
| `--width` | int | Frame width (default: 640) |
| `--height` | int | Frame height (default: 480) |
| `--invert` | str | Flip camera feed (`none`, `horizontal`, `vertical`, or `both`) |

---

## 🧠 How It Works
1. **MediaPipe Hands** detects hand landmarks in real-time.  
2. **Contour Detection** checks if a product-like object is being held.  
3. **Tesseract OCR** scans the cropped region and extracts readable text (product name).  
4. The system overlays the detected product name and FPS on the video feed.  
5. All results are saved to a log file and output video.  

---

## 📄 Output Files
| File | Description |
|------|--------------|
| `output.mp4` | Recorded output with overlays |
| `gesture_recognition.log` | Log file of detected products and hands |

---

## 💡 Example Run
```
python hand_product_name_detection.py --camera 0 --invert horizontal
```

**Console Output:**
```
Hand: Left
Product detected in Left hand: CocaCola
Hand: Right
Product detected in Right hand: Pepsi
Application terminated.
```

---

## 🛠️ Tesseract Configuration
If Tesseract is installed in a custom path, update it in your code:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 📦 Future Improvements
- Add real-time product database lookup (e.g., via barcode/ML model)  
- Improve OCR filtering for better product name recognition  
- Implement multiple camera stream support  
- Add voice output for detected product names  

---

## 🧰 Author
Created by **Amit Kadam**  
📧 Email: kadamamit462@gmail.com  
📍 Location: Bhalki, India

---

### 🏁 Enjoy coding your AI-powered hand & product recognition system! ✨
