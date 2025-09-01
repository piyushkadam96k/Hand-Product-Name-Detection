"""
Hand Recognition to Detect Product and Product Name in Hand using MediaPipe + OpenCV + Tesseract

Features:
- Detects hands using MediaPipe
- Crops the hand region and uses contour detection to confirm a product is held
- Applies Tesseract OCR to extract product name from the cropped region
- Supports camera inversion (--invert none, horizontal, vertical, or both)
- Overlays hand detection, product detection, product name, and FPS with transparency
- Logs detections to gesture_recognition.log
- Saves output to output.mp4

Dependencies:
    pip install opencv-python mediapipe numpy pytesseract
    Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki

Run:
    python hand_product_name_detection.py --camera 0 --width 640 --height 480 --invert horizontal
"""

import cv2
import time
import argparse
import logging
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("MediaPipe is required. Install with: pip install mediapipe") from e

try:
    import pytesseract
except ImportError as e:
    raise SystemExit("Pytesseract is required. Install with: pip install pytesseract") from e

# Set Tesseract path (update based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Setup logging
logging.basicConfig(
    filename="gesture_recognition.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------
# Product Name Detection
# -------------------------------------------------------------

def extract_product_name(image):
    """Extract text from image using Tesseract OCR."""
    try:
        # Preprocess image for better OCR accuracy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run Tesseract OCR
        text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
        
        # Filter text: take first meaningful word (assume it's the product name)
        words = [word for word in text.split() if len(word) > 2]  # Ignore short strings
        return words[0] if words else "Unknown"
    except Exception as e:
        logging.error(f"OCR failed: {str(e)}")
        return "prodduct error"

# -------------------------------------------------------------
# Main Application
# -------------------------------------------------------------

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hand Recognition to Detect Product and Product Name")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--invert", choices=['none', 'horizontal', 'vertical', 'both'], default='none',
                        help="Invert camera feed: none, horizontal, vertical, or both")
    args = parser.parse_args()

    # Initialize MediaPipe
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
    except Exception as e:
        print(f"[ERROR] MediaPipe import failed: {str(e)}")
        logging.error(f"MediaPipe import failed: {str(e)}")
        raise

    # Initialize webcam
    try:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            logging.error("Could not open webcam.")
            raise SystemExit("Could not open webcam.")
    except Exception as e:
        print(f"[ERROR] Webcam initialization failed: {str(e)}")
        logging.error(f"Webcam initialization failed: {str(e)}")
        raise

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize hand detector
    try:
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.4)
    except Exception as e:
        print(f"[ERROR] MediaPipe initialization failed: {str(e)}")
        logging.error(f"MediaPipe initialization failed: {str(e)}")
        raise

    # Initialize video writer
    try:
        out = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (args.width, args.height)
        )
    except Exception as e:
        print(f"[ERROR] Video writer initialization failed: {str(e)}")
        logging.error(f"Video writer initialization failed: {str(e)}")
        raise

    prev_time = time.time()
    fps = 0.0

    while True:
        try:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from webcam.")
                logging.error("Failed to read frame from webcam.")
                break
        except Exception as e:
            print(f"[ERROR] Frame read failed: {str(e)}")
            logging.error(f"Frame read failed: {str(e)}")
            break

        # Invert the frame based on --invert argument
        if args.invert == 'horizontal':
            frame = cv2.flip(frame, 1)  # Horizontal flip
        elif args.invert == 'vertical':
            frame = cv2.flip(frame, 0)  # Vertical flip
        elif args.invert == 'both':
            frame = cv2.flip(frame, -1)  # Both horizontal and vertical flip

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay = frame.copy()

        # Hand detection
        results = hands.process(rgb_frame)
        product_detected = False
        product_name = "None"
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                mp_drawing.draw_landmarks(
                    overlay,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                h, w, _ = frame.shape
                lm_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                handedness_label = handedness.classification[0].label
                xs = [p[0] for p in lm_px]
                ys = [p[1] for p in lm_px]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cv2.rectangle(overlay, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"{handedness_label}: Hand Detected",
                    (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                logging.info(f"Hand: {handedness_label}")

                # Crop hand region for product detection
                hand_crop = frame[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                if hand_crop.size == 0:
                    continue

                # Contour detection for product
                gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Check for product
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 1000:  # Adjust threshold for product size
                        product_detected = True
                        break

                if product_detected:
                    # Extract product name using OCR
                    product_name = extract_product_name(hand_crop)
                    cv2.putText(
                        overlay,
                        f"product: {product_name}",
                        (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    logging.info(f"Product detected in {handedness_label} hand: {product_name}")

        # Calculate and display FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Apply transparency to overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Write to video file
        try:
            out.write(frame)
        except Exception as e:
            print(f"[ERROR] Video write failed: {str(e)}")
            logging.error(f"Video write failed: {str(e)}")
            break

        # Display frame
        cv2.imshow("Hand Product Name Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    hands.close()
    logging.info("Application terminated.")

if __name__ == "__main__":
    main()
