import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

MODEL_DIR = Path("models")
LABELS_PATH = MODEL_DIR / "labels.json"
LBPH_PATH = MODEL_DIR / "lbph.yml"

def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_labels() -> Dict[str, str]:
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def build_haar() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def crop_align(
    gray: np.ndarray,
    box: Tuple[int, int, int, int],
    out_w: int = 170,
    out_h: int = 200,
    pad: float = 1.0,
):
  
    x, y, w, h = box
    h_img, w_img = gray.shape[:2]

    # Center of detected face
    cx = x + w // 2
    cy = y + h // 2

    # Half sizes with padding
    half_w = int(w * pad / 2.0)
    half_h = int(h * pad / 2.0)

    # Bounding box, clamped to image borders
    x0 = max(0, cx - half_w)
    y0 = max(0, cy - half_h)
    x1 = min(w_img, cx + half_w)
    y1 = min(h_img, cy + half_h)

    if x1 <= x0 or y1 <= y0:
        return None

    face = gray[y0:y1, x0:x1]
    if face.size == 0:
        return None

    return cv2.resize(face, (out_w, out_h), interpolation=cv2.INTER_AREA)

def verify_lbph(target_user: str, cam: int) -> None:
    ensure_dirs()

    if not LBPH_PATH.exists():
        raise SystemExit(
            "LBPH model not found.\n"
            "Train your model using data_collection.py (with auto-train enabled) "
            "before running verification."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(LBPH_PATH))

    id2name = load_labels()
    if not id2name:
        raise SystemExit(
            "labels.json not found or empty.\n"
            "Ensure that data_collection.py finished training and saved labels."
        )

    # JSON keys are strings: {"0": "name1", "1": "name2", ...}
    name2id = {v: int(k) for k, v in id2name.items()}
    if target_user not in name2id:
        raise SystemExit(f"User '{target_user}' not found in labels.json.")
    target_id = name2id[target_user]

    cascade = build_haar()

    # Camera (try MSMF then DSHOW)
    cap = cv2.VideoCapture(cam, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Camera {cam} could not be opened.")

    print(f"[VERIFY] target user: {target_user}")
    print("[INFO] Press 'q' to quit.")

    conf_history = []

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(90, 90)
        )

        status_text = "NO FACE DETECTED"
        status_color = (0, 255, 255)

        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
            face = crop_align(gray, (x, y, w, h), out_w=170, out_h=200, pad=1.0)
            if face is not None:
                face = cv2.equalizeHist(face)
                pred_id, conf = recognizer.predict(face)

                conf_history.append(conf)
                if len(conf_history) > 12:
                    conf_history.pop(0)
                conf_avg = sum(conf_history) / len(conf_history)

                # Map LBPH distance to a rough 0–100 "confidence" metric
                confidence = max(0, min(100, 100 - conf_avg))
                pred_name = id2name.get(str(pred_id), "Unknown")
                if pred_id == target_id and confidence >= 50:
                    status_text = f"ACCESS GRANTED: USER {pred_name} with Confidence({confidence:.1f}%)"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"ACCESS DENIED Confidence({confidence:.1f}%)"
                    status_color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)

        # Draw status text centered near the top
        text_size = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3
        )[0]
        text_x = max(10, (frame.shape[1] - text_size[0]) // 2)
        cv2.putText(
            frame,status_text,(text_x, 50),cv2.FONT_HERSHEY_SIMPLEX,
            1.1,status_color,
            3,cv2.LINE_AA,
        )

        cv2.imshow("Live Face Recognition (offline)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline face verification using pre-trained LBPH model."
    )
    parser.add_argument(
        "--user",
        required=True,
        help="User name to verify (must exist in models/labels.json).",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera index (default: 0).",
    )

    args = parser.parse_args()
    verify_lbph(args.user, args.cam)

if __name__ == "__main__":
    main()
