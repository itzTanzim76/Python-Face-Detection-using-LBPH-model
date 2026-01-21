import argparse
import time
import shutil
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import cv2
import numpy as np

# ===============================
# Paths / setup
# ===============================
DATA_DIR = Path("data/faces") 
MODEL_DIR = Path("models")
LBPH_PATH = MODEL_DIR / "lbph.yml"
LABELS_PATH = MODEL_DIR / "labels.json"

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Quality checks
# ===============================
def laplacian_var(img: np.ndarray) -> float:
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def avg_brightness(img_gray: np.ndarray) -> float:
    return float(np.mean(img_gray))

def quality_ok(
    gray: np.ndarray,
    box: Tuple[int, int, int, int],
    min_size: int,
    blur_thresh: float,
    brightness_min: float,
) -> Tuple[bool, Dict[str, float], str]:
    x, y, w, h = box
    metrics = {"size_min": min(w, h), "blur": 0.0, "brightness": 0.0}
    if min(w, h) < min_size:
        return False, metrics, f"size {min(w,h)}<{min_size}"
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return False, metrics, "empty ROI"
    metrics["blur"] = laplacian_var(roi)
    if metrics["blur"] < blur_thresh:
        return False, metrics, f"blur {metrics['blur']:.0f}<{blur_thresh:.0f}"
    metrics["brightness"] = avg_brightness(roi)
    if metrics["brightness"] < brightness_min:
        return False, metrics, f"dark {metrics['brightness']:.0f}<{brightness_min:.0f}"
    return True, metrics, "ok"

# ===============================
# Detection (Haar Cascades)
# ===============================
def haar_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_one_face(gray: np.ndarray, cascade) -> Optional[Tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(90, 90))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    return (int(x), int(y), int(w), int(h))

# ===============================
# Tracking
# ===============================
def make_tracker(kind: str):
    k = (kind or "").lower()
    if k == "csrt": return cv2.TrackerCSRT_create()
    return None

# ===============================
# Crop / align
# ===============================
def crop_face(frame_bgr: np.ndarray, box: Tuple[int, int, int, int], out_w=170, out_h=200, pad: float = 1.0) -> Optional[np.ndarray]:

    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    side = int(max(w, h) * float(pad))
    x0 = max(cx - side // 2, 0)
    y0 = max(cy - side // 2, 0)
    x1 = min(x0 + side, frame_bgr.shape[1])
    y1 = min(y0 + side, frame_bgr.shape[0])
    face = frame_bgr[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return cv2.resize(face, (out_w, out_h), interpolation=cv2.INTER_AREA)

# ===============================
# Training helpers (LBPH)
# ===============================
def scan_dataset(img_size=(200, 200)):
    images, labels, id2name = [], [], {}
    current_id = 0
    for user_dir in sorted(DATA_DIR.glob("*")):
        if not user_dir.is_dir():
            continue
        user = user_dir.name
        id2name[current_id] = user
        for img_path in sorted(user_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            img = cv2.equalizeHist(img)
            images.append(img)
            labels.append(current_id)
        current_id += 1
    return images, labels, id2name

def save_labels(mapping: dict):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def train_lbph():
    print("[TRAIN] Scanning dataset…")
    images, labels, id2name = scan_dataset(img_size=(170, 200))
    if len(images) < 2:
        print("[TRAIN] Not enough images to train. Skipping model update.")
        return False

    print(f"[TRAIN] images={len(images)} users={len(set(labels))}")
    # requires opencv-contrib
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.save(str(LBPH_PATH))
    save_labels(id2name)
    print(f"[TRAIN] Saved model → {LBPH_PATH}")
    print(f"[TRAIN] Saved labels → {LABELS_PATH}")
    return True

# ===============================
# Main – Capture + Auto-train
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Offline face data collection (tracking) + optional full-frame saves + auto-train LBPH."
    )
    parser.add_argument("--user", required=True, help="User label (folder name).")
    parser.add_argument("--cam", type=int, default=0, help="Camera index.")
    parser.add_argument("--per-pose", type=int, default=25, help="Images to capture.")
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between saves.")
    parser.add_argument("--min-size", type=int, default=90, help="Min face size (px).")
    parser.add_argument("--blur-thresh", type=float, default=18.0, help="Min Laplacian variance.")
    parser.add_argument("--brightness-min", type=float, default=45.0, help="Min brightness.")
    parser.add_argument("--tracker", choices=["none", "csrt", "kcf"], default="csrt", help="Stabilize bounding box.")
    parser.add_argument("--bypass-quality", action="store_true", help="Save even if quality checks fail.")
    parser.add_argument("--no-train", action="store_true", help="Do not train after capture (default trains).")

    # NEW: saving behavior
    parser.add_argument("--save-mode", choices=["crop", "frame", "both"], default="crop",
                        help="What to save: 200x200 cropped face, full frame with background, or both.")
    parser.add_argument("--pad", type=float, default=1.0,
                        help="Crop padding as a scale of max(w,h). 1.0=tight, >1.0=more background.")

    args = parser.parse_args()
    ensure_dirs()

    # Clean only THIS user's old face data (others kept for multi-user training)
    user_dir = DATA_DIR / args.user
    if user_dir.exists():
        print(f"[WARN] Old data for '{args.user}' found in faces/. Deleting…")
        shutil.rmtree(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)

    # Camera (MSMF → DSHOW)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Camera {args.cam} could not be opened.")

    cascade = haar_detector()
    use_tracker = args.tracker != "none"
    tracker = None
    have_track = False

    total_needed = args.per_pose
    saved_total = 0
    last_save = 0.0

    print(f"[INFO] Capturing {total_needed} images for '{args.user}'")
    print("[INFO] Keys:  Q=Quit")

    while saved_total < total_needed:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Keep a clean copy BEFORE drawing HUD/boxes
        frame_clean = frame.copy()
        gray = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2GRAY)

        # Detect / track
        box = detect_one_face(gray, cascade)
        if box is not None:
            if use_tracker:
                tracker = make_tracker(args.tracker)
                if tracker is not None:
                    tracker.init(frame_clean, tuple(box))
                    have_track = True
        elif use_tracker and tracker is not None and have_track:
            ok_trk, bb = tracker.update(frame_clean)
            if ok_trk:
                x, y, w, h = [int(v) for v in bb]
                box = (x, y, w, h)

        # ----- HUD (YELLOW) on the DISPLAY frame only -----
        hud_lines = [
            f"User: {args.user}",
            f"Captured: {saved_total}/{total_needed}",
            "Keys: S=force-save  Q=quit"
        ]
        y0 = 26
        hud_color = (0, 255, 255)  # yellow
        for line in hud_lines:
            cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)
            y0 += 26
        # --------------------------------------------------

        message, msg_color = "No face detected", (0, 220, 255)

        if box is not None:
            x, y, w, h = box
            # draw guide box on the DISPLAY frame (not saved)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 80), 2)

            ok_q, _, why = quality_ok(
                gray, box,
                min_size=args.min_size,
                blur_thresh=args.blur_thresh,
                brightness_min=args.brightness_min
            )

            now = time.time()
            if (ok_q or args.bypass_quality) and (now - last_save) >= args.interval:
                wrote_any = False

                # 1) Save crop if requested
                if args.save_mode in ("crop", "both"):
                    face_img = crop_face(frame_clean, box, out_w=170, out_h=200, pad=args.pad)
                    if face_img is not None:
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        fname_crop = f"{args.user}_{saved_total:03d}.png"
                        if cv2.imwrite(str(user_dir / fname_crop), gray_face):
                            wrote_any = True

                if wrote_any:
                    saved_total += 1
                    last_save = now
                    message, msg_color = "[SAVE] ok", (0, 255, 0)
                else:
                    message, msg_color = "Write failed", (0, 0, 255)
            else:
                message = f"Hold: {why}" if (not ok_q and not args.bypass_quality) else "Hold still…"

        # Bottom status (on DISPLAY frame)
        cv2.putText(frame, message, (10, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2, cv2.LINE_AA)

        cv2.imshow("Offline LBPH based Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit pressed.")
            break

        if key == ord('s') and box is not None:
            wrote_any = False

    cap.release()
    cv2.destroyAllWindows()

    print("[DONE] Collected:")
    print(f"[TOTAL] {saved_total} images saved to: {user_dir}")

    # ===== Auto-train =====
    if not args.no_train:
        print("\n[INFO] Starting training (LBPH)…")
        ok = train_lbph()
        if ok:
            print("[INFO] Training complete. You can verify immediately with:")
            print("  python pic_password.py --user \"{name}\" --cam 0".format(name=args.user))
        else:
            print("[INFO] Training skipped (not enough images).")

if __name__ == "__main__":
    main()
