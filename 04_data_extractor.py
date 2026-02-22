import cv2
import csv
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
VIDEO_PATH = "input.mp4"
MASK_PATH = "face_mask_proc.png"
OUT_CSV = "signals.csv"

START_FRAME = 0
# ----------------------------

def mean_bgr_under_mask(frame_bgr: np.ndarray, mask_bool: np.ndarray):
    """Return mean (R,G,B) under mask_bool. If mask empty -> (nan,nan,nan)."""
    if mask_bool.sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    # frame is BGR in OpenCV
    b = frame_bgr[:, :, 0][mask_bool].mean()
    g = frame_bgr[:, :, 1][mask_bool].mean()
    r = frame_bgr[:, :, 2][mask_bool].mean()
    return (r, g, b)

# Load mask (BGR)
mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_COLOR)
if mask_img is None:
    raise RuntimeError(f"Cannot read mask image: {MASK_PATH}")

# Build boolean masks.
# In OpenCV BGR: red is (0,0,255), green is (0,255,0)
red_mask = (mask_img[:, :, 0] == 0) & (mask_img[:, :, 1] == 0) & (mask_img[:, :, 2] == 255)
green_mask = (mask_img[:, :, 0] == 0) & (mask_img[:, :, 1] == 255) & (mask_img[:, :, 2] == 0)

if red_mask.sum() == 0:
    raise RuntimeError("Red mask is empty. Check that face_mask.png uses pure red (255,0,0).")
if green_mask.sum() == 0:
    raise RuntimeError("Green mask is empty. Check that face_mask.png uses pure green (0,255,0).")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

# Jump to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

out_path = Path(OUT_CSV)
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "face_r_mean", "face_g_mean", "face_b_mean",
        "ref_r_mean",  "ref_g_mean",  "ref_b_mean",
    ])

    frame_idx = START_FRAME
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stopped early at frame {frame_idx} (cannot read).")
            break

        # Sanity: mask and frame must match size
        if frame.shape[:2] != mask_img.shape[:2]:
            cap.release()
            raise RuntimeError(
                f"Frame size {frame.shape[:2]} != mask size {mask_img.shape[:2]}. "
                "Mask must be drawn for original video frames."
            )

        fr, fg, fb = mean_bgr_under_mask(frame, red_mask)    # face
        rr, rg, rb = mean_bgr_under_mask(frame, green_mask)  # reference (wall)

        writer.writerow([frame_idx, fr, fg, fb, rr, rg, rb])

        frame_idx += 1

cap.release()
print(f"Saved {START_FRAME}..{frame_idx-1} to {OUT_CSV}")
