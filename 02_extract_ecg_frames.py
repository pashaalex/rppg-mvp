import cv2
from pathlib import Path

VIDEO_PATH = "input.mp4"
# ROI
X1, Y1 = 470, 700
X2, Y2 = 640, 830
OUT_DIR = Path("ECG")


OUT_DIR.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[Y1:Y2, X1:X2]
    out_path = OUT_DIR / f"{frame_idx:05d}.png"
    cv2.imwrite(str(out_path), roi)
    frame_idx += 1

cap.release()
print(f"Saved frames to '{OUT_DIR}'")
