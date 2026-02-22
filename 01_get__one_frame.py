import cv2

VIDEO_PATH = "input.mp4"
FRAME_INDEX = 50
OUT_PATH = "one_frame.png"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()
cv2.imwrite(OUT_PATH, frame)
print(f"Saved frame {FRAME_INDEX} to {OUT_PATH}")
