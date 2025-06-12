import os, sys, time, argparse
from ultralytics import YOLO

# … parse args as before …

# install and load model
model = YOLO(args.model, task='detect')

# open stream
cap = cv2.VideoCapture(args.source)
if args.resolution:
    w,h = map(int, args.resolution.split('x'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

print("Starting headless detection…")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended.")
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    count = sum(1 for b in boxes if float(b.conf) >= args.thresh)

    # print timestamped count
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}  →  Apples detected: {count}")

cap.release()
