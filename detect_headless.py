import os, sys, time, argparse
import cv2
from ultralytics import YOLO

# === 1) Parse CLI args ===
parser = argparse.ArgumentParser()
parser.add_argument('--model',      required=True,
                    help='Path to YOLO model file (.tflite, .pt, etc.)')
parser.add_argument('--source',     required=True,
                    help='Video source: file path or IP stream URL')
parser.add_argument('--thresh',     type=float, default=0.5,
                    help='Min confidence threshold')
parser.add_argument('--resolution', default=None,
                    help='WxH to resize frames (e.g. "640x480")')
args = parser.parse_args()

# === 2) Validate and load model ===
if not os.path.exists(args.model):
    print(f'ERROR: Model not found at {args.model}')
    sys.exit(1)
model = YOLO(args.model, task='detect')  # loads the TFLite or PyTorch model

# === 3) Open stream ===
cap = cv2.VideoCapture(args.source)
if args.resolution:
    w, h = map(int, args.resolution.split('x'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

print("Starting headless detection… (CTRL-C to stop)")
  
# === 4) Inference loop with label output ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended.")
        break

    # run model
    results = model(frame, verbose=False)
    boxes   = results[0].boxes  # list of Box objects
    ts      = time.strftime('%Y-%m-%d %H:%M:%S')

    detections = []
    for box in boxes:
        conf = float(box.conf)
        if conf < args.thresh:
            continue
        cls_id = int(box.cls[0])  # class index
        label  = model.names[cls_id]  # e.g. "normal_apple" or "rotten_apple"
        detections.append(f"{label} ({conf:.2f})")

    # print each detection on its own line
    if detections:
        print(f"{ts}  →  Detected:")
        for det in detections:
            print(f"    • {det}")
    else:
        print(f"{ts}  →  No detections above {args.thresh:.2f}")

    # summary count
    print(f"    →  Total apples this frame: {len(detections)}\n")

cap.release()
