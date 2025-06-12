import os, sys, time, argparse
from ultralytics import YOLO

# … parse args as before …
parser = argparse.ArgumentParser()
parser.add_argument('--model',   help='Path to YOLO model file (e.g. "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source',  help='Image source: file, folder, video file, USB camera ("usb0"), or IP stream URL',
                    required=True)
parser.add_argument('--thresh',  help='Min confidence threshold (default: 0.5)', default=0.5, type=float)
parser.add_argument('--resolution', help='Display resolution WxH (e.g. "640x480")', default=None)
parser.add_argument('--record',  help='Record output to "demo1.avi" (requires --resolution)', action='store_true')
args = parser.parse_args()

model_path  = args.model
img_source  = args.source
min_thresh  = args.thresh
user_res    = args.resolution
record      = args.record

# === 2) Validate model path ===
if not os.path.exists(model_path):
    print(f'ERROR: Model not found at {model_path}')
    sys.exit(1)
    
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
