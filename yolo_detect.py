import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# === 1) Parse CLI args ===
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

# === 3) Load YOLO & labels ===
model  = YOLO(model_path, task='detect')
labels = model.names

# === 4) Determine source type ===
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mp4','.mov','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1].lower()
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.startswith(('rtsp://','http://','https://')):
    source_type = 'ip'
    cap_arg     = img_source
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx     = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx   = int(img_source[8:])
else:
    print(f'Invalid source: {img_source}')
    sys.exit(1)

# === 5) Parse resolution ===
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# === 6) Setup recording if requested ===
if record:
    if source_type not in ('video','usb','ip'):
        print('Recording only works for video, USB or IP streams.')
        sys.exit(1)
    if not resize:
        print('Recording requires --resolution.')
        sys.exit(1)
    record_name = 'demo1.avi'
    record_fps  = 30
    recorder    = cv2.VideoWriter(record_name,
                     cv2.VideoWriter_fourcc(*'MJPG'),
                     record_fps,
                     (resW, resH))

# === 7) Initialize source handles ===
if source_type in ('image','folder'):
    if source_type == 'image':
        imgs_list = [img_source]
    else:
        imgs_list = [f for f in glob.glob(os.path.join(img_source,'*'))
                     if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ('video','usb','ip'):
    # create VideoCapture
    if source_type == 'video':
        cap = cv2.VideoCapture(img_source)
    elif source_type == 'usb':
        cap = cv2.VideoCapture(usb_idx)
    else:  # ip
        cap = cv2.VideoCapture(cap_arg)
    # set resolution
    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# === 8) Colors & helpers ===
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168),(169,162,241),
               (98,118,150), (172,176,184)]

fps_buffer   = []
fps_avg_len  = 200
img_count    = 0
avg_fps      = 0

# === 9) Inference loop ===
while True:
    t0 = time.perf_counter()

    # --- load frame ---
    if source_type in ('image','folder'):
        if img_count >= len(imgs_list):
            print('Done.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret: break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Camera disconnected.')
            break

    elif source_type == 'ip':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('IP stream ended or error.')
            break

    else:  # picamera
        bgra  = cap.capture_array()
        frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    # --- resize display if needed ---
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # --- detect ---
    results = model(frame, verbose=False)
    dets    = results[0].boxes

    count = 0
    for d in dets:
        conf = float(d.conf)
        if conf < min_thresh:
            continue

        x1,y1,x2,y2 = map(int, d.xyxy[0].cpu().numpy())
        cls_id       = int(d.cls[0].cpu().numpy())
        label        = f"{labels[cls_id]}:{conf:.2f}"
        color        = bbox_colors[cls_id % len(bbox_colors)]

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-4), (x1+tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        count += 1

    # --- compute FPS ---
    dt = time.perf_counter() - t0
    fps_buffer.append(1.0/dt)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer)/len(fps_buffer)

    # --- overlay info ---
    if source_type in ('video','usb','ip','picamera'):
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f'Count: {count}', (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # --- show & record ---
    cv2.imshow('YOLO detection', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('s'), ord('S')):
        cv2.waitKey(0)
    if key in (ord('p'), ord('P')):
        cv2.imwrite('capture.png', frame)

# === 10) Cleanup ===
print(f'Average FPS: {avg_fps:.1f}')
if source_type in ('video','usb','ip'):
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
