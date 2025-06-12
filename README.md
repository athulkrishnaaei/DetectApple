# DetectApple
create virtual env
put image size to 320 while int8tflite conversion

python yolo_detect.py --model=best_saved_model/best_full_integer_quant.tflite --source=https://192.168.178.51:8080


python3 detect_headless.py \
  --model best_saved_model/best_full_integer_quant.tflite \
  --source https://192.168.178.51:8080 \
  --thresh 0.5