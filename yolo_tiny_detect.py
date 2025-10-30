from ultralytics import YOLO

image = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Resources\20251017 Meeting\crack_image\7Q3A9060-18.jpg"

# Load the Tiny YOLO model (example: yolov8n.pt as a tiny model variant)
model = YOLO('yolov8n.pt')

# Run inference on the image
results = model(image)

for result in results:
    result.show()