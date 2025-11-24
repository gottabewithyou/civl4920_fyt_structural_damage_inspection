from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')
model.train(data=r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\dataset.yaml", epochs=50, imgsz=640, batch=16, device='cpu')
