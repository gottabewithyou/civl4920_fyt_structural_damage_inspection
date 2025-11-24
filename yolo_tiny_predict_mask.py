from yolov5 import train

train.run(
    data=r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\dataset.yaml",
    cfg='yolov5s-seg.yaml',
    weights='yolov5s-seg.pt',
    epochs=50,
    batch=16,
    imgsz=640,
    device='cpu'
)
