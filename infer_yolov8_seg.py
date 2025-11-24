import cv2
import numpy as np
from yolov5 import YOLOv5

def main():
    model = YOLOv5("yolov5s-seg.pt", device="cpu")   # or 'cuda'

    img_path = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\testdata\images\20160222_080933.jpg"
    img = cv2.imread(img_path)

    results = model.predict(img)

    masks = results.masks  # list of binary np arrays

    overlay = img.copy()
    for mask in masks:
        color_mask = np.zeros_like(img)
        color_mask[:, :, 1] = (mask.astype(np.uint8)) * 255  # green overlay
        overlay = cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)

    cv2.imshow("Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
