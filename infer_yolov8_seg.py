import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO("best.pt")  # YOLOv8s-seg
    
    img_path = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding 2\datasets\CRACK500\testdata\images\20160222_114759.jpg"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb, verbose=False)[0]
    overlay = img_rgb.copy()
    
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        h, w = overlay.shape[:2]
        
        for mask in masks:
            mask_resized = cv2.resize(mask, (w, h)) > 0.5
            
            # Preserve original colors everywhere EXCEPT cracks
            colored_mask = overlay.copy()
            colored_mask[mask_resized] = [0, 255, 0]  # Green cracks only
            
            overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
    
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    # FORCE full display
    cv2.namedWindow("CRACK500 Full", cv2.WINDOW_NORMAL)
    cv2.imshow("CRACK500 Full", overlay_bgr)
    cv2.imwrite("full_result.jpg", overlay_bgr)  # Verify in file explorer
    cv2.waitKey(0)

if __name__ == '__main__':
    main()