from ultralytics import YOLO

# image = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Resources\20251017 Meeting\crack_image\7Q3A9060-18.jpg"

# # Load the Tiny YOLO model (example: yolov8n.pt as a tiny model variant)
# model = YOLO('yolov8n.pt')

# # Run inference on the image
# results = model(image, conf=0.1)


# for result in results:
#     result.show()

import numpy as np
import os
from PIL import Image
from scipy.ndimage import label

def bounding_boxes_from_mask(mask):
    bounding_boxes = []
    if np.any(mask):
        labeled_mask, num_features = label(mask)
        for region_label in range(1, num_features + 1):
            region = (labeled_mask == region_label)
            coords = np.argwhere(region)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bounding_boxes.append((x_min, y_min, x_max, y_max))
    return bounding_boxes

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def save_yolo_label(label_path, bounding_boxes, img_width, img_height):
    with open(label_path, 'w') as f:
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            x_center, y_center, width, height = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)
            line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            f.write(line)

def process_masks_and_images(image_dir, mask_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for mask_file in os.listdir(mask_dir):
        if not mask_file.lower().endswith('.png'):
            continue
        base_name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, base_name + '.jpg')
        if not os.path.exists(image_path):
            print(f"Image {base_name}.jpg not found. Skipping.")
            continue
        
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        binary_mask = (mask_array > 127).astype(np.uint8)
        
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        bboxes = bounding_boxes_from_mask(binary_mask)
        
        label_path = os.path.join(label_dir, base_name + '.txt')
        save_yolo_label(label_path, bboxes, img_width, img_height)
        print(f"Processed {base_name} - labels saved to {label_path}")

# Example usage:
image_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traincrop\traincrop"
mask_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traincrop\traincrop_mask"
label_output_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traincrop"

process_masks_and_images(image_folder, mask_folder, label_output_folder)