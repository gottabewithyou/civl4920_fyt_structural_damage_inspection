import numpy as np
import cv2
import os

def binary_mask_to_yolov8_polygon(mask_array, epsilon=2.0, min_area=10):
    """
    Convert a binary mask numpy array to a list of YOLOv8 polygon lines.

    Args:
        mask_array (np.ndarray): 2D binary mask (0/1).
        epsilon (float): Polygon approximation accuracy.
        min_area (float): Minimum contour area to keep.

    Returns:
        List[str]: List of polygon annotation lines in YOLOv8 format (class_id + normalized coords).
    """
    mask_cv = (mask_array * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask_array.shape
    polygon_lines = []
    class_id = 0

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        normalized = []
        for point in approx:
            x_norm = point[0][0] / width
            y_norm = point[0][1] / height
            normalized.extend([x_norm, y_norm])
        line = f"{class_id} " + " ".join([f"{c:.6f}" for c in normalized])
        polygon_lines.append(line)

    return polygon_lines

def batch_convert_masks(input_dir, output_dir, epsilon=2.0, min_area=10):
    """
    Batch convert all binary mask .txt files from input_dir to YOLO polygon .txt files in output_dir.

    Args:
        input_dir (str): Folder containing input binary mask txt files.
        output_dir (str): Folder to save YOLO polygon annotation txt files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
        input_path = os.path.join(input_dir, filename)
        mask_array = np.loadtxt(input_path, dtype=np.uint8, delimiter=",")

        polygon_lines = binary_mask_to_yolov8_polygon(mask_array, epsilon, min_area)

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            for line in polygon_lines:
                f.write(line + '\n')

        print(f"Converted {filename}: {len(polygon_lines)} polygon(s) found")

if __name__ == '__main__':
    input_mask_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traindata\images"
    output_label_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traindata\label"

    batch_convert_masks(input_mask_folder, output_label_folder)
