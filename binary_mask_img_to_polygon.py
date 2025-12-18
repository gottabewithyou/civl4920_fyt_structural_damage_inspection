import numpy as np
import cv2
import os


def binary_mask_to_yolov8_polygon(mask_array, epsilon=2.0, min_area=10, class_id=0):
    """
    Convert a binary mask numpy array to a list of YOLOv8 polygon lines.

    Args:
        mask_array (np.ndarray): 2D binary mask (0/1).
        epsilon (float): Polygon approximation accuracy.
        min_area (float): Minimum contour area to keep.
        class_id (int): YOLO class id.

    Returns:
        List[str]: List of polygon annotation lines in YOLOv8 format
                   (class_id + normalized coords).
    """
    mask_cv = (mask_array * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask_array.shape
    polygon_lines = []

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


def mask_image_to_yolo_label(mask_path, label_path, epsilon=2.0, min_area=10, class_id=0):
    """
    Convert a single binary mask image to a YOLO polygon label txt file.

    Args:
        mask_path (str): Path to the binary mask image (e.g., .png).
        label_path (str): Path to save the YOLO txt label.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask image: {mask_path}")

    # Binarize in case mask is not strictly 0/255
    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_array = (mask_bin > 0).astype(np.uint8)

    polygon_lines = binary_mask_to_yolov8_polygon(mask_array, epsilon, min_area, class_id)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for line in polygon_lines:
            f.write(line + '\n')


def batch_convert_mask_images(input_dir, output_dir,
                              epsilon=2.0, min_area=10, class_id=0,
                              exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    """
    Batch convert all binary mask images in input_dir to YOLO polygon .txt files in output_dir.

    Args:
        input_dir (str): Folder containing input binary mask images.
        output_dir (str): Folder to save YOLO polygon annotation txt files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(exts):
            continue

        mask_path = os.path.join(input_dir, filename)
        # Replace image extension with .txt for label
        base_name, _ = os.path.splitext(filename)
        label_filename = base_name + ".txt"
        label_path = os.path.join(output_dir, label_filename)

        mask_image_to_yolo_label(mask_path, label_path, epsilon, min_area, class_id)
        print(f"Converted {filename} -> {label_filename}")


if __name__ == '__main__':
    input_mask_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding 2\datasets\CRACK500\testdata\testdata_mask"
    output_label_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding 2\datasets\CRACK500\testdata\labels"

    batch_convert_mask_images(input_mask_folder, output_label_folder)
