import numpy as np
import cv2
import os

def yolo_polygon_to_binary_mask(label_path, img_width, img_height, class_id=0):
    """
    Convert YOLOv8 polygon label file to a binary mask numpy array.
    
    Args:
        label_path (str): Path to YOLOv8 polygon .txt file
        img_width (int): Original image width
        img_height (int): Original image height  
        class_id (int): Class ID to extract (default 0)
    
    Returns:
        np.ndarray: Binary mask (0/1) of shape (height, width)
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not os.path.exists(label_path):
        return mask
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        parsed_class_id = int(float(parts[0]))
        if parsed_class_id != class_id:
            continue
            
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue
            
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_width)
            y = int(coords[i+1] * img_height)
            points.append([x, y])
        
        if len(points) < 3:
            continue
            
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    
    return mask

def batch_yolo_to_masks(label_dir, image_dir, output_dir, class_id=0):
    """
    Batch convert YOLO polygon labels to binary mask images.
    AUTO-DETECTS image dimensions from corresponding images.
    
    Args:
        label_dir (str): Directory with YOLO .txt label files
        image_dir (str): Directory with corresponding images (.jpg)
        output_dir (str): Directory to save binary mask PNG files
        class_id (int): Class ID to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
            
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, filename)
        img_path = os.path.join(image_dir, f"{base_name}.jpg")  # Adjust extension if needed
        
        if not os.path.exists(img_path):
            print(f"⚠ Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Cannot read image: {img_path}")
            continue
            
        img_height, img_width = img.shape[:2]
        print(f"Using size {img_width}x{img_height} for {filename}")
        
        mask = yolo_polygon_to_binary_mask(label_path, img_width, img_height, class_id)
        mask_path = os.path.join(output_dir, f"{base_name}.png")
        
        cv2.imwrite(mask_path, mask * 255)
        print(f"✓ {filename} -> {os.path.basename(mask_path)}")

def get_mask_from_single_label(label_path, img_path, class_id=0):
    """Quick function to get mask array from single label file + image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_height, img_width = img.shape[:2]
    return yolo_polygon_to_binary_mask(label_path, img_width, img_height, class_id)

if __name__ == '__main__':
    label_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding 2\datasets\CRACK500\traincrop\labels"
    image_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding 2\datasets\CRACK500\traincrop\images"  # Your images folder
    output_mask_folder = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding 2\datasets\CRACK500\traincrop\traincrop_mask2"
    
    batch_yolo_to_masks(label_folder, image_folder, output_mask_folder)
