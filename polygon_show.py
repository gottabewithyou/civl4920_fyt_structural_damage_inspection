import cv2
import numpy as np

def draw_polygons_on_blank(image_path, annotation_path):
    # Load original image just to get dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Create a blank black image
    blank_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Read polygon annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        # Build numpy array of polygon points (scaled back to pixel coords)
        points = np.array([
            [int(coords[i] * width), int(coords[i + 1] * height)]
            for i in range(0, len(coords), 2)
        ], np.int32)

        points = points.reshape((-1, 1, 2))

        # Draw polygon in green on blank image
        cv2.polylines(blank_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Show the blank image with polygons
    cv2.imshow('Polygon Visualization on Blank', blank_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_polygons_on_image(image_path, annotation_path):
    # Load image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Read polygon annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        # Build numpy array of polygon points (scaled back to img pixel coords)
        points = np.array([
            [int(coords[i] * width), int(coords[i + 1] * height)]
            for i in range(0, len(coords), 2)
        ], np.int32)

        points = points.reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Show the image with polygons
    cv2.imshow('Polygon Visualization', img)
    cv2.waitKey(0)  # Wait until key press
    cv2.destroyAllWindows()  # Close window


image_file = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traindata\images\20160222_081011.jpg"          # Your image path
annotation_file = r"C:\Kit\HKUST\OneDrive - HKUST Connect\2025~26 Year 4\CIVL 4920 T01 - Civil and Environmental Engineering Final Year Thesis\Coding\datasets\CRACK500\traindata\label\20160222_081011.txt"  # Corresponding polygon txt annotations


draw_polygons_on_blank(image_file, annotation_file)
draw_polygons_on_image(image_file, annotation_file)

# Example usa