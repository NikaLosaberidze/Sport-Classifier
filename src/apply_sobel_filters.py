import cv2
import os
import numpy as np
import pandas as pd

FRAME_DIR = 'frames'
OUTPUT_DIR = 'filtered'
FEATURE_CSV = 'frame_features (N).csv'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = np.uint8(np.clip(edge_magnitude, 0, 255))
    return edge_magnitude

def process_frames():
    data = []

    for sport_class in os.listdir(FRAME_DIR):
        class_input_dir = os.path.join(FRAME_DIR, sport_class)
        class_output_dir = os.path.join(OUTPUT_DIR, sport_class)
        ensure_dir(class_output_dir)

        for frame_file in os.listdir(class_input_dir):
            frame_path = os.path.join(class_input_dir, frame_file)
            image = cv2.imread(frame_path)
            if image is None:
                print(f"[WARN] Couldn't load {frame_path}")
                continue

            edge = apply_sobel(image)

            output_path = os.path.join(class_output_dir, frame_file)
            cv2.imwrite(output_path, edge)

            mean_edge = np.mean(edge)
            std_edge = np.std(edge)

            data.append({
                'frame': frame_file,
                'class': sport_class,
                'mean_edge': mean_edge,
                'std_edge': std_edge
            })

    df = pd.DataFrame(data)
    df.to_csv(FEATURE_CSV, index=False)
    print(f"[INFO] Saved features to {FEATURE_CSV}")

if __name__ == "__main__":
    process_frames()
