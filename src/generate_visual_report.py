import os
import cv2
import pandas as pd
from collections import defaultdict
import shutil

FRAME_DIR = 'frames'
FILTERED_DIR = 'filtered'
FEATURE_CSV = 'frame_features.csv'
REPORT_DIR = 'visual_report'
CLUSTER_PLOT = 'cluster_plot.png'
CONFUSION_MATRIX = 'confusion_matrix.png'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def side_by_side(original, filtered):
    """Combine original and filtered image horizontally."""
    return cv2.hconcat([original, cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)])

def generate_comparison_images():
    """Create side-by-side visualizations for a few frames per class."""
    ensure_dir(REPORT_DIR)
    for weight_class in os.listdir(FRAME_DIR):
        input_class_dir = os.path.join(FRAME_DIR, weight_class)
        filtered_class_dir = os.path.join(FILTERED_DIR, weight_class)
        output_class_dir = os.path.join(REPORT_DIR, weight_class)
        ensure_dir(output_class_dir)

        sample_frames = sorted(os.listdir(input_class_dir))[:3]  # Show 3 frames per class
        for frame_name in sample_frames:
            original_path = os.path.join(input_class_dir, frame_name)
            filtered_path = os.path.join(filtered_class_dir, frame_name)
            output_path = os.path.join(output_class_dir, f"compare_{frame_name}")

            original = cv2.imread(original_path)
            filtered = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)
            if original is None or filtered is None:
                print(f"[WARN] Skipping: {frame_name}")
                continue

            combined = side_by_side(original, filtered)
            cv2.imwrite(output_path, combined)

def generate_video_class_summary():
    """Predict class for each video by majority vote of frame clusters."""
    df = pd.read_csv(FEATURE_CSV)
    if 'cluster' not in df.columns:
        print("Run kmeans_clustering.py first to generate cluster assignments!")
        return

    df['video'] = df['frame'].apply(lambda f: "_".join(f.split("_")[:-2]))  

    # Get all unique labels assigned
    unique_clusters = sorted(df['cluster'].unique())

    cluster_to_class = {}
    for cluster in unique_clusters:
        true_labels = df[df['cluster'] == cluster]['class']
        if not true_labels.empty:
            most_common = true_labels.mode()[0]
            cluster_to_class[cluster] = most_common
        else:
            cluster_to_class[cluster] = "Unknown"

    predictions = defaultdict(list)
    for _, row in df.iterrows():
        predictions[row['video']].append(row['cluster'])

    summary = []
    for video, clusters in predictions.items():
        most_common_cluster = pd.Series(clusters).mode()[0]
        predicted_class = cluster_to_class.get(most_common_cluster, "Unknown")
        summary.append((video, predicted_class))

    ensure_dir(REPORT_DIR)
    summary_df = pd.DataFrame(summary, columns=['Video', 'Predicted Class'])
    summary_df.to_csv(os.path.join(REPORT_DIR, 'video_predictions.csv'), index=False)
    print("Saved video-level predictions to visual_report/video_predictions.csv")



def main():
    generate_comparison_images()
    generate_video_class_summary()
    print(f"\nVisual evidence generated in: {REPORT_DIR}/")

if __name__ == "__main__":
    main()
