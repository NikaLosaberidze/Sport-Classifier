# აუცილებელი ბიბლიოთეკების ჩატვირთვა
import os           # ფაილური სისტემის მართვა
import cv2          # OpenCV - გამოსახულებებთან სამუშაოდ
import pandas as pd # ცხრილებთან სამუშაოდ (DataFrame)
from collections import defaultdict  # დეფოლტური ლექსიკონი

# საქაღალდეებისა და ფაილების სახელები
FRAME_DIR = 'frames'                       # ორიგინალი კადრები
FILTERED_DIR = 'filtered'                  # დამუშავებული კადრები (სობელის ფილტრით)
FEATURE_CSV = 'frame_features.csv'         # კადრების მახასიათებლები (mean/std + cluster)
REPORT_DIR = 'visual_report'               # ანგარიშის საქაღალდე
CLUSTER_PLOT = 'cluster_plot.png'          # კლასტერის ვიზუალიზაცია
CONFUSION_MATRIX = 'confusion_matrix.png'  # ქაოსის მატრიცის გამოსახულება

# ამოწმებს, არსებობს თუ არა საქაღალდე, თუ არა — ქმნის
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ორ გამოსახულებას გვერდიგვერდ აერთიანებს (ორიგინალი + ფილტრირებული)
def side_by_side(original, filtered):
    """ორიგინალის და ფილტრირებული გამოსახულების ჰორიზონტალური გაერთიანება."""
    return cv2.hconcat([original, cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)])

# თითოეული კლასისთვის რამდენიმე კადრის ვიზუალური შედარება (3 კადრი)
def generate_comparison_images():
    """ყოველი კლასისთვის რამდენიმე კადრის ორიგინალის და ფილტრირებული ვერსიის შედარება."""
    ensure_dir(REPORT_DIR)  # ანგარიშის დირექტორიის შექმნა

    for weight_class in os.listdir(FRAME_DIR):  # თითოეულ წონით კატეგორიაზე (კლასი)
        input_class_dir = os.path.join(FRAME_DIR, weight_class)
        filtered_class_dir = os.path.join(FILTERED_DIR, weight_class)
        output_class_dir = os.path.join(REPORT_DIR, weight_class)
        ensure_dir(output_class_dir)

        sample_frames = sorted(os.listdir(input_class_dir))[:3]  # მხოლოდ 3 კადრი კლასიდან
        for frame_name in sample_frames:
            original_path = os.path.join(input_class_dir, frame_name)
            filtered_path = os.path.join(filtered_class_dir, frame_name)
            output_path = os.path.join(output_class_dir, f"compare_{frame_name}")

            original = cv2.imread(original_path)
            filtered = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)
            if original is None or filtered is None:
                print(f"[WARN] გამოტოვებულია: {frame_name}")
                continue

            combined = side_by_side(original, filtered)
            cv2.imwrite(output_path, combined)  # შენახვა

# ვიდეოს დონეზე კლასის პროგნოზირება კლასტერების უმრავლესობით
def generate_video_class_summary():
    """თითოეული ვიდეოს პროგნოზი — უმრავლესობის მიხედვით ფრეიმების კლასტერიდან."""
    df = pd.read_csv(FEATURE_CSV)  # ფრეიმების მახასიათებლების წაკითხვა

    if 'cluster' not in df.columns:
        print("გაუშვი kmeans_clustering.py რათა შეიქმნას cluster სვეტი!")
        return

    # ვიდეოს სახელის ამოღება ფრეიმის სახელიდან
    df['video'] = df['frame'].apply(lambda f: "_".join(f.split("_")[:-2]))

    # უნიკალური კლასტერები
    unique_clusters = sorted(df['cluster'].unique())

    # თითოეული კლასტერი შეესაბამება ყველაზე ხშირ კლასს
    cluster_to_class = {}
    for cluster in unique_clusters:
        true_labels = df[df['cluster'] == cluster]['class']
        if not true_labels.empty:
            most_common = true_labels.mode()[0]
            cluster_to_class[cluster] = most_common
        else:
            cluster_to_class[cluster] = "Unknown"

    # თითოეული ვიდეოსთვის ყველა კადრის კლასტერი
    predictions = defaultdict(list)
    for _, row in df.iterrows():
        predictions[row['video']].append(row['cluster'])

    # საბოლოო პროგნოზები ვიდეოს დონეზე
    summary = []
    for video, clusters in predictions.items():
        most_common_cluster = pd.Series(clusters).mode()[0]
        predicted_class = cluster_to_class.get(most_common_cluster, "Unknown")
        summary.append((video, predicted_class))

    ensure_dir(REPORT_DIR)
    summary_df = pd.DataFrame(summary, columns=['Video', 'Predicted Class'])
    summary_df.to_csv(os.path.join(REPORT_DIR, 'video_predictions.csv'), index=False)
    print("შენახულია ვიდეოების პროგნოზები: visual_report/video_predictions.csv")

# მთავარი ფუნქცია
def main():
    generate_comparison_images()        # ვიზუალური შედარებების გენერაცია
    generate_video_class_summary()      # ვიდეოების პროგნოზი კლასზე
    print(f"\nვიზუალური მასალა და ანგარიში შენახულია საქაღალდეში: {REPORT_DIR}/")

# სკრიპტის გაშვების წერტილი
if __name__ == "__main__":
    main()