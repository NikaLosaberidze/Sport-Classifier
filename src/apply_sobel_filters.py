# აუცილებელი ბიბლიოთეკების იმპორტი
import cv2  # OpenCV ვიდეოებისა და გამოსახულებების დასამუშავებლად
import os   # ფაილური სისტემის ოპერაციებისთვის
import numpy as np  # მათემატიკური ოპერაციებისთვის
import pandas as pd  # ცხრილებისა და მონაცემთა შენახვისთვის CSV ფორმატში

# საქაღალდეების სახელები
FRAME_DIR = 'frames'  # ფრეიმების (სურათების) შესანახი დირექტორია
OUTPUT_DIR = 'filtered'  # ფილტრირებული გამოსახულებების დირექტორია
FEATURE_CSV = 'frame_features (N).csv'  # CSV ფაილი, სადაც შეინახება გამოყვანილი მახასიათებლები

# ამოწმებს, არსებობს თუ არა საქაღალდე. თუ არა — ქმნის
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Sobel ფილტრის გამოყენება გამოსახულებაზე — წიბოების (edge) აღმოსაჩენად
def apply_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # გარდაქმნა შავ-თეთრად
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X მიმართულებით წიბოების პოვნა
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y მიმართულებით წიბოების პოვნა
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # წიბოების სრული სიმძლავრის გამოთვლა
    edge_magnitude = np.uint8(np.clip(edge_magnitude, 0, 255))  # შედეგის გარდაქმნა გამოსახულებად (0-255)
    return edge_magnitude

# მთლიანი ფრეიმების დამუშავების პროცესი
def process_frames():
    data = []  # მონაცემების სია, რომელშიც თითოეული ფრეიმის მახასიათებლები ჩაიწერება

    # გადადის ყველა სპორტის კლასის საქაღალდეში
    for sport_class in os.listdir(FRAME_DIR):
        class_input_dir = os.path.join(FRAME_DIR, sport_class)  # შემომავალი ფაილების ბილიკი
        class_output_dir = os.path.join(OUTPUT_DIR, sport_class)  # ფილტრირებული გამოსახულებების ბილიკი
        ensure_dir(class_output_dir)  # ამ საქაღალდის შექმნა, თუ არ არსებობს

        # თითოეული ფრეიმის დამუშავება
        for frame_file in os.listdir(class_input_dir):
            frame_path = os.path.join(class_input_dir, frame_file)
            image = cv2.imread(frame_path)  # გამოსახულების ჩატვირთვა
            if image is None:
                print(f"[WARN] ვერ ჩატვირთა {frame_path}")  # გაფრთხილება ჩატვირთვის შეცდომაზე
                continue

            edge = apply_sobel(image)  # წიბოების გამოყოფა sobel ფილტრით

            output_path = os.path.join(class_output_dir, frame_file)
            cv2.imwrite(output_path, edge)  # ფილტრირებული გამოსახულების შენახვა

            mean_edge = np.mean(edge)  # წიბოების საშუალო მნიშვნელობა
            std_edge = np.std(edge)  # წიბოების სტანდარტული გადახრა

            # ფრეიმის მახასიათებლების დამატება სიაში
            data.append({
                'frame': frame_file,
                'class': sport_class,
                'mean_edge': mean_edge,
                'std_edge': std_edge
            })

    # მონაცემების ცხრილის შექმნა და CSV ფაილად შენახვა
    df = pd.DataFrame(data)
    df.to_csv(FEATURE_CSV, index=False)
    print(f"[INFO] მახასიათებლები შენახულია ფაილში: {FEATURE_CSV}")

# სკრიპტის შესრულება
if __name__ == "__main__":
    process_frames()
