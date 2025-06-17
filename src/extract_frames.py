# აუცილებელი ბიბლიოთეკების ჩატვირთვა
import cv2  # OpenCV - ვიდეოს ფრეიმების წასაკითხად
import os   # ფაილურ სისტემასთან სამუშაოდ

# საქაღალდეების სახელები და პარამეტრი
VIDEO_DIR = 'videos'       # ვიდეოების დირექტორია
OUTPUT_DIR = 'frames'      # გამოყოფილი ფრეიმების შენახვის დირექტორია
FRAME_RATE = 1             # რამდენ წამში ერთხელ ამოიღოს ფრეიმი (მაგ. 1 წამში ერთხელ)

# ამოწმებს, არსებობს თუ არა მითითებული საქაღალდე. თუ არა — ქმნის
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ფუნქცია, რომელიც ვიდეოდან ამოიღებს ფრეიმებს და შეინახავს მითითებულ საქაღალდეში
def extract_frames_from_video(video_path, output_path, frame_rate):
    cap = cv2.VideoCapture(video_path)  # ვიდეო ფაილის გახსნა
    fps = cap.get(cv2.CAP_PROP_FPS)  # ვიდეოს კადრების სიხშირე (fps)

    frame_interval = int(fps * frame_rate)  # რამდენ კადრში ერთხელ უნდა ამოიღოს ფრეიმი
    frame_idx = 0  # მიმდინარე კადრის ინდექსი
    save_idx = 0   # შენახული ფრეიმების მნიშნელობა (სახელის დასამატებლად)

    while True:
        ret, frame = cap.read()  # ვიდეოდან კადრის წაკითხვა
        if not ret:
            break  # თუ ვიდეო დასრულდა — გაჩერდეს

        # თუ კადრის ინდექსი სწორ ინტერვალზეა, შენახე
        if frame_idx % frame_interval == 0:
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{save_idx:03d}.png"
            cv2.imwrite(os.path.join(output_path, frame_filename), frame)
            save_idx += 1

        frame_idx += 1  # გადადით შემდეგ კადრზე

    cap.release()  # ვიდეო ფაილის გათავისუფლება

# მთავარი ფუნქცია: დირექტორიების გარჩევა და ყველა ვიდეოს დამუშავება
def main():
    for weight_class in os.listdir(VIDEO_DIR):  # თითოეულ ქვესაქაღალდეში (კლასი, წონითი კატეგორია და ა.შ.)
        class_path = os.path.join(VIDEO_DIR, weight_class)
        if not os.path.isdir(class_path):
            continue  # თუ ეს არ არის საქაღალდე — გამოტოვე

        output_class_dir = os.path.join(OUTPUT_DIR, weight_class)
        ensure_dir(output_class_dir)  # შექმენი გამოსახულებების საქაღალდე, თუ არ არსებობს

        for video_file in os.listdir(class_path):  # თითოეულ ვიდეო ფაილზე
            if video_file.endswith(('.mp4', '.avi', '.mov')):  # თუ ვიდეოს გაფართოება სწორია
                video_path = os.path.join(class_path, video_file)
                print(f"მიმდინარეობს დამუშავება: {video_path}")
                extract_frames_from_video(video_path, output_class_dir, FRAME_RATE)

# სკრიპტის შესრულების წერტილი
if __name__ == "__main__":
    main()
