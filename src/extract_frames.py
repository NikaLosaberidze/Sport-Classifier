import cv2
import os


VIDEO_DIR = 'videos'
OUTPUT_DIR = 'frames'
FRAME_RATE = 1

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames_from_video(video_path, output_path, frame_rate):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps * frame_rate)
    frame_idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{save_idx:03d}.png"
            cv2.imwrite(os.path.join(output_path, frame_filename), frame)
            save_idx += 1
        frame_idx += 1

    cap.release()

def main():
    for weight_class in os.listdir(VIDEO_DIR):
        class_path = os.path.join(VIDEO_DIR, weight_class)
        if not os.path.isdir(class_path):
            continue

        output_class_dir = os.path.join(OUTPUT_DIR, weight_class)
        ensure_dir(output_class_dir)

        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(class_path, video_file)
                print(f"Processing: {video_path}")
                extract_frames_from_video(video_path, output_class_dir, FRAME_RATE)

if __name__ == "__main__":
    main()