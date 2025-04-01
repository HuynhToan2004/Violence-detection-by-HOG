import os
import cv2
import numpy as np
from skimage.feature import hog

def load_data(image_dir):
    X = []
    y = []

    # NonViolence
    NonViolence = os.path.join(image_dir, 'NonViolence')
    for img in os.listdir(NonViolence):
        X.append(os.path.join(NonViolence, img))
        y.append(0)

    # Violence    
    Violence = os.path.join(image_dir, 'Violence')
    for img in os.listdir(Violence):
        X.append(os.path.join(Violence, img))
        y.append(1)

    return X, y

def preprocess_frame(frame, threshold=50, resize_dim=(128,128)):
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, resize_dim)
        return resized_frame
    except Exception as e:
        print(f"Lỗi trong preprocess_frame: {e}")
        return None

def extract_hog_features(image):
    try:
        features = hog(image, orientations=9, pixels_per_cell=(4,4),
                      cells_per_block=(3,3), block_norm='L2-Hys', visualize=False, feature_vector=True)
        return features
    except Exception as e:
        print(f"Lỗi trong extract_hog_features: {e}")
        return None
        
def process_video(video, label, resize_dim=(128,128)):
    feature_list = []
    label_list = []

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Không thể xác định FPS cho video: {video}. Sử dụng mặc định frame_interval=1.")
        frame_interval = 1
    else:
        frame_interval = max(int(round(fps / 5)), 1)

    frame_count = 0
    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_interval != 0:
            continue

        try:
            binary_frame = preprocess_frame(frame, resize_dim=resize_dim)
            if binary_frame is None:
                continue

            hog_features = extract_hog_features(binary_frame)
            if hog_features is None or len(hog_features) == 0:
                continue

            feature_list.append(hog_features)
            label_list.append(label)
            processed_frames += 1
        except Exception as e:
            print(f"Lỗi khi xử lý khung hình {frame_count} trong video {video}: {e}")
            continue

    cap.release()
    feature_list = np.array(feature_list)
    return np.mean(feature_list, axis=0), label_list
