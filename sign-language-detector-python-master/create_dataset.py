import os
import pickle
import warnings

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Suppress specific UserWarning related to google.protobuf.symbol_database
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Initialize absl logging
import absl.logging
absl.logging.set_verbosity('info')
absl.logging.set_stderrthreshold('info')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

print("Starting to process images...")

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        print(f"Processing directory: {dir_path}")
        for img_path in os.listdir(dir_path):
            print(f"Processing image: {img_path}")
            data_aux = []

            x_ = []
            y_ = []
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)
                print(f"Added data for image: {img_path}")

print("Finished processing images.")
print(f"Number of data entries: {len(data)}")
print(f"Number of labels: {len(labels)}")

if data and labels:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("data.pickle file created successfully.")
else:
    print("No data to write to data.pickle file.")


