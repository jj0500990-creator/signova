import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []

dirs = sorted(os.listdir(DATA_DIR), key=lambda x: int(x))
total_dirs = len(dirs)

for idx, dir_ in enumerate(dirs):
    dir_path = os.path.join(DATA_DIR, dir_)
    img_files = os.listdir(dir_path)
    print(f"Processing class {dir_} ({idx+1}/{total_dirs}) - {len(img_files)} images")

    success = 0
    fail = 0

    for img_path in img_files:
        data_aux = []
        full_path = os.path.join(dir_path, img_path)

        img = cv2.imread(full_path)
        if img is None:
            fail += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(dir_)
            success += 1
        else:
            fail += 1

    print(f"  ✓ Success: {success} | ✗ Failed (qo'l topilmadi): {fail}")

print(f"\nJami: {len(data)} ta sample saqlandi")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("data.pickle saqlandi!")