import os
import pickle
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import create_landmarker, detect_hands, extract_features

landmarker = create_landmarker()

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
        full_path = os.path.join(dir_path, img_path)

        img = cv2.imread(full_path)
        if img is None:
            fail += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = detect_hands(landmarker, img_rgb)

        if hands:
            data.append(extract_features(hands))
            labels.append(dir_)
            success += 1
        else:
            fail += 1

    print(f"  ✓ Success: {success} | ✗ Failed (qo'l topilmadi): {fail}")

print(f"\nJami: {len(data)} ta sample saqlandi")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("data.pickle saqlandi!")
