import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import LABELS, label_for

DATASET_DIR = './data'
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Har bir harf uchun bitta papka (data/0 ... data/21) — hand_features.LABELS bilan bir xil
number_of_classes = len(LABELS)
dataset_size = 200

cap = cv2.VideoCapture(0)

for j in range(0, number_of_classes):
    class_dir = os.path.join(DATASET_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"\nReady for class {j} ({label_for(j)})")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f'Class: {j} ({label_for(j)}) / {number_of_classes - 1}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "Press 's' to START", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if key == ord('s'):
            print(f"Capturing images for class {j}...")
            img_count = 0

            while img_count < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, f'Capturing: {img_count}/{dataset_size}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("frame", frame)
                cv2.waitKey(1)

                img_path = os.path.join(class_dir, f'{img_count}.jpg')
                cv2.imwrite(img_path, frame)

                img_count += 1

            print(f"Finished capturing class {j}")
            break

cap.release()
cv2.destroyAllWindows()