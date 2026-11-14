import os
import pickle
import sys
from collections import deque, Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import create_landmarker, detect_hands, extract_features, pad_features, label_for

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
max_length = model_dict['max_length']

cap = cv2.VideoCapture(0)
landmarker = create_landmarker()

predicted_label = ""
x1, y1, x2, y2 = 0, 0, 0, 0
prediction_buffer = deque(maxlen=7)
CONFIDENCE_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Qo'l aylantirilmagan kadrdan aniqlanadi — o'quv rasmlari ham aylantirilmagan.
    # Kadr faqat ekranda ko'zgudagidek ko'rinishi uchun aylantiriladi.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = detect_hands(landmarker, frame_rgb)
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    if hands:
        x_ = [1 - lm.x for hand in hands for lm in hand]  # ekrandagi (aylantirilgan) koordinata
        y_ = [lm.y for hand in hands for lm in hand]
        for x, y in zip(x_, y_):
            cv2.circle(frame, (int(x * W), int(y * H)), 3, (0, 255, 0), -1)

        data_aux = pad_features(extract_features(hands), max_length)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        try:
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(proba)

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(model.classes_[np.argmax(proba)])
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                predicted_label = label_for(most_common)
            else:
                predicted_label = ""

        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_label = ""

    else:
        prediction_buffer.clear()
        predicted_label = ""

    if predicted_label:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
