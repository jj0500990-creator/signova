import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
max_length = model_dict['max_length']

labels_dict = {
    '0': 'A', '1': 'B', '2': 'E', '3': 'F', '4': 'G',
    '5': 'H', '6': 'I', '7': 'J', '8': 'L', '9': 'M',
    '10': 'N', '11': 'O', '12': 'P', '13': 'Q', '14': 'R',
    '15': 'S', '16': 'T', '17': 'U', '18': 'V', '19': 'X',
    '20': 'Sh', '21': 'Ch'
}

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

predicted_label = ""
x1, y1, x2, y2 = 0, 0, 0, 0
prediction_buffer = deque(maxlen=7)
CONFIDENCE_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        if len(data_aux) < max_length:
            data_aux = data_aux + [0] * (max_length - len(data_aux))
        else:
            data_aux = data_aux[:max_length]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        try:
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(proba)

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = model.predict([np.asarray(data_aux)])
                prediction_buffer.append(str(prediction[0]))

                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                predicted_label = labels_dict[most_common]
            else:
                predicted_label = ""

        except Exception as e:
            print(f"Prediction error: {e}")

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
cv2.destroyAllWindows()