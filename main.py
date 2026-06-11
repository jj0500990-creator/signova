import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import eel
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import pickle
import numpy as np
from collections import deque, Counter
import base64
import threading

eel.init('web')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
max_length = model_dict['max_length']

labels_dict = {
    '0': 'A',  '1': 'B',  '2': 'E',  '3': 'F',  '4': 'G',
    '5': 'H',  '6': 'I',  '7': 'J',  '8': 'L',  '9': 'M',
    '10': 'N', '11': 'O', '12': 'P', '13': 'Q', '14': 'R',
    '15': 'S', '16': 'T', '17': 'U', '18': 'V', '19': 'X',
    '20': 'Sh', '21': 'Ch'
}

print("Model sinflari tartibi:", list(model.classes_))

running = False
last_gesture = ""
output_text = ""


def inference_loop():
    global running, last_gesture, output_text, last_update_time

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path='./hand_landmarker.task'),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.5
    )

    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = None
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    else:
        print("No camera found")
        eel.inference_stopped()
        return

    prediction_buffer = deque(maxlen=15)
    CONFIDENCE_THRESHOLD = 0.8
    predicted_label = ""
    current_candidate = None
    gesture_count = 0
    stable_threshold = 5

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame_rgb, dtype=np.uint8))
        results = landmarker.detect(mp_image)

        x1, y1, x2, y2 = 0, 0, 0, 0

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * W)
                    y = int(landmark.y * H)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            data_aux, x_, y_ = [], [], []

            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

            min_x, min_y = min(x_), min(y_)

            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            if len(data_aux) < max_length:
                data_aux += [0] * (max_length - len(data_aux))
            else:
                data_aux = data_aux[:max_length]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            try:
                proba = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(proba)
                best_idx = np.argmax(proba)
                predicted_class = model.classes_[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(predicted_class)
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    
                    if most_common == current_candidate:
                        gesture_count += 1
                    else:
                        current_candidate = most_common
                        gesture_count = 1
                    
                    if gesture_count >= stable_threshold and current_candidate != last_gesture:
                        output_text += labels_dict[current_candidate]
                        last_gesture = current_candidate
                        gesture_count = 0
                        print(f"Stable gesture confirmed: {labels_dict[current_candidate]} | Confidence: {confidence:.2f}")
                    
                    predicted_label = labels_dict[current_candidate] if current_candidate else ""
                else:
                    current_candidate = None
                    gesture_count = 0
                    predicted_label = ""
                    print(f"Confidence too low: {confidence:.2f} | Sinf: {predicted_class}")
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_label = ""
        else:
            prediction_buffer.clear()
            predicted_label = ""
            current_candidate = None
            gesture_count = 0

        if predicted_label and x2 > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        eel.update_frame(frame_b64, output_text)

    cap.release()
    running = False
    eel.inference_stopped()


@eel.expose
def start_inference():
    global running
    if not running:
        running = True
        t = threading.Thread(target=inference_loop, daemon=True)
        t.start()
        return {'status': 'started'}
    return {'status': 'already_running'}


@eel.expose
def stop_inference():
    global running
    running = False
    return {'status': 'stopped'}


@eel.expose
def clear_output():
    global output_text, last_gesture
    output_text = ""
    last_gesture = ""
    return {'status': 'cleared'}


eel.start('index.html', size=(1080, 720))