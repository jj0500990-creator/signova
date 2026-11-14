import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import eel
import cv2
import pickle
import numpy as np
from collections import deque, Counter
import base64
import threading

from hand_features import create_landmarker, detect_hands, extract_features, pad_features, label_for

eel.init('web')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
max_length = model_dict['max_length']

print("Model sinflari tartibi:", list(model.classes_))

last_gesture = ""
output_text = ""

# Ishlab turgan aniqlash oqimi va uni to'xtatish signali. Har bir ishga
# tushirishda yangi Event yaratiladi, shuning uchun eski oqim yangisini
# "qayta yoqib" yubormaydi.
worker = None
stop_event = threading.Event()
worker_lock = threading.Lock()


def open_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def notify_stopped(message=""):
    # Faqat hozirgi oqim UI'ni "to'xtadi" holatiga o'tkazadi — eski, to'xtayotgan
    # oqim yangi ishga tushirilgan aniqlashning tugmalarini buzmasin.
    if worker is threading.current_thread():
        eel.inference_stopped(message)


def inference_loop(my_stop, previous):
    global last_gesture, output_text

    # Oldingi oqim kamerani bo'shatib bo'lishini kutamiz — ikkita oqim bir
    # vaqtda kameradan o'qimasin va matnga ikki marta yozmasin.
    if previous is not None:
        previous.join()

    if my_stop.is_set():
        return

    landmarker = create_landmarker()

    cap = open_camera()
    if cap is None:
        print("No camera found")
        notify_stopped("Kamera topilmadi")
        return

    prediction_buffer = deque(maxlen=15)
    CONFIDENCE_THRESHOLD = 0.8
    predicted_label = ""
    current_candidate = None
    gesture_count = 0
    stable_threshold = 5
    # Qo'l shuncha kadr ramkadan chiqsa yoki imo noaniq bo'lsa, imo "qo'yib yuborilgan"
    # hisoblanadi va o'sha harfni yana yozish mumkin bo'ladi (masalan, "LL").
    RELEASE_FRAMES = 8
    idle_frames = 0

    while not my_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Qo'l aylantirilmagan kadrdan aniqlanadi — o'quv rasmlari ham aylantirilmagan.
        # Kadr faqat ekranda ko'zgudagidek ko'rinishi uchun aylantiriladi.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = detect_hands(landmarker, frame_rgb)
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        x1, y1, x2, y2 = 0, 0, 0, 0
        new_letter = ""
        confident = False

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
                best_idx = np.argmax(proba)
                predicted_class = model.classes_[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    confident = True
                    prediction_buffer.append(predicted_class)
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    
                    if most_common == current_candidate:
                        gesture_count += 1
                    else:
                        current_candidate = most_common
                        gesture_count = 1
                    
                    if gesture_count >= stable_threshold and current_candidate != last_gesture:
                        new_letter = label_for(current_candidate)
                        output_text += new_letter
                        last_gesture = current_candidate
                        gesture_count = 0
                        print(f"Stable gesture confirmed: {new_letter} | Confidence: {confidence:.2f}")
                    
                    predicted_label = label_for(current_candidate) if current_candidate else ""
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

        if confident:
            idle_frames = 0
        else:
            idle_frames += 1
            if idle_frames >= RELEASE_FRAMES and last_gesture:
                last_gesture = ""
                prediction_buffer.clear()

        if predicted_label and x2 > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        eel.update_frame(frame_b64, output_text, new_letter)

    cap.release()
    landmarker.close()
    notify_stopped()


@eel.expose
def start_inference():
    global worker, stop_event
    with worker_lock:
        if worker is not None and worker.is_alive() and not stop_event.is_set():
            return {'status': 'already_running'}
        stop_event = threading.Event()
        worker = threading.Thread(target=inference_loop,
                                  args=(stop_event, worker), daemon=True)
        worker.start()
        return {'status': 'started'}


@eel.expose
def stop_inference():
    with worker_lock:
        stop_event.set()
    return {'status': 'stopped'}


@eel.expose
def clear_output():
    global output_text, last_gesture
    output_text = ""
    last_gesture = ""
    return {'status': 'cleared'}


eel.start('index.html', size=(1080, 720))