"""Qo'l nuqtalarini aniqlash va model uchun xususiyat (feature) tayyorlash.

main.py, py/create_dataset.py va py/inference_classiFier.py shu modulni
ishlatadi — shunda o'qitish va aniqlash bir xil formatda bo'ladi.
"""
import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(ROOT_DIR, 'hand_landmarker.task')


def create_landmarker():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        # Barcha harflar bitta qo'l bilan ko'rsatiladi. Faqat eng ishonchli qo'l
        # olinadi — ikkinchi qo'l (yoki xato aniqlash) xususiyatlarni buzmaydi.
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.5
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame_rgb):
    """RGB kadrdagi qo'llarni topadi. Har bir qo'l — 21 ta landmark ro'yxati."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.asarray(frame_rgb, dtype=np.uint8))
    return landmarker.detect(mp_image).hand_landmarks


def extract_features(hands):
    """Barcha nuqtalarning (x - min_x, y - min_y) qiymatlari, ketma-ket."""
    points = [(lm.x, lm.y) for hand in hands for lm in hand]
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    features = []
    for x, y in points:
        features.append(x - min_x)
        features.append(y - min_y)
    return features


def pad_features(features, max_length):
    features = list(features)
    if len(features) < max_length:
        return features + [0] * (max_length - len(features))
    return features[:max_length]
