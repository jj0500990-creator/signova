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

# Sinf raqami (data/<raqam> papkasi) -> harf. Yangi harf qo'shilsa, faqat shu
# yerga yoziladi — ilova, yig'ish va tekshirish skriptlari shu ro'yxatdan oladi.
LABELS = {
    '0': 'A',  '1': 'B',  '2': 'E',  '3': 'F',  '4': 'G',
    '5': 'H',  '6': 'I',  '7': 'J',  '8': 'L',  '9': 'M',
    '10': 'N', '11': 'O', '12': 'P', '13': 'Q', '14': 'R',
    '15': 'S', '16': 'T', '17': 'U', '18': 'V', '19': 'X',
    '20': 'Sh', '21': 'Ch'
}


def label_for(class_id):
    """Sinf raqamini harfga aylantiradi; noma'lum sinf uchun raqamning o'zi."""
    return LABELS.get(str(class_id), str(class_id))


def create_landmarker():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        # Barcha harflar bitta qo'l bilan ko'rsatiladi. Faqat eng ishonchli qo'l
        # olinadi — ikkinchi qo'l (yoki xato aniqlash) xususiyatlarni buzmaydi.
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame_rgb):
    """RGB kadrdagi qo'llarni topadi. Har bir qo'l — 21 ta landmark ro'yxati."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.asarray(frame_rgb, dtype=np.uint8))
    return landmarker.detect(mp_image).hand_landmarks


def extract_features(hands):
    """Barcha nuqtalarning (x - min_x, y - min_y) / o'lcham qiymatlari, ketma-ket.

    Qo'l o'lchamiga bo'linadi, shuning uchun qo'l kameraga yaqin yoki uzoq
    bo'lishi natijaga ta'sir qilmaydi.
    """
    points = [(lm.x, lm.y) for hand in hands for lm in hand]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, min_y = min(xs), min(ys)
    scale = max(max(xs) - min_x, max(ys) - min_y) or 1e-6
    features = []
    for x, y in points:
        features.append((x - min_x) / scale)
        features.append((y - min_y) / scale)
    return features


def pad_features(features, max_length):
    features = list(features)
    if len(features) < max_length:
        return features + [0] * (max_length - len(features))
    return features[:max_length]
