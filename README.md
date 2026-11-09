<div align="center">

<img src="web/logo.png" alt="Signova logo" width="120">

# Signova

**O'zbek daktil alifbosi imo-ishoralarini real vaqtda matnga aylantiruvchi desktop ilova**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Eel](https://img.shields.io/badge/GUI-Eel-2E7D32)

</div>

---

## ✨ Imkoniyatlar

- 📷 **Real vaqtda aniqlash** — veb-kamera orqali qo'l harakatlarini kuzatadi
- ✋ **Ikki qo'lni qo'llab-quvvatlaydi** — MediaPipe Hand Landmarker yordamida 21 × 2 nuqta
- 🧠 **Yuqori aniqlik** — RandomForest modeli, test to'plamida **~99.8%**
- 🛡️ **Barqarorlik filtri** — harf faqat bir necha kadr davomida ishonchli (≥ 80%) aniqlangandagina yoziladi
- 🔤 **Matn yig'ish** — aniqlangan harflar ketma-ket so'zga qo'shiladi
- 📊 **Harflar statistikasi** — har bir harf necha marta yozilgani ko'rsatiladi

## 🔤 Taniladigan harflar

22 ta belgi:

| A | B | E | F | G | H | I | J | L | M | N |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **O** | **P** | **Q** | **R** | **S** | **T** | **U** | **V** | **X** | **Sh** | **Ch** |

## ⚙️ Qanday ishlaydi

```
 Kamera kadri
     │
     ▼
 MediaPipe Hand Landmarker ──► 21 ta nuqta har bir qo'l uchun
     │
     ▼
 Normallashtirish (x − min_x, y − min_y) ──► 84 ta son
     │
     ▼
 RandomForest (model.p) ──► harf + ishonch darajasi
     │
     ▼
 Barqarorlik filtri ──► matnga qo'shiladi
     │
     ▼
 Eel ──► HTML interfeys (kadr + matn)
```

Xususiyatlarni tayyorlash `hand_features.py` da bir marta yozilgan — o'qitish
ham, ilova ham aynan shu koddan foydalanadi, shuning uchun ular doim bir xil
formatda ishlaydi.

## 🚀 O'rnatish

**Talablar:** Python 3.11, veb-kamera, Chrome yoki Edge brauzeri (Eel oynasi uchun).

```bash
git clone <repo-url>
cd gui
pip install eel opencv-python mediapipe==0.10.33 scikit-learn numpy
```

## ▶️ Ishga tushirish

```bash
python main.py
```

1. **«Aniqlashni Boshlash»** tugmasini bosing
2. Qo'lingizni kamera ramkasiga joylashtiring va harfni ~1 soniya ushlab turing
3. Aniqlangan matn o'ng panelda paydo bo'ladi
4. **«Tozalash»** — matn va statistikani nolga qaytaradi, **«To'xtatish»** — kamerani o'chiradi

> 💡 **Maslahat:** yaxshi yoritilgan joyda, oddiy fon oldida ishlating.

## 🧪 Modelni qayta o'qitish

Barcha buyruqlar loyiha ildizidan ishga tushiriladi:

| Qadam | Buyruq | Natija |
|---|---|---|
| 1. Rasm yig'ish | `python py/collect_imgs.py` | `data/<sinf>/*.jpg` (har sinfga 200 ta) |
| 2. Dataset yaratish | `python py/create_dataset.py` | `data.pickle` |
| 3. Modelni o'qitish | `python py/train_classiFier.py` | `model.p` |
| 4. Tekshirish | `python check_model.py` | sinflar va dataset balansi |
| 5. Tezkor sinov (GUI'siz) | `python py/inference_classiFier.py` | OpenCV oynasi, `q` — chiqish |

> ⚠️ 3-qadam ildizdagi `model.p` ni **ustidan yozadi** — kerak bo'lsa avval nusxasini saqlang.

Yangi harf qo'shish uchun: `data/` ga yangi sinf papkasini to'ldiring, so'ng
`main.py` dagi `labels_dict` va `web/index.html` dagi harflar ro'yxatiga
qo'shing.

## 📁 Loyiha tuzilishi

```
gui/
├── main.py                  # Asosiy ilova (Eel + kamera + bashorat)
├── hand_features.py         # Qo'l nuqtalari va xususiyatlar (umumiy modul)
├── check_model.py           # Model va dataset haqida ma'lumot
├── model.p                  # O'qitilgan RandomForest modeli
├── hand_landmarker.task     # MediaPipe qo'l aniqlash modeli
├── data/                    # O'quv rasmlari: 0..21 sinflar
├── py/
│   ├── collect_imgs.py      # Kameradan rasm yig'ish
│   ├── create_dataset.py    # Rasmlar → data.pickle
│   ├── train_classiFier.py  # data.pickle → model.p
│   └── inference_classiFier.py  # GUI'siz sinov
└── web/
    ├── index.html           # Interfeys
    ├── css/style.css
    └── logo.png
```

## 🗺️ Rejalar

- [ ] Qolgan harflarni qo'shish (C, D, K, Y, Z, O', G' …)
- [ ] Bir xil harfni ketma-ket yozish imkoniyati (masalan, «LL»)
- [ ] P va T harflari uchun datasetni yaxshilash (qo'l ba'zi rasmlarda topilmaydi)
- [ ] `requirements.txt` qo'shish
