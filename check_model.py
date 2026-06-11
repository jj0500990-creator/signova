import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

print("=" * 50)
print(f"Model turi: {type(model)}")
print(f"Max length: {model_dict['max_length']}")
print("=" * 50)

# Modelning sinflari
if hasattr(model, 'classes_'):
    print(f"\nModel sinflari: {model.classes_}")
    print(f"Sinflar soni: {len(model.classes_)}")

# RandomForest bo'lsa — har bir sinf uchun o'rtacha votes
if hasattr(model, 'estimators_'):
    print(f"\nDaraxtlar soni: {len(model.estimators_)}")

# Dataset bo'lsa tekshirish
try:
    data = pickle.load(open('./data.pickle', 'rb'))
    labels = np.array(data['labels'])
    unique, counts = np.unique(labels, return_counts=True)
    print("\n" + "=" * 50)
    print("Har bir sinf uchun rasm soni:")
    print("=" * 50)
    for cls, cnt in zip(unique, counts):
        bar = "█" * (cnt // 10)
        print(f"  Sinf {cls:>3}: {cnt:>4} ta  {bar}")
    print("=" * 50)
    print(f"Jami: {len(labels)} ta rasm")
    print(f"Min: {counts.min()} | Max: {counts.max()} | O'rtacha: {counts.mean():.0f}")
    if counts.max() / counts.min() > 3:
        print("\n⚠️  MUAMMO: Dataset balanssiz! Max/Min nisbati:", round(counts.max()/counts.min(), 1))
    else:
        print("\n✅ Dataset muvozanatli")
except FileNotFoundError:
    print("\n'data.pickle' topilmadi — faqat model tekshirildi")