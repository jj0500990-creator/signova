import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data_dict = pickle.load(open('data.pickle', 'rb'))

data_raw = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Pad all samples to the same length
max_length = max(len(sample) for sample in data_raw)
print(f"Max sample length: {max_length}")

data = []
for sample in data_raw:
    if len(sample) < max_length:
        sample = list(sample) + [0] * (max_length - len(sample))
    else:
        sample = list(sample)[:max_length]
    data.append(sample)

data = np.asarray(data)
print(f"Data shape: {data.shape}")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier(n_estimators=200, random_state=42)  # 200 ta daraxt
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"\nModel accuracy: {score * 100:.2f}%")

# Har bir class aniqligini ko'rsatadi
print("\nHar bir harf uchun aniqlik:")
print(classification_report(y_test, y_predict))

label_order = sorted(list(set(labels)))
print(f"Label order: {label_order}")

with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'max_length': max_length,
        'label_order': label_order
    }, f)

print("Model saved.")