import pandas as pd
import numpy as np
import re
import joblib
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r'C:\Users\smitr\OneDrive\Documents\python_projects\password_strength_checker_app\backend\data.csv', low_memory=False)
data['password'] = data['password'].astype(str)
data = data[data['password'] != 'nan']
data = data.sample(frac=0.09, random_state=42)

def extract_password_features(password):
    features = {
        'password_length': len(password),
        'has_special_char': int(bool(re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/\\]', password))),
        'has_uppercase': int(bool(re.search(r'[A-Z]', password))),
        'has_lowercase': int(bool(re.search(r'[a-z]', password))),
        'has_digit': int(bool(re.search(r'\d', password))),
        'consecutive_digits': len(re.findall(r'\d{2,}', password)),
        'consecutive_letters': len(re.findall(r'[a-zA-Z]{2,}', password))
    }
    return features

feature_columns = ['password_length', 'has_special_char', 'has_uppercase', 'has_lowercase', 'has_digit', 'consecutive_digits', 'consecutive_letters']
password_features = data['password'].apply(extract_password_features).apply(pd.Series)
X = password_features[feature_columns].values
y = pd.to_numeric(data['strength'], errors='coerce').fillna(0).astype(int)
y_encoded = keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['password'])
max_length = min(20, max(len(x) for x in tokenizer.texts_to_sequences(data['password'])))

def preprocess_password(password):
    sequence = tokenizer.texts_to_sequences([password])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

model.save('password_strength_model.h5')
joblib.dump(tokenizer, 'password_tokenizer.pkl')
joblib.dump(scaler, 'password_strength_scaler.pkl')