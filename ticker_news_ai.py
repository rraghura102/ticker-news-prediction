# ticker_news_ai.py
# Author: Rayan Raghuram
# Copyright (c) 2025 Rayan Raghuram
# Description: Multi-input Keras model using JAX backend to combine OHLC and news headlines

import warnings
warnings.filterwarnings("ignore", category=Warning)

# Set JAX as the backend before importing Keras
import os
os.environ["KERAS_BACKEND"] = "jax"  

import keras_core as keras
import pandas as pd
import numpy as np
import pickle

from keras_core import layers
from keras_core.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === STEP 1: Load & preprocess training data ===

# Load OHLC and news data
ohlc_df = pd.read_csv("data/train_ohlc.csv")
news_df = pd.read_csv("data/train_news.csv")

# Normalize OHLC numeric features
scaler = StandardScaler()
ohlc_scaled = scaler.fit_transform(ohlc_df[['open', 'low', 'high', 'close', 'volume']])
ohlc_df[['open', 'low', 'high', 'close', 'volume']] = ohlc_scaled

# Merge OHLC and news on 'ticker'
merged = pd.merge(ohlc_df, news_df, on='ticker')

# Tokenize only merged headlines
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(merged['headline'])
news_seq = tokenizer.texts_to_sequences(merged['headline'])
news_pad = pad_sequences(news_seq, maxlen=10)

# Prepare model inputs
X_ohlc = merged[['open', 'low', 'high', 'close', 'volume']].values
X_news = news_pad

# Dummy binary classification target (replace with real labels for actual use)
y = np.random.randint(0, 2, len(merged))

# === STEP 2: Split into train/test ===

X_train_ohlc, X_test_ohlc, X_train_news, X_test_news, y_train, y_test = train_test_split(
    X_ohlc, X_news, y, test_size=0.2, random_state=42
)

# === STEP 3: Build multi-input Keras functional model ===

# OHLC input branch
input_ohlc = keras.Input(shape=(5,), name="ohlc_input")
x1 = layers.Dense(16, activation="relu")(input_ohlc)

# News input branch
input_news = keras.Input(shape=(10,), name="news_input")
x2 = layers.Embedding(input_dim=1000, output_dim=16)(input_news)
x2 = layers.GlobalAveragePooling1D()(x2)

# Combine both branches
combined = layers.concatenate([x1, x2])
z = layers.Dense(16, activation="relu")(combined)
z = layers.Dense(1, activation="sigmoid")(z)  # Binary output

# Build and compile model
model = Model(inputs=[input_ohlc, input_news], outputs=z)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(
    {"ohlc_input": X_train_ohlc, "news_input": X_train_news},
    y_train,
    epochs=5,
    validation_split=0.2
)

# === STEP 4: Save model, tokenizer, and scaler ===

model.save("multi_input_model.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === STEP 5: Zero-shot test prediction ===

try:
    # Load zero-shot test sets
    ohlc_test = pd.read_csv("data/test_ohlc.csv")
    news_test = pd.read_csv("data/test_news.csv")

    # Restore tokenizer and scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Preprocess test data
    ohlc_test_scaled = scaler.transform(ohlc_test[['open', 'low', 'high', 'close', 'volume']])
    news_seq_test = tokenizer.texts_to_sequences(news_test['headline'])
    news_pad_test = pad_sequences(news_seq_test, maxlen=10)

    # Load trained model and predict
    model = keras.models.load_model("multi_input_model.keras")
    preds = model.predict({"ohlc_input": ohlc_test_scaled, "news_input": news_pad_test})

    print("\nZero-shot Predictions:")
    for i, p in enumerate(preds):
        print(f"Ticker: {ohlc_test.iloc[i]['ticker']} → Prediction: {p[0]:.4f}")

except FileNotFoundError:
    print("\nZero-shot test skipped: test_ohlc.csv or test_news.csv not found.")
