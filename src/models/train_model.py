# src/models/train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from ..features.build_features import build_dataset

def train_lstm_policy():
    df = build_dataset()
    X = df[['AAPL_Close','BTC_Close','sentiment']].values
    y = (df['AAPL_Close'].pct_change().shift(-1) > 0).astype(int).dropna()
    X = X[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # TODO: reshape for LSTM and build simple LSTM + dense policy head
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    print(model.evaluate(X_test, y_test))

if __name__ == "__main__":
    train_lstm_policy()
