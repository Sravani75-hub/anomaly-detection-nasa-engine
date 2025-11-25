import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tensorflow import keras
from preprocessing import load_cmapss
from visualize import plot_results

def autoencoder_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def detect_anomalies():
    df, sensor_cols = load_cmapss("dataset/train_FD001.txt")

    X = df[sensor_cols].values

    # Isolation Forest
    iso = IsolationForest(contamination=0.03, random_state=42)
    df['iso_flag'] = iso.fit_predict(X)  # -1 = anomaly

    # Autoencoder
    ae = autoencoder_model(len(sensor_cols))
    ae.fit(X, X, epochs=20, batch_size=64, verbose=0)

    reconstruction = ae.predict(X)
    mse = np.mean(np.power(X - reconstruction, 2), axis=1)

    threshold = np.percentile(mse, 95)
    df['ae_flag'] = (mse > threshold).astype(int)

    # Final anomaly
    df['anomaly'] = ((df['iso_flag'] == -1) | (df['ae_flag'] == 1)).astype(int)

    # Save log
    df.to_csv("results/anomaly_log.txt", index=False)

    # Visualization
    plot_results(df, sensor_cols)

detect_anomalies()
