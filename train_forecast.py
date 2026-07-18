"""
Train a supervised forecast model: issue context → SST+DHW at +1/+3/+7 days.

Saves pinn_forecast_best.h5 (used preferentially by validate_forecast.py).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pinn_physics import wrap_forecast_model

_ROOT = os.path.dirname(os.path.abspath(__file__))


class ForecastCheckpoint(keras.callbacks.Callback):
    def __init__(self, path="pinn_forecast_best.h5", monitor="val_loss"):
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is not None and current < self.best:
            self.best = current
            self.model.save(self.path)
            print(f"\nEpoch {epoch + 1}: {self.monitor}={current:.6f} -> {self.path}")


def train(
    epochs: int = 100,
    batch_size: int = 128,
    physics_note: str = "Forecast model uses MSE on SST+DHW (no PDE on 6-D inputs)",
):
    X_train = np.load(os.path.join(_ROOT, "X_forecast_train.npy"))
    y_train = np.load(os.path.join(_ROOT, "y_forecast_train.npy"))
    X_val = np.load(os.path.join(_ROOT, "X_forecast_val.npy"))
    y_val = np.load(os.path.join(_ROOT, "y_forecast_val.npy"))

    print(physics_note)
    print(f"X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"X_val   {X_val.shape}  y_val   {y_val.shape}")

    model = wrap_forecast_model(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
            ),
            ForecastCheckpoint("pinn_forecast_best.h5"),
        ],
        verbose=1,
    )

    model.save(os.path.join(_ROOT, "pinn_forecast_final.h5"))
    hist_df_path = os.path.join(_ROOT, "forecast_training_history.csv")
    import pandas as pd

    pd.DataFrame(history.history).to_csv(hist_df_path, index=False)
    print(f"Saved pinn_forecast_best.h5 / pinn_forecast_final.h5")
    print(f"History -> {hist_df_path}")
    return history


if __name__ == "__main__":
    train()
