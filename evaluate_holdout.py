"""
Hold-out evaluation: time-based test split + leave-one-location-out.

Reports MAE/RMSE in normalised and physical (°C) units.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import tensorflow as tf

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _inverse_temp(scaler, y_norm: np.ndarray) -> np.ndarray:
    import pandas as pd

    X = np.asarray(y_norm, dtype=np.float32).reshape(-1, 1)
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn):
        return scaler.inverse_transform(pd.DataFrame(X, columns=fn)).flatten()
    return scaler.inverse_transform(X).flatten()


def _metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "n": int(len(y_true)),
    }


def evaluate_split(model, scalers, X_path, y_path, name: str) -> dict:
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"[skip] {name}: missing {X_path}")
        return {}
    X = np.load(X_path)
    y = np.load(y_path)
    pred_n = model.predict(X, verbose=0)
    y_true_c = _inverse_temp(scalers["scaler_temp"], y)
    y_pred_c = _inverse_temp(scalers["scaler_temp"], pred_n)
    m_norm = _metrics(y.flatten(), pred_n.flatten())
    m_phys = _metrics(y_true_c, y_pred_c)
    result = {"split": name, "norm": m_norm, "physical_C": m_phys}
    print(f"\n{name}")
    print(f"  n={m_phys['n']}")
    print(f"  MAE  {m_phys['mae']:.4f} °C   (norm {m_norm['mae']:.4f})")
    print(f"  RMSE {m_phys['rmse']:.4f} °C   (norm {m_norm['rmse']:.4f})")
    print(f"  bias {m_phys['bias']:.4f} °C")
    return result


def run():
    model = tf.keras.models.load_model(
        os.path.join(_ROOT, "pinn_model_best.h5"), compile=False
    )
    with open(os.path.join(_ROOT, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(_ROOT, "sensor_info.pkl"), "rb") as f:
        info = pickle.load(f)

    print("=" * 60)
    print("HOLD-OUT EVALUATION")
    print("=" * 60)
    print(f"data_source: {info.get('data_source')}")
    print(f"holdout_location: {info.get('holdout_location')}")
    print(f"time_range: {info.get('time_range')}")

    results = []
    for name, xp, yp in [
        ("val (time)", "X_val.npy", "y_val.npy"),
        ("test (time)", "X_test.npy", "y_test.npy"),
        (
            f"location holdout ({info.get('holdout_location', '?')})",
            "X_loc_holdout.npy",
            "y_loc_holdout.npy",
        ),
    ]:
        r = evaluate_split(
            model,
            scalers,
            os.path.join(_ROOT, xp),
            os.path.join(_ROOT, yp),
            name,
        )
        if r:
            results.append(r)

    out = os.path.join(_ROOT, "holdout_evaluation.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {out}")
    return results


if __name__ == "__main__":
    run()
