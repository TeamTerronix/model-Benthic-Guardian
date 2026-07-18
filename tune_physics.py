"""
Sweep physics_weight for the interpolating PINN and pick the best val loss.

Writes physics_tune_results.csv and physics_best.pkl (recommended weight).
Does a short training run per candidate so notebooks stay fast.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from pinn_physics import wrap_pinn

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.3]


def _load_adv():
    path = os.path.join(_ROOT, "advection.pkl")
    if not os.path.exists(path):
        return 0.0, 0.0, 0.01
    with open(path, "rb") as f:
        d = pickle.load(f)
    return float(d.get("u_adv", 0.0)), float(d.get("v_adv", 0.0)), float(d.get("alpha", 0.01))


def tune(
    weights: Optional[List[float]] = None,
    epochs: int = 25,
    batch_size: int = 128,
) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    X_train = np.load(os.path.join(_ROOT, "X_train.npy"))
    y_train = np.load(os.path.join(_ROOT, "y_train.npy"))
    X_val = np.load(os.path.join(_ROOT, "X_val.npy"))
    y_val = np.load(os.path.join(_ROOT, "y_val.npy"))

    # Support dual-output if y has 2 cols
    output_dim = int(y_train.shape[1]) if y_train.ndim == 2 else 1
    u, v, alpha = _load_adv()

    rows = []
    best = {"physics_weight": 0.01, "val_loss": np.inf}

    for w in weights:
        print(f"\n=== physics_weight={w} ===")
        model = wrap_pinn(
            physics_weight=float(w),
            alpha=alpha,
            u_adv=u,
            v_adv=v,
            output_dim=output_dim,
        )
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
            verbose=0,
        )
        val_loss = float(min(hist.history["val_loss"]))
        val_phys = float(hist.history.get("val_physics_loss", [np.nan])[-1])
        val_data = float(hist.history.get("val_data_loss", [np.nan])[-1])
        val_mae = float(min(hist.history.get("val_mae", [np.nan])))
        rows.append(
            {
                "physics_weight": w,
                "best_val_loss": val_loss,
                "final_val_data_loss": val_data,
                "final_val_physics_loss": val_phys,
                "best_val_mae": val_mae,
                "epochs_run": len(hist.history["loss"]),
            }
        )
        print(
            f"  best_val_loss={val_loss:.6f}  val_mae={val_mae:.6f}  "
            f"val_physics={val_phys:.6f}"
        )
        if val_loss < best["val_loss"]:
            best = {
                "physics_weight": float(w),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "u_adv": u,
                "v_adv": v,
                "alpha": alpha,
            }

    results = pd.DataFrame(rows)
    out_csv = os.path.join(_ROOT, "physics_tune_results.csv")
    results.to_csv(out_csv, index=False)
    best_path = os.path.join(_ROOT, "physics_best.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best, f)

    print("\n" + "=" * 60)
    print("PHYSICS TUNE RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))
    print(f"\nBest physics_weight = {best['physics_weight']}  "
          f"(val_loss={best['val_loss']:.6f})")
    print(f"Saved -> {out_csv}")
    print(f"Saved -> {best_path}")
    print("Re-run 02_pinn_model.ipynb; it will load physics_best.pkl if present.")
    return results


if __name__ == "__main__":
    tune()
