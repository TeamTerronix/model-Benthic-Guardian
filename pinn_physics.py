"""
Physics-informed training utilities for the coral reef PINN.

Enforces the heat advection–diffusion equation during training:

    ∂T/∂t + u·∂T/∂lon + v·∂T/∂lat − α·(∂²T/∂lat² + ∂²T/∂lon²) ≈ 0

via nested GradientTape residuals (not batch-order smoothness).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Defaults match forecaster.py (normalised coordinate units)
DEFAULT_ALPHA = 0.01
DEFAULT_U_ADV = 0.0
DEFAULT_V_ADV = 0.0


def heat_equation_residual(
    model: keras.Model,
    x: tf.Tensor,
    alpha: float = DEFAULT_ALPHA,
    u_adv: float = DEFAULT_U_ADV,
    v_adv: float = DEFAULT_V_ADV,
    training: bool = False,
) -> tf.Tensor:
    """
    Squared PDE residual for a batch of normalised inputs (N, 3):
    columns = [lat_norm, lon_norm, time_norm].

    Returns shape (N,) of |R|².
    """
    # Identity so GradientTape can watch independent coordinate streams
    lat = tf.identity(x[:, 0:1])
    lon = tf.identity(x[:, 1:2])
    t = tf.identity(x[:, 2:3])

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([lat, lon, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([lat, lon, t])
            inp = tf.concat([lat, lon, t], axis=1)
            T = model(inp, training=training)

        dT_dlat = tape1.gradient(T, lat)
        dT_dlon = tape1.gradient(T, lon)
        dT_dt = tape1.gradient(T, t)

    d2T_dlat2 = tape2.gradient(dT_dlat, lat)
    d2T_dlon2 = tape2.gradient(dT_dlon, lon)
    del tape1, tape2

    def _safe(g, ref):
        return g if g is not None else tf.zeros_like(ref)

    R = (
        _safe(dT_dt, t)
        + u_adv * _safe(dT_dlon, lon)
        + v_adv * _safe(dT_dlat, lat)
        - alpha * (_safe(d2T_dlat2, lat) + _safe(d2T_dlon2, lon))
    )
    return tf.reduce_sum(tf.square(R), axis=1)


def build_pinn_model(
    input_dim: int = 3,
    hidden_layers: Sequence[int] = (128, 128, 64, 64, 32),
    activation: str = "tanh",
) -> keras.Model:
    """MLP PINN: (lat, lon, time) → normalised temperature."""
    inputs = layers.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=activation, name=f"hidden_{i + 1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="PINN")


class PhysicsInformedModel(keras.Model):
    """
    Keras Model wrapper that adds PDE residual loss in train_step / test_step.

    Total loss = data_mse + physics_weight * mean(|R|²)
    """

    def __init__(
        self,
        backbone: keras.Model,
        physics_weight: float = 0.01,
        alpha: float = DEFAULT_ALPHA,
        u_adv: float = DEFAULT_U_ADV,
        v_adv: float = DEFAULT_V_ADV,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.physics_weight = float(physics_weight)
        self.alpha = float(alpha)
        self.u_adv = float(u_adv)
        self.v_adv = float(v_adv)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.data_loss_tracker = keras.metrics.Mean(name="data_loss")
        self.physics_loss_tracker = keras.metrics.Mean(name="physics_loss")
        self.mae_tracker = keras.metrics.MeanAbsoluteError(name="mae")
        self.mse_tracker = keras.metrics.MeanSquaredError(name="mse")

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.data_loss_tracker,
            self.physics_loss_tracker,
            self.mae_tracker,
            self.mse_tracker,
        ]

    def _compute_losses(self, x, y, training: bool):
        y_pred = self.backbone(x, training=training)
        data_loss = tf.reduce_mean(tf.square(y - y_pred))
        phys = heat_equation_residual(
            self.backbone,
            x,
            alpha=self.alpha,
            u_adv=self.u_adv,
            v_adv=self.v_adv,
            training=training,
        )
        physics_loss = tf.reduce_mean(phys)
        total = data_loss + self.physics_weight * physics_loss
        return total, data_loss, physics_loss, y_pred

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total, data_loss, physics_loss, y_pred = self._compute_losses(
                x, y, training=True
            )
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total)
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.mae_tracker.update_state(y, y_pred)
        self.mse_tracker.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        total, data_loss, physics_loss, y_pred = self._compute_losses(
            x, y, training=False
        )
        self.loss_tracker.update_state(total)
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.mae_tracker.update_state(y, y_pred)
        self.mse_tracker.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return {
            "physics_weight": self.physics_weight,
            "alpha": self.alpha,
            "u_adv": self.u_adv,
            "v_adv": self.v_adv,
        }


def wrap_pinn(
    hidden_layers: Sequence[int] = (128, 128, 64, 64, 32),
    physics_weight: float = 0.01,
    alpha: float = DEFAULT_ALPHA,
    u_adv: float = DEFAULT_U_ADV,
    v_adv: float = DEFAULT_V_ADV,
    learning_rate: float = 1e-3,
) -> PhysicsInformedModel:
    """Build + compile a physics-informed PINN ready for .fit()."""
    backbone = build_pinn_model(hidden_layers=hidden_layers)
    model = PhysicsInformedModel(
        backbone,
        physics_weight=physics_weight,
        alpha=alpha,
        u_adv=u_adv,
        v_adv=v_adv,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    # Build weights with a dummy batch
    model(tf.zeros((1, 3)))
    return model
