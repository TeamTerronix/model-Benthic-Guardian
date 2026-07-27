# Coral Reef Digital Twin — Model Documentation

**Project:** Prototype digital twin for coral-reef monitoring around Sri Lanka  
**Goal:** Simulate **3+ underwater temperature sensors**, estimate temperatures between them, forecast short-term change, and score **coral bleaching risk**.

**Prototype note:** Satellite **SST** (sea surface temperature) is used as a stand-in for seabed loggers. Early models place **3 triangle sensors** with small noise to mimic seabed-vs-surface and sensor differences. The end architecture is designed so real logger streams can replace that simulation later.

---

## 1. Prototype pipeline (recommended)

```
SST + DHW (satellite)
        │
        ▼
Simulate 3+ “seabed” sensors (triangle layout + small noise)   ← prototype only
        │
        ├──────────────────────────────┐
        ▼                              ▼
   PINN (spatial twin)          ANN–LSTM @ 60 days
   (lat, lon, time) → T         (60-day SST/DHW history)
        │                              │
        ▼                              ▼
   Heatmap / area fill           +1 / +3 / +7 day forecast
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Bleaching risk (DHW-aware)
```

| Role | Model | Why |
|------|--------|-----|
| Map / fill between sensors | **Physics-Informed NN (PINN)** | Continuous field in space–time |
| Short-term forecast | **ANN–LSTM (60-day history)** | Best +1/+3/+7 skill in experiments |
| Health alert | **DHW-aware risk scorer** | NOAA-style stress thresholds |

---

## 2. Evolution overview

| # | Model | Main idea | Status |
|---|--------|-----------|--------|
| 0 | Persistence | “Tomorrow = today” | Strong baseline |
| 1 | Original PINN (triangle + noise) | First spatial prototype | Superseded as sole forecaster |
| 2 | Improved PINN (real PDE + advection) | Better physics & data | **Keep for maps** |
| 3 | PINN + bias / recalibration | Fix warm drift for forecasts | Intermediate |
| 4 | 6-input forecast head | Issue-time snapshot → future | Beaten by LSTM |
| 5 | ANN–LSTM / CNN–ANN–LSTM | Learn from history | **ANN–LSTM wins** |
| 6 | Lookback ablation | Find best history length | **60 days best** |
| 7 | Deploy ANN–LSTM | Operational forecast notebook | Current deploy path |

---

## 3. Model-by-model description

### 3.0 Persistence (baseline)

**What it is:** Copy the last known SST forward for every horizon.

**Architecture:** None (rule-based).

**Why keep it:** Ocean SST changes slowly. Any learned model must beat this at **3–7 days** to be useful.

**Typical test MAE (°C)** (sequence experiment, n=1680):

| 1-day | 3-day | 7-day |
|------:|------:|------:|
| 0.156 | 0.240 | 0.320 |

---

### 3.1 Original PINN (first model)

**Intent:** Prototype “3 underwater sensors in a triangle” using SST.

**Data approach:**
- One SST series  
- Three points around a site (triangle offsets)  
- Random noise (~±0.2°C) to mimic seabed vs surface / sensor noise  

**Architecture:**

```
Input  (3):   [lat_norm, lon_norm, time_norm]
Hidden:       Dense(128, tanh) + BatchNorm
              Dense(128, tanh) + BatchNorm
              Dense(64,  tanh) + BatchNorm
              Dense(64,  tanh) + BatchNorm
              Dense(32,  tanh) + BatchNorm
Output (1):   Dense(1, sigmoid) → normalised SST
```

**Parameters:** ~33,217  

**Early “physics” loss (weak):**  
`MSE(data) + λ × mean((ŷ[i+1] − ŷ[i])²)` — batch-order smoothness, **not** a real PDE.

**Improvements needed (and later done):**
- Real PDE residual via `GradientTape`  
- Real multi-site data / honest hold-outs  
- Separate forecasting from interpolation  

**Role today:** Conceptual ancestor; triangle + noise still valid as **sensor simulation** for the prototype story.

---

### 3.2 Improved PINN (spatial digital twin)

**Notebooks / code:** `02_pinn_model.ipynb`, `pinn_physics.py`, `04_real_spatial_data.ipynb`, `05_estimate_advection.ipynb`

**What changed:**
- Optional training on **5 real satellite sites** (Hikkaduwa, Kalpitiya, Passikudah, South East, Trincomalee)  
- Time and location **hold-outs** (not shuffled leakage)  
- Physics loss = heat **advection–diffusion** residual:

\[
\frac{\partial T}{\partial t} + u\frac{\partial T}{\partial lon} + v\frac{\partial T}{\partial lat}
- \alpha\nabla^2 T \approx 0
\]

- `u, v` estimated from multi-site SST gradients (`advection.pkl`)  
- Total loss: `data MSE + λ × mean(|R|²)`  

**Architecture:** Same MLP as above (`[128,128,64,64,32]`, tanh, BN, sigmoid). Wrapped as `PhysicsInformedModel` with custom `train_step`.

**Interpolation performance (physical °C):**

| Split | MAE | RMSE | Notes |
|-------|----:|-----:|-------|
| Validation (time) | **0.73** | 0.93 | In-period reconstruction |
| Test (newest years) | **0.93** | 1.17 | Warm bias ~+0.55°C |
| Location hold-out (Trincomalee) | **0.75** | 0.96 | Spatial transfer OK |

Normalised val MAE ≈ 0.098.

**Finding:** Good enough as a **map / fill-between-sensors** model. **Not** competitive as a raw short-term forecaster vs persistence.

---

### 3.3 PINN + bias / recalibration

**Code:** `forecaster.py`, `recalibrate.py`, `07_improve_forecasts.ipynb`

**Idea:**
- Predict with PINN at future time  
- Add **bias** = mean(actual − PINN) over recent days at issue time  
- Optional **recalibration** offsets from validation residuals  

**Forecast MAE after bias (°C)** (interpolating PINN):

| Horizon | PINN raw | PINN+bias | Persistence |
|--------:|---------:|----------:|------------:|
| 1-day | 0.87 | **0.22** | 0.15 |
| 3-day | 0.87 | **0.28** | 0.24 |
| 7-day | 0.87 | **0.37** | 0.34 |

**Finding:** Bias correction is essential if you use PINN for forecasts, but the model still **loses to persistence**. Beats climatology after bias.

---

### 3.4 Six-input forecast head

**Notebook / code:** `07_improve_forecasts.ipynb`, `prepare_forecast_data.py`, `train_forecast.py`  
**Artefact:** `pinn_forecast_best.h5`

**Architecture:**

```
Input  (6):  [lat, lon, time_target, horizon_norm, SST_issue_norm, DHW_issue_norm]
Hidden:      Dense(128,tanh)+BN → 128 → 64 → 64 → 32
Output (2):  [SST_norm, DHW_norm]   (sigmoid)
```

Same width as the PINN backbone, but **supervised MSE** (no PDE on 6-D inputs). Multi-horizon training via `horizon` feature.

**Test MAE (°C)** vs persistence:

| Horizon | Forecast head | Persistence |
|--------:|--------------:|------------:|
| 1-day | 0.202 | 0.154 |
| 3-day | 0.286 | 0.244 |
| 7-day | 0.371 | 0.343 |

**Finding:** Much better than raw PINN as a forecaster, still behind persistence. Issue-time **snapshot is not enough** — history is needed.

---

### 3.5 ANN–LSTM (winning forecaster)

**Notebooks:** `09_lstm_sequence_forecast.ipynb`, `11_deploy_ann_lstm_forecaster.ipynb`  
**Artefacts:** `lstm_experiment/ann_lstm_best.h5`, `lstm_lookback_ablation/ann_lstm_L60_best.h5`

**Input:** Last **L** days of `[SST_norm, DHW_norm]` per site → shape `(L, 2)`  
**Output:** SST & DHW at +1, +3, +7 → shape `(6,)`

**Architecture:**

```
Input:     (batch, LOOKBACK, 2)
           │
TimeDistributed Dense(32, relu)     ← ANN feature mix per day
           │
LSTM(64)                            ← sequence memory
           │
Dense(64, relu) + Dropout(0.2)
           │
Dense(6, sigmoid)                   ← [sst1,dhw1, sst3,dhw3, sst7,dhw7]
```

**Parameters:** ~29.5k (LOOKBACK=60)

**Sequence experiment results (LOOKBACK=60, n=1680):**

| Model | 1-day | 3-day | 7-day |
|--------|------:|------:|------:|
| **ANN–LSTM** | **0.154** | **0.228** | **0.294** |
| Persistence | 0.156 | 0.240 | 0.320 |
| CNN–ANN–LSTM | 0.161 | 0.236 | 0.305 |
| Forecast head | 0.202 | 0.284 | 0.364 |
| PINN + bias | 0.209 | 0.260 | 0.328 |

Skill vs persistence (ANN–LSTM): about **+0.02 / +0.05 / +0.08** at 1 / 3 / 7 days.

**Finding:** Historical sequence learning **does** improve short-term forecasts. ANN–LSTM is the best forecaster in this project.

---

### 3.6 CNN–ANN–LSTM

**Same as ANN–LSTM**, plus 1D convolutions before the dense/LSTM stack:

```
Input (LOOKBACK, 2)
  → Conv1D(32, k=5) → Conv1D(32, k=3)
  → TimeDistributed Dense(32)
  → LSTM(64) → Dense(64)+Dropout → Dense(6)
```

**Parameters:** ~33.9k  

**Finding:** Close second to ANN–LSTM; CNN did not help enough on this daily SST signal (slow, smooth). Prefer the simpler ANN–LSTM.

---

### 3.7 Lookback ablation (history length)

**Notebook:** `10_lookback_ablation.ipynb`  

Same ANN–LSTM, lookbacks **7 / 14 / 30 / 60 / 90** days.

| Lookback | MAE 1d | MAE 3d | MAE 7d | Mean |
|---------:|-------:|-------:|-------:|-----:|
| 7 | 0.158 | 0.237 | 0.317 | 0.237 |
| 14 | 0.155 | 0.236 | 0.320 | 0.237 |
| 30 | 0.161 | 0.239 | 0.317 | 0.239 |
| **60** | **0.155** | **0.229** | **0.296** | **0.227** |
| 90 | 0.160 | 0.232 | 0.298 | 0.230 |

**Finding:** **60 days is the sweet spot.** Shorter windows underperform at 7-day lead; 90 days adds little and is slightly worse overall.

---

### 3.8 Deploy ANN–LSTM

**Notebook:** `11_deploy_ann_lstm_forecaster.ipynb`  

Loads `ann_lstm_L60_best.h5`, takes last 60 days for a reef, outputs +1/+3/+7 SST/DHW, persistence comparison, and DHW-aware bleaching risk. Outputs under `lstm_deploy/`.

---

## 4. Bleaching risk layer

**Code:** `utils.calculate_bleaching_risk`

Not a neural net. Combines:
- **DHW** (primary; NOAA-style: ≥4 warning, ≥8 danger)  
- Temperature anomaly vs monthly baseline  
- Duration of warm stress  
- Warming rate  

Used on top of PINN maps and LSTM forecasts.

---

## 5. Side-by-side comparison (forecasting)

SST MAE (°C), lower is better — sequence test set:

| Rank | Model | 1d | 3d | 7d | Beats persistence @ 3–7d? |
|-----:|--------|---:|---:|---:|---------------------------|
| 1 | **ANN–LSTM (60d)** | 0.154 | 0.228 | 0.294 | **Yes** |
| 2 | CNN–ANN–LSTM | 0.161 | 0.236 | 0.305 | Yes |
| 3 | Persistence | 0.156 | 0.240 | 0.320 | — |
| 4 | Forecast head (6-in) | 0.202 | 0.284 | 0.364 | No |
| 5 | PINN + bias | 0.209 | 0.260 | 0.328 | No |

**Interpolation (PINN alone):** ~0.7–0.9°C MAE on hold-outs — different task (maps, not day-ahead forecast).

---

## 6. What we found

1. **Triangle + SST noise** is a valid **prototype** of seabed sensors until loggers exist.  
2. **PINN** is right for **spatial** filling; wrong as the only **forecast** engine.  
3. Real PDE residual + advection + hold-outs made the twin more honest, not magically a better 1-day forecaster.  
4. **Bias correction** hugely helps PINN forecasts but still loses to persistence.  
5. A **snapshot forecast head** is better than raw PINN, still not enough.  
6. **60 days of history + ANN–LSTM** is the first approach that **beats persistence** at 3–7 days (and slightly at 1 day).  
7. **60 > 90** for lookback: more history is not always better.  
8. Split the product: **PINN = map**, **ANN–LSTM = forecast**, **DHW risk = health**.

---

## 7. Architecture cheat-sheet

| Model | Input | Core | Output | ~Params |
|--------|--------|------|--------|--------:|
| PINN | `(lat, lon, t)` | MLP 128×2–64×2–32 + PDE loss | SST | 33k |
| Forecast head | 6 snapshot features | Same MLP width, MSE | SST+DHW | ~33k |
| ANN–LSTM | `(60, 2)` SST/DHW | TD-Dense → LSTM → Dense | 6 (3 horizons × 2) | 29k |
| CNN–ANN–LSTM | `(60, 2)` | Conv1D×2 → TD-Dense → LSTM → Dense | 6 | 34k |
| Persistence | last SST | — | same SST | 0 |
| Risk | T, DHW, baseline | Weighted rules | score 0–1, level 0–2 | 0 |

---

## 8. Key notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_preparation.ipynb` | Triangle sensor simulation from SST |
| `02_pinn_model.ipynb` | Train PINN |
| `03_visualization.ipynb` | Risk maps |
| `04_real_spatial_data.ipynb` | Real multi-site data + hold-outs |
| `05_estimate_advection.ipynb` | Estimate `u, v` |
| `06_evaluation.ipynb` | Hold-out + forecast skill bars |
| `07_improve_forecasts.ipynb` | Bias, recal, 6-input head |
| `08_tune_physics.ipynb` | Sweep physics weight λ |
| `09_lstm_sequence_forecast.ipynb` | ANN–LSTM vs CNN–ANN–LSTM |
| `10_lookback_ablation.ipynb` | History length 7→90 |
| `11_deploy_ann_lstm_forecaster.ipynb` | Run 60-day ANN–LSTM forecasts |

---

## 9. One-sentence summary

We started with a **triangle-sensor PINN prototype** on noisy SST to mimic seabed loggers, improved physics and evaluation for a **spatial digital twin**, then showed that **short-term coral-temperature forecasting needs sequence memory** — culminating in an **ANN–LSTM with 60 days of SST/DHW history** as the best forecast model, while the **PINN remains the map engine**.
