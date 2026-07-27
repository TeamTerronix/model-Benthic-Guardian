# Coral Reef Digital Twin

Prototype digital twin for coral-reef monitoring around Sri Lanka. Simulate **3+ underwater temperature sensors**, fill temperatures between them, forecast **+1 / +3 / +7 days**, and score **coral bleaching risk**.

**Prototype note:** Satellite **SST** + **DHW** stand in for seabed loggers. Early notebooks place **3 triangle sensors** with small noise (±0.2°C) to mimic seabed-vs-surface and sensor error. Real logger streams can replace that simulation later.

| Role | Model | Why |
|------|--------|-----|
| Map / fill between sensors | **PINN** | Continuous space–time field + heat PDE |
| Short-term forecast | **ANN–LSTM (60-day history)** | Best +1/+3/+7 skill vs persistence |
| Health alert | **DHW-aware risk scorer** | NOAA-style thresholds (≥4 watch, ≥8 alert) |

---

## Documentation

Full write-up (architectures, results, diagrams):

| File | Description |
|------|-------------|
| **[`Coral_Reef_Digital_Twin_Report.pdf`](Coral_Reef_Digital_Twin_Report.pdf)** | Main report — **start here** |
| [`MODEL_DOCUMENTATION.md`](MODEL_DOCUMENTATION.md) | Same content in Markdown |
| [`Coral_Reef_Digital_Twin_Report.docx`](Coral_Reef_Digital_Twin_Report.docx) | Word version (`python generate_report.py`) |

---

## Pipeline

```
SST + DHW (satellite)
        │
        ▼
Simulate 3+ “seabed” sensors (triangle + noise)   ← prototype only
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

---

## Quick start

```bash
pip install -r requirements.txt
```

### Recommended path (spatial twin + forecasts)

1. **`04_real_spatial_data.ipynb`** — multi-site SST/DHW + time/location hold-outs  
2. **`05_estimate_advection.ipynb`** — estimate coastal `u, v` → `advection.pkl`  
3. **`08_tune_physics.ipynb`** — sweep PDE weight λ → `physics_best.pkl`  
4. **`02_pinn_model.ipynb`** — train interpolating PINN  
5. **`07_improve_forecasts.ipynb`** — bias, recalibration, 6-input forecast head  
6. **`09_lstm_sequence_forecast.ipynb`** — ANN–LSTM vs CNN–ANN–LSTM vs baselines  
7. **`10_lookback_ablation.ipynb`** — history length 7 → 90 days (**60 best**)  
8. **`11_deploy_ann_lstm_forecaster.ipynb`** — operational +1/+3/+7 forecasts + risk  
9. **`06_evaluation.ipynb`** — hold-outs + skill vs persistence/climatology  

**Success bars:** beat persistence at **3–7 days**; beat climatology always; 1-day persistence winning is OK.

### Legacy demo (triangle sensors only)

1. `01_data_preparation.ipynb` — triangle + noise from SST  
2. `02_pinn_model.ipynb` — train PINN  
3. `03_visualization.ipynb` — Folium risk map → `coral_reef_risk_map.html`  

Prefer `04_…` for real multi-site data. Triangle + noise remains the **sensor-simulation** story for the prototype.

---

## Project structure

```
model/
├── Coral_Reef_Digital_Twin_Report.pdf   # Full report (PDF)
├── MODEL_DOCUMENTATION.md
├── generate_report.py                   # Rebuild Word report + diagrams
│
├── dataset/                             # SST / DHW CSVs + triangle_data.csv
│
├── 01_data_preparation.ipynb            # Triangle sensor simulation
├── 02_pinn_model.ipynb                  # PINN spatial twin
├── 03_visualization.ipynb               # Heatmaps / Folium
├── 04_real_spatial_data.ipynb           # Multi-site + hold-outs
├── 05_estimate_advection.ipynb          # u, v from SST gradients
├── 06_evaluation.ipynb                  # Hold-out + forecast skill
├── 07_improve_forecasts.ipynb           # Bias / recal / forecast head
├── 08_tune_physics.ipynb                # λ sweep
├── 09_lstm_sequence_forecast.ipynb      # ANN–LSTM experiments
├── 10_lookback_ablation.ipynb           # Lookback 7–90 days
├── 11_deploy_ann_lstm_forecaster.ipynb  # Deploy forecasts
│
├── pinn_physics.py                      # PDE residual + PhysicsInformedModel
├── prepare_data.py / prepare_forecast_data.py
├── estimate_advection.py / tune_physics.py
├── train_forecast.py / validate_forecast.py
├── recalibrate.py / evaluate_holdout.py / forecaster.py
├── utils.py                             # Bleaching risk helpers
└── requirements.txt
```

---

## How it works

### Prototype sensors

Three virtual sensors in a triangle around a reef; each series = SST + small noise (±0.2°C) to mimic seabed vs surface and sensor error.

```
        Sensor 1
              /\
             /  \
            /____\
     Sensor 3    Sensor 2
```

### PINN (maps)

Input `(lat, lon, time)` → SST. Physics loss is the heat **advection–diffusion** residual via `GradientTape` (not batch smoothness):

```
R = ∂T/∂t + u ∂T/∂lon + v ∂T/∂lat − α ∇²T
Loss = MSE(data) + λ × mean(|R|²)
```

`u, v` come from multi-site SST gradients (`advection.pkl`). Good for filling between sensors; **not** the best short-term forecaster alone.

### ANN–LSTM (forecasts)

Last **60 days** of `[SST, DHW]` → SST & DHW at +1, +3, +7 days.

```
(batch, 60, 2)
  → TimeDistributed Dense(32)
  → LSTM(64)
  → Dense(64) + Dropout
  → Dense(6)   # sst1,dhw1, sst3,dhw3, sst7,dhw7
```

---

## Results (forecast SST MAE °C)

Sequence test set; lower is better:

| Model | 1-day | 3-day | 7-day | Beats persistence @ 3–7d? |
|-------|------:|------:|------:|---------------------------|
| **ANN–LSTM (60d)** | **0.154** | **0.228** | **0.294** | **Yes** |
| CNN–ANN–LSTM | 0.161 | 0.236 | 0.305 | Yes |
| Persistence | 0.156 | 0.240 | 0.320 | — |
| Forecast head (6-in) | 0.202 | 0.284 | 0.364 | No |
| PINN + bias | 0.209 | 0.260 | 0.328 | No |

**Lookback ablation:** 60 days best mean MAE; 90 days slightly worse.

**PINN interpolation (hold-outs):** ~0.73–0.93°C MAE — map task, not day-ahead forecast.

---

## Bleaching risk (DHW-aware)

Not a neural net. Combines DHW (primary), temperature anomaly, warm-stress duration, and warming rate.

| DHW | Level | Meaning |
|-----|-------|---------|
| &lt; 4 | Safe | No significant stress |
| 4–8 | Watch | Possible bleaching |
| &gt; 8 | Alert | Likely / severe bleaching |

Absolute °C alone is not enough — sustained heat (DHW) drives bleaching risk.

---

## Key findings

1. Triangle + SST noise is a valid **prototype** until real seabed loggers exist.  
2. **PINN = map**; **ANN–LSTM = forecast**; **DHW risk = health**.  
3. Bias correction helps PINN forecasts but still loses to persistence.  
4. Snapshot features (6-input head) are not enough — **history** is required.  
5. **60-day ANN–LSTM** is the first approach that beats persistence at 3–7 days.

Details, diagrams, and model-by-model write-ups: **[`Coral_Reef_Digital_Twin_Report.pdf`](Coral_Reef_Digital_Twin_Report.pdf)**.

---

## Dependencies

NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow, Folium, SciPy — see `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## Next steps

1. Replace simulated sensors with real seabed logger streams  
2. Retrain / recalibrate on live data  
3. Alerts for high DHW / risk  
4. Multi-reef dashboards  

---

## License

Educational and research use.

---

## Checklist

- [x] Triangle sensor prototype from SST  
- [x] PINN with real PDE + advection  
- [x] Hold-out evaluation  
- [x] Forecast skill vs persistence  
- [x] ANN–LSTM + lookback ablation (60d)  
- [x] Deploy notebook + full PDF report  
- [ ] Production deploy  
- [ ] Real underwater sensors  

---

**Happy Coral Monitoring**
