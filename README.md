# CHEM-SHIELD AI/ML: Physical-Process Anomaly Detection

This component learns a normal simulated process baseline with one model:
scikit-learn `IsolationForest`. It reports unusual physical behavior only; it
does not determine whether an attack occurred or make operational decisions.

## Telemetry and normal operating ranges

Every training and prediction observation requires finite numeric values for:

| Field | Meaning | Normal-data generator range |
| --- | --- | --- |
| `rpm` | Actual machine speed (the supplied speed measurement) | 0-1500 rpm |
| `target_rpm` | RPM setpoint used to calculate deviation | 0-1500 rpm |
| `temperature` | Process temperature | 0-60 C |
| `power` | Machine power | 1-12 W |
| `pressure` | Process pressure | 1-5 |

The generator creates smooth, varying normal telemetry spanning these ranges.
Those ranges describe generated training data; prediction does not use
hand-written range checks. Isolation Forest makes the anomaly decision from the
learned telemetry distribution. The ranges alone do not specify a real
machine-specific operating profile.

The model uses the same ordered feature extraction for training and inference:

```text
rpm
temperature
pressure
power
rpm_deviation
rpm_rate_of_change
temperature_rate_of_change
pressure_rate_of_change
```

`rpm_deviation = rpm - target_rpm`. Rate-of-change features are the difference
between the current and previous observation (per observation, not per second);
the first sample in an independent stream gets zero. `target_rpm` is an input
used to calculate deviation, not a separate model feature. Missing, nonnumeric,
NaN, and infinite required values are rejected. No scaler is used.

## Train, persist, and predict

Install the dependencies in `requirements.txt`, then run from this directory:

```bash
python3 scripts/generate_normal_data.py
python3 scripts/train_model.py
python3 scripts/run_detection.py --input data/test_scenarios.csv
```

Training rejects fewer than 10 observations and rejects any training file
containing labeled abnormal rows. It trains once on the normal dataset and
saves the Isolation Forest and its feature order/calibration to
`models/isolation_forest.joblib`. Detection loads that artifact and does not
retrain. To retrain, explicitly run `scripts/train_model.py` again.

For continuous input, load a detector once and call `predict` for each
observation. The detector retains the previous RPM, temperature, and pressure
to calculate rates. `reset_stream()` starts a new independent stream:

```python
from src.detector import PhysicalAnomalyDetector

detector = PhysicalAnomalyDetector.load("models/isolation_forest.joblib")
result = detector.predict({
    "rpm": 748.5,
    "target_rpm": 750.0,
    "temperature": 30.2,
    "power": 6.4,
    "pressure": 3.1,
})
print(result)
# {"anomaly_status": "NORMAL", "anomaly_score": <model score>}
```

Each result exposes `anomaly_status` (`NORMAL` or `ANOMALOUS`) and
`anomaly_score`, plus `timestamp` only when supplied. `anomaly_status` maps
Isolation Forest's native prediction (`1` = normal, `-1` = anomalous).
`anomaly_score` is a normalized `[0, 1]` indicator based on negated
`score_samples`, calibrated to the observed normal training-score range; higher
means more anomaly evidence. It is not a probability and does not decide the
status.

## Validation

Run the focused tests with:

```bash
python3 -m pytest tests/test_detector.py
```

The data and scenarios are synthetic laboratory examples, not measurements from
a real machine. A physical anomaly is not an attack classification.
