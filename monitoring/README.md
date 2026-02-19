# Monitoring

Data drift detection and prediction logging for the DengueMLOps project.

## How it works

The monitoring system has three independent components:

### 1. Prediction logging (runtime)

The FastAPI API automatically logs every prediction to a CSV file:

```text
POST /predict → buffer in memory → flush every 100 predictions → monitoring/predictions_log.csv
```

You can force a flush via `POST /monitoring/flush` or from the Streamlit dashboard.

### 2. Drift detection (offline)

The `DriftDetector` class compares production data against training data (2010-2021) using [Evidently](https://www.evidentlyai.com/). It analyzes all 15 model features for distribution changes.

**Quick demo** (works without the API running):

```bash
# Auto-detect: uses real data if available, synthetic otherwise
python scripts/monitoring_demo.py

# Force synthetic data with artificial drift
python scripts/monitoring_demo.py --simulate

# Synthetic data without drift (baseline comparison)
python scripts/monitoring_demo.py --simulate --no-drift
```

**From API predictions** (requires accumulated predictions):

```bash
python -m src.monitoring.drift_detector
```

Both generate an interactive HTML report in `monitoring/reports/`.

### 3. Visualization (dashboard)

The Streamlit dashboard "Monitoreo" tab shows:

- Prediction log statistics and distribution chart
- Drift reports (HTML viewer with dropdown selector)
- Flush button to write pending predictions to CSV

## File structure

```text
monitoring/
├── predictions_log.csv     # Accumulated API predictions
├── reports/
│   ├── demo_real.html      # Report from real data (if available)
│   ├── demo_synthetic.html # Report from synthetic data
│   └── drift_report_*.html # Reports from API predictions
└── README.md               # This file
```

## Data flow

```text
                    ┌──────────────┐
                    │ Mosqlimate   │
                    │ API data     │
                    └──────┬───────┘
                           │ setup_data.py --all
                           ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│ API /predict│───▶│ predictions  │───▶│ drift_detector.py│
│ (runtime)   │    │ _log.csv     │    │ (offline)        │
└─────────────┘    └──────────────┘    └────────┬─────────┘
                                                │
                           OR                   │
                                                ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│ monitoring  │───▶│ Real/synth   │───▶│ Evidently report │
│ _demo.py    │    │ data         │    │ (HTML)           │
└─────────────┘    └──────────────┘    └────────┬─────────┘
                                                │
                                                ▼
                                       ┌──────────────────┐
                                       │ Streamlit tab    │
                                       │ "Monitoreo"      │
                                       └──────────────────┘
```

## Future improvements

- **Online drift**: Streaming detection with Kafka instead of batch CSV
- **Automated pipeline**: Drift detected → retrain → evaluate → promote
- **Periodic S3 sync**: Keep production data updated automatically
- **Alerts**: Slack/email notifications when drift exceeds threshold
- **Ground truth**: Compare predictions against actual observed alert levels
