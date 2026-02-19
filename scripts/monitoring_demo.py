#!/usr/bin/env python3
"""
Monitoring demo — generates a data drift report using real or simulated data.

Adapts to available data:
  - With historical data (setup_data.py --all/--data): uses real train/test split
  - With current year only (--latest): uses train as reference, current year as production
  - Without data: generates synthetic features to demo the drift detection pipeline

Usage:
  python scripts/monitoring_demo.py              # auto-detect available data
  python scripts/monitoring_demo.py --simulate   # force synthetic data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.constants import ALL_ENGINEERED_FEATURES
from src.monitoring.drift_detector import DriftDetector, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def load_real_data():
    """Try to load real data from parquet files and apply feature engineering."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "historical_api_data"
    parquets = list(raw_dir.rglob("*.parquet")) if raw_dir.exists() else []

    if not parquets:
        return None, None

    print(f"  Found {len(parquets)} parquet files in data/raw/")

    from src.data.mosqlimate_loader import MosqlimateDataLoader
    from src.features.feature_engineer import DengueFeatureEngineer

    data_path = str(PROJECT_ROOT / "data" / "raw" / "historical_api_data")
    loader = MosqlimateDataLoader(data_path=data_path)
    df_raw = loader.load_all_states_data()

    fe = DengueFeatureEngineer()
    df = fe.transform(df_raw)
    df = fe.get_production_dataset(df)

    # Split: reference = train period, current = most recent data
    ref = df[df["year"] <= 2021].copy()
    current = df[df["year"] >= 2023].copy()

    if current.empty:
        # Only old data, use last year available as "production"
        max_year = df["year"].max()
        current = df[df["year"] == max_year].copy()
        ref = df[df["year"] < max_year].copy()

    return ref, current


def generate_synthetic_data(n_reference=5000, n_current=1000, add_drift=True):
    """Generate synthetic data that mimics the feature distributions."""
    rng = np.random.default_rng(42)

    def make_features(n, drift_factor=0.0):
        """Generate n rows of synthetic features with optional drift."""
        data = {}

        # Temporal — sinusoidal features
        angles = rng.uniform(0, 2 * np.pi, n)
        data["month_sin"] = np.sin(angles) + drift_factor * 0.3
        data["month_cos"] = np.cos(angles)
        data["se_sin"] = np.sin(angles * 0.5) + drift_factor * 0.1
        data["se_cos"] = np.cos(angles * 0.5)
        data["is_peak_season"] = rng.integers(0, 2, n)
        data["quarter"] = rng.integers(1, 5, n)

        # Climate — continuous with realistic ranges
        data["tempmed_lag8w"] = rng.normal(25 + drift_factor * 3, 4, n)
        data["tempmed_roll12w"] = rng.normal(24 + drift_factor * 2, 3, n)
        data["umidmed_roll4w"] = rng.normal(70 - drift_factor * 10, 12, n)
        data["temp_x_humid_lag4w"] = (
            data["tempmed_lag8w"] * data["umidmed_roll4w"] / 100
            + rng.normal(0, 1, n)
        )

        # Geo
        data["pop_log"] = rng.normal(11.5, 1.5, n)
        for region in ["region_Nordeste", "region_Centro-Oeste", "region_Sudeste", "region_Sul"]:
            data[region] = rng.integers(0, 2, n)

        return pd.DataFrame(data)

    reference = make_features(n_reference, drift_factor=0.0)
    drift = 1.0 if add_drift else 0.0
    current = make_features(n_current, drift_factor=drift)

    return reference, current


def main():
    parser = argparse.ArgumentParser(
        description="Generate a monitoring drift report demo"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Force synthetic data even if real data is available"
    )
    parser.add_argument(
        "--no-drift", action="store_true",
        help="Generate synthetic data without drift (for comparison)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DengueMLOps — Monitoring Demo")
    print("=" * 60)

    reference = None
    current = None
    data_source = "synthetic"

    if not args.simulate:
        print("\n[1/3] Checking available data...")
        reference, current = load_real_data()
        if reference is not None:
            data_source = "real"
            print(f"  Using real data: {len(reference)} reference, {len(current)} current")

    if reference is None:
        print("\n[1/3] Generating synthetic data...")
        add_drift = not args.no_drift
        reference, current = generate_synthetic_data(add_drift=add_drift)
        drift_label = "with drift" if add_drift else "no drift"
        print(f"  Generated: {len(reference)} reference, {len(current)} current ({drift_label})")

    # Build detector and run
    print("\n[2/3] Running Evidently drift detection...")
    detector = DriftDetector(reference)
    report_name = f"demo_{data_source}"
    results = detector.detect(current, report_name=report_name)

    print(f"  Features analyzed: {results['features_analyzed']}")

    # Also write a sample predictions log so the dashboard has something to show
    print("\n[3/3] Writing sample predictions log...")
    log_path = PROJECT_ROOT / "monitoring" / "predictions_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build log from current data (add timestamp and prediction columns)
    log_df = current[ALL_ENGINEERED_FEATURES].copy()
    log_df["timestamp"] = pd.Timestamp.now().isoformat()
    log_df["predicted_nivel"] = np.random.default_rng(0).integers(1, 5, len(log_df))
    log_df.head(200).to_csv(log_path, index=False)
    print(f"  Wrote {min(200, len(log_df))} entries to {log_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print()
    print(f"  Data source:  {data_source}")
    print(f"  Drift report: {results['report_path']}")
    print()
    print("View the report:")
    print(f"  Open {results['report_path']} in a browser")
    print("  Or start the dashboard: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
