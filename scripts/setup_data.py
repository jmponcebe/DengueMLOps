#!/usr/bin/env python3
"""
Setup script to download required data and model artifacts.

Downloads:
  1. Champion model from GitHub Releases (or uses local if available)
  2. Brazil GeoJSON from IBGE for the Streamlit dashboard
  3. Historical dengue data from Mosqlimate API (optional, requires API key)

Usage:
  python scripts/setup_data.py                      # Download model + GeoJSON
  python scripts/setup_data.py --all                 # Download everything (needs API key)
  python scripts/setup_data.py --latest              # Model + GeoJSON + current year data
  python scripts/setup_data.py --data --from 2023    # Data from 2023 to present
  python scripts/setup_data.py --data --from 2022 --to 2024  # Data for specific range
"""

import argparse
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# GitHub Release config
GITHUB_REPO = "jmponcebe/DengueMLOps"
RELEASE_TAG = "v1.0.0"
MODEL_ASSET_NAME = "champion-model.zip"
MODEL_DEST = PROJECT_ROOT / "mlflow-artifacts" / "models"

# GeoJSON config
IBGE_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?formato=application/vnd.geo+json&intrarregiao=UF"
)
GEOJSON_DEST = PROJECT_ROOT / "data" / "external" / "brazil_uf.geojson"

IBGE_TO_UF = {
    '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA',
    '16': 'AP', '17': 'TO', '21': 'MA', '22': 'PI', '23': 'CE',
    '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE',
    '29': 'BA', '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
    '41': 'PR', '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT',
    '52': 'GO', '53': 'DF',
}


def download_model():
    """Download champion model from GitHub Releases."""
    # Check if model already exists locally
    model_dirs = list(MODEL_DEST.glob("m-*/artifacts/model.xgb"))
    if model_dirs:
        print(f"  Model already exists: {model_dirs[0].parent}")
        return True

    print("  Downloading champion model from GitHub Releases...")
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}"

    try:
        req = Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urlopen(req, timeout=15) as resp:
            release = json.loads(resp.read())

        # Find the model asset
        asset_url = None
        for asset in release.get("assets", []):
            if asset["name"] == MODEL_ASSET_NAME:
                asset_url = asset["browser_download_url"]
                break

        if not asset_url:
            print(f"  Asset '{MODEL_ASSET_NAME}' not found in release {RELEASE_TAG}")
            print("  You can train the model manually by running notebooks 01-03")
            return False

        # Download and extract
        req = Request(asset_url)
        with urlopen(req, timeout=120) as resp:
            data = BytesIO(resp.read())

        MODEL_DEST.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data) as zf:
            zf.extractall(MODEL_DEST)

        print(f"  Model extracted to {MODEL_DEST}")
        return True

    except URLError as e:
        print(f"  Could not download model: {e}")
        print("  You can train the model manually by running notebooks 01-03")
        return False


def download_geojson():
    """Download Brazil UF GeoJSON from IBGE."""
    if GEOJSON_DEST.exists():
        print(f"  GeoJSON already exists: {GEOJSON_DEST}")
        return True

    print("  Downloading Brazil UF boundaries from IBGE API...")
    try:
        with urlopen(IBGE_URL, timeout=30) as resp:
            data = json.loads(resp.read())

        # Add UF abbreviations (sigla) to properties
        for feat in data["features"]:
            cod = feat["properties"]["codarea"]
            feat["properties"]["sigla"] = IBGE_TO_UF.get(cod, cod)

        GEOJSON_DEST.parent.mkdir(parents=True, exist_ok=True)
        with open(GEOJSON_DEST, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        size_kb = GEOJSON_DEST.stat().st_size / 1024
        print(f"  Saved to {GEOJSON_DEST} ({size_kb:.0f} KB)")
        return True

    except URLError as e:
        print(f"  Could not download GeoJSON: {e}")
        print("  The dashboard has a built-in fallback that downloads it at runtime")
        return False


def download_data(year_start=2010, year_end=None):
    """Download historical dengue data from Mosqlimate API."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data.api_client import MosqlimateAPIClient
    except ImportError:
        print("  Could not import api_client. Install dependencies first:")
        print("  pip install -r requirements.txt")
        return False

    api_key = os.environ.get("MOSQLIMATE_API_KEY")
    if not api_key:
        # Try loading from .env
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("MOSQLIMATE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    os.environ["MOSQLIMATE_API_KEY"] = api_key
                    break

    if not api_key:
        print("  Mosqlimate API key not found.")
        print()
        print("  To download historical data:")
        print("  1. Go to https://api.mosqlimate.org/")
        print("  2. Sign in with GitHub")
        print("  3. Copy your UID-Key from your profile")
        print("  4. Add to .env: MOSQLIMATE_API_KEY=your_key_here")
        print("  5. Re-run: python scripts/setup_data.py --all")
        return False

    if year_end is None:
        from datetime import datetime as dt
        year_end = dt.now().year

    period = f"{year_start}-{year_end}" if year_start != year_end else str(year_start)
    print(f"  Downloading data from Mosqlimate API ({period})...")
    if year_start <= 2015:
        print("  This may take 30-60 minutes for the full dataset")

    client = MosqlimateAPIClient()
    client.sync_data(year_start=year_start, year_end=year_end)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download required data and model artifacts for DengueMLOps"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download everything: model + GeoJSON + full historical data (2010-present)"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Download model + GeoJSON + current year data (enough for dashboard map)"
    )
    parser.add_argument(
        "--data", action="store_true",
        help="Download Mosqlimate data (combine with --from/--to for date range)"
    )
    parser.add_argument(
        "--from", type=int, dest="year_from", default=2010, metavar="YEAR",
        help="Start year for data download (default: 2010)"
    )
    parser.add_argument(
        "--to", type=int, dest="year_to", default=None, metavar="YEAR",
        help="End year for data download (default: current year)"
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip model download"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DengueMLOps - Data Setup")
    print("=" * 60)

    # Mode: --data only (just download data for the specified range)
    if args.data and not args.all and not args.latest:
        print(f"\n[1/1] Historical dengue data ({args.year_from}-{args.year_to or 'present'})")
        download_data(year_start=args.year_from, year_end=args.year_to)
        print("\nDone!")
        return

    # Determine what to download
    include_model = not args.no_model
    include_data = args.all or args.latest

    if args.latest:
        from datetime import datetime as dt
        year_start = dt.now().year
        year_end = year_start
    elif args.all:
        year_start = args.year_from
        year_end = args.year_to
    else:
        year_start = args.year_from
        year_end = args.year_to

    steps = []
    if include_model:
        steps.append("model")
    steps.append("geojson")
    if include_data:
        steps.append("data")
    total = len(steps)
    step = 1

    if include_model:
        print(f"\n[{step}/{total}] Champion model")
        download_model()
        step += 1

    print(f"\n[{step}/{total}] Brazil GeoJSON")
    download_geojson()
    step += 1

    if include_data:
        period = f"{year_start}-{year_end or 'present'}"
        print(f"\n[{step}/{total}] Historical dengue data ({period})")
        download_data(year_start=year_start, year_end=year_end)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("Next steps:")
    print("  docker compose up --build    # Run with Docker")
    print("  uvicorn app.api:app          # Or run API locally")
    if not include_data:
        print()
        print("Optional: download data for the dashboard alert map:")
        print("  python scripts/setup_data.py --latest     # Current year only")
        print("  python scripts/setup_data.py --all        # Full dataset (~1.5GB)")
        print("  python scripts/setup_data.py --data --from 2023  # Custom range")
    print("=" * 60)


if __name__ == "__main__":
    main()
