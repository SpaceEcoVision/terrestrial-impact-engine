"""
Example: measure the physical buildout of a spaceport from Sentinel-2 imagery.

Computes the share of built-up surface (NDBI > 0.05) inside a bounding box for
each year, using the engine's SatelliteAPIManager. Real Copernicus data — needs
valid SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET in .env.

Run from the repo root:  python examples/spaceport_buildout.py
"""
import os
import sys
import json

# make the repo root importable regardless of cwd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from managers.api_manager import SatelliteAPIManager

TARGET = "Starbase (Boca Chica), TX"
# Tightened to the build site + launch pad, trimming open Gulf water to the east.
BBOX = [-97.18, 25.985, -97.145, 26.005]   # SpaceX Starbase (land-focused)
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
# Fixed seasonal window each year (Jan–Apr) so we compare like-with-like across years.
SEASON_START, SEASON_END = "01-01", "04-30"


def main():
    cid = os.getenv("SENTINEL_CLIENT_ID")
    csec = os.getenv("SENTINEL_CLIENT_SECRET")
    if not cid or not csec or "paste_your" in cid:
        sys.exit("Missing Sentinel credentials in .env")

    api = SatelliteAPIManager(cid, csec)
    by_year = {}
    print(f"Target: {TARGET}  bbox={BBOX}")
    for y in YEARS:
        try:
            pct = api.fetch_urban_index(BBOX, f"{y}-{SEASON_START}", f"{y}-{SEASON_END}")
            by_year[str(y)] = round(float(pct), 2)
            print(f"  {y}: built-up {by_year[str(y)]:.2f}%")
        except Exception as e:
            print(f"  {y}: failed ({e})")

    out = {
        "target": TARGET,
        "bbox": BBOX,
        "metric": "percent_built_up (NDBI > 0.05; SCL cloud+water masked)",
        "season_window": f"{SEASON_START}..{SEASON_END} each year",
        "source": "Copernicus Sentinel-2 L2A",
        "by_year": by_year,
    }
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out_path = os.path.join(ROOT, "data", "starbase_buildout.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
