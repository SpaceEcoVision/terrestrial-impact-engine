"""
Multi-site buildout vs launch-cadence lag analysis.

For each spaceport in TARGETS, this script:
  1. Fetches (or loads cached) Sentinel-2 NDBI built-up % per year via the engine's
     SatelliteAPIManager — same Jan–Apr seasonal window used for Starbase.
  2. Loads launch cadence per year from the committed LL2 snapshot.
  3. Computes year-over-year change in both signals, then cross-correlates them at
     lags 0, +1, +2 years to test whether buildout *leads* launches.
  4. Prints a per-site summary table and writes results/multisite_lag.json.

Run from the repo root:
    python examples/multisite_lag_analysis.py

Flags:
    --skip-fetch   skip Sentinel-2 pulls; use only what's already in data/
    --plot         render a matplotlib figure (requires matplotlib)

Requires valid SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET in .env.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from constants import TARGETS
from managers.api_manager import SatelliteAPIManager
from managers.launch_manager import LaunchActivityManager

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
SEASON_START, SEASON_END = "01-01", "04-30"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_name(site_name: str) -> str:
    """Filesystem-safe slug for a site name."""
    return site_name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def fetch_or_load_buildout(api: SatelliteAPIManager, target: dict, skip_fetch: bool) -> dict:
    """Return {year: pct_built_up} dict, from cache or fresh Sentinel-2 pull."""
    slug = _safe_name(target["name"])
    cache_path = DATA_DIR / f"{slug}_buildout.json"

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        by_year = {int(k): v for k, v in cached.get("by_year", {}).items()}
        missing = [y for y in YEARS if y not in by_year]
        if not missing:
            print(f"  [cache] {target['name']}")
            return by_year

    if skip_fetch:
        print(f"  [skip]  {target['name']} — no cache, --skip-fetch set")
        return {}

    print(f"  [fetch] {target['name']}")
    by_year: dict[int, float] = {}
    for y in YEARS:
        try:
            pct = api.fetch_urban_index(
                target["bbox"],
                f"{y}-{SEASON_START}",
                f"{y}-{SEASON_END}",
            )
            by_year[y] = round(float(pct), 2)
            print(f"    {y}: {by_year[y]:.2f}%")
        except Exception as e:
            print(f"    {y}: error ({e})")

    DATA_DIR.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps({
        "target": target["name"],
        "bbox": target["bbox"],
        "metric": "percent_built_up (NDBI > 0.05; SCL cloud+water masked)",
        "season_window": f"{SEASON_START}..{SEASON_END} each year",
        "source": "Copernicus Sentinel-2 L2A",
        "by_year": {str(k): v for k, v in by_year.items()},
    }, indent=2))
    return by_year


def yoy_changes(by_year: dict, years: list) -> dict:
    """Year-over-year absolute change for each year (key = later year)."""
    changes = {}
    for i in range(1, len(years)):
        y0, y1 = years[i - 1], years[i]
        if y0 in by_year and y1 in by_year:
            changes[y1] = round(by_year[y1] - by_year[y0], 4)
    return changes


def cross_corr(x: dict, y: dict, years: list, lag: int) -> float | None:
    """
    Pearson-like correlation between x (signal A) at year t and y (signal B) at year t+lag.
    Both signals are YoY-change dicts keyed by the *later* year.
    Returns None if fewer than 3 paired observations.
    """
    pairs = []
    for yr in years:
        x_yr = yr
        y_yr = yr + lag
        if x_yr in x and y_yr in y:
            pairs.append((x[x_yr], y[y_yr]))

    n = len(pairs)
    if n < 3:
        return None

    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    denom = (sum((p[0] - mx) ** 2 for p in pairs) * sum((p[1] - my) ** 2 for p in pairs)) ** 0.5
    return round(num / denom, 3) if denom > 0 else None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true", help="Skip Sentinel-2 pulls; use cache only")
    ap.add_argument("--plot", action="store_true", help="Render matplotlib figure")
    args = ap.parse_args()

    cid = os.getenv("SENTINEL_CLIENT_ID", "")
    csec = os.getenv("SENTINEL_CLIENT_SECRET", "")
    if not args.skip_fetch and (not cid or "paste_your" in cid):
        sys.exit("Missing Sentinel credentials in .env — run with --skip-fetch to use cached data only")

    api = SatelliteAPIManager(cid, csec) if not args.skip_fetch else None
    launch_mgr = LaunchActivityManager()

    print("\n=== Fetching buildout data ===")
    results = []
    for target in TARGETS:
        buildout = fetch_or_load_buildout(api, target, args.skip_fetch)
        if not buildout:
            continue

        launches = launch_mgr.get_cadence(target["name"], YEARS)
        bo_yoy = yoy_changes(buildout, YEARS)
        lc_yoy = yoy_changes(launches, YEARS)

        lags = {}
        for lag in [0, 1, 2]:
            # correlation: does buildout change at t predict launch change at t+lag?
            r = cross_corr(bo_yoy, lc_yoy, list(bo_yoy.keys()), lag)
            lags[lag] = r

        best_lag = max((lag for lag in lags if lags[lag] is not None), key=lambda l: abs(lags[l] or 0), default=None)

        results.append({
            "site": target["name"],
            "active": target["active"],
            "buildout_pct": {str(y): buildout.get(y) for y in YEARS},
            "launches": {str(y): launches.get(y, 0) for y in YEARS},
            "buildout_yoy": {str(k): v for k, v in bo_yoy.items()},
            "launches_yoy": {str(k): v for k, v in lc_yoy.items()},
            "lag_correlations": {f"lag_{k}": v for k, v in lags.items()},
            "best_lag_years": best_lag,
        })

    if not results:
        print("No data — run without --skip-fetch to fetch from Sentinel-2.")
        return

    # ── print table ──────────────────────────────────────────────────────────
    print("\n=== Lag analysis: does buildout lead launches? ===")
    print(f"{'Site':<35} {'r(lag=0)':>10} {'r(lag=1)':>10} {'r(lag=2)':>10}  best lag")
    print("-" * 75)
    for r in results:
        lc = r["lag_correlations"]
        print(
            f"{r['site']:<35}"
            f"  {str(lc.get('lag_0', 'n/a')):>8}"
            f"  {str(lc.get('lag_1', 'n/a')):>8}"
            f"  {str(lc.get('lag_2', 'n/a')):>8}"
            f"  {r['best_lag_years'] if r['best_lag_years'] is not None else 'n/a'}"
        )

    print("\n  Positive r at lag > 0: buildout change *precedes* launch-cadence change.")
    print("  Strongest positive lag = how many years ground build leads launch ramp.\n")

    # ── save JSON ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "multisite_lag.json"
    summary = {
        "_meta": {
            "description": "Buildout-leads-launches lag analysis for US spaceports",
            "buildout_source": "Copernicus Sentinel-2 L2A (NDBI > 0.05, Jan-Apr window)",
            "cadence_source": "Launch Library 2 (committed snapshot)",
            "years": YEARS,
            "method": "Pearson cross-correlation of YoY-change signals at lags 0/1/2 years",
        },
        "sites": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {out_path}")

    # ── optional plot ────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            n = len(results)
            fig = plt.figure(figsize=(14, 4 * n))
            gs = gridspec.GridSpec(n, 1, hspace=0.5)

            for i, r in enumerate(results):
                ax = fig.add_subplot(gs[i])
                yrs = [y for y in YEARS if str(y) in r["buildout_pct"] and r["buildout_pct"][str(y)] is not None]
                bo = [r["buildout_pct"][str(y)] for y in yrs]
                lc = [r["launches"][str(y)] for y in yrs]

                ax2 = ax.twinx()
                ax.plot(yrs, bo, "o-", color="#c9a84c", label="Built-up % (Sentinel-2)")
                ax2.bar(yrs, lc, alpha=0.35, color="#4c9ac9", label="Launches (LL2)")
                ax.set_title(r["site"], fontsize=10, fontweight="bold")
                ax.set_ylabel("Built-up %", color="#c9a84c")
                ax2.set_ylabel("Launches/yr", color="#4c9ac9")
                ax.set_xticks(yrs)

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

            fig.suptitle("Spaceport Buildout vs Launch Cadence (2019–2024)", fontsize=13, fontweight="bold")
            plot_path = RESULTS_DIR / "multisite_lag.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {plot_path}")
            plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
