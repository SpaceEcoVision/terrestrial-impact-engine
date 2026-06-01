"""
LaunchActivityManager — launch cadence (launches/year) per spaceport.

Source: Launch Library 2 (The Space Devs), a free, public, no-key API.
Because LL2 is rate-limited, the engine ships a committed real-data snapshot
(reference/launch_cadence.json) and reads that by default — so it runs offline
and contributors don't all hit the API. Refresh on demand:

    python -m managers.launch_manager --refresh
"""
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("LaunchActivityManager")

LL2_BASE = "https://ll.thespacedevs.com/2.2.0/launch/"
USER_AGENT = "SpaceEcoVision/1.0 (research; +https://spaceecovision.org)"
SNAPSHOT_PATH = Path(__file__).parent.parent / "reference" / "launch_cadence.json"


class LaunchActivityManager:
    def __init__(self, snapshot_path: Path = SNAPSHOT_PATH):
        self.snapshot_path = snapshot_path
        self._snapshot = self._load_snapshot()

    def _load_snapshot(self) -> dict:
        try:
            return json.loads(self.snapshot_path.read_text()).get("sites", {})
        except Exception as e:
            logger.warning(f"No launch snapshot at {self.snapshot_path} ({e})")
            return {}

    def get_cadence(self, site_name: str, years: List[int]) -> Dict[int, int]:
        """Launches per year for a site, from the committed snapshot. Missing years -> 0."""
        site = self._snapshot.get(site_name, {})
        by_year = site.get("by_year", {})
        return {y: int(by_year.get(str(y), 0)) for y in years}

    # --- refresh (hits LL2; rate-limited) ---
    @staticmethod
    def _get(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def refresh(self, targets: List[dict], start: int = 2019, end: int = 2024) -> dict:
        """Re-fetch cadence for the given targets and rewrite the snapshot file."""
        sites = {}
        for t in targets:
            loc = t.get("location_id")
            if not loc:
                continue
            by_year = {str(y): 0 for y in range(start, end + 1)}
            url = (f"{LL2_BASE}?location__ids={loc}&net__gte={start}-01-01"
                   f"&net__lte={end}-12-31&limit=100&mode=list&ordering=net")
            pages = 0
            while url and pages < 8:
                data = self._get(url)
                pages += 1
                for launch in data.get("results", []):
                    yr = (launch.get("net") or "")[:4]
                    if yr in by_year:
                        by_year[yr] += 1
                url = data.get("next")
                time.sleep(2)  # be polite to the rate-limited API
            sites[t["name"]] = {"location_id": loc, "by_year": by_year}
            logger.info(f"{t['name']}: {by_year}")
        out = {
            "_provenance": {
                "source": "Launch Library 2 (The Space Devs)",
                "endpoint": LL2_BASE,
                "metric": "launch attempts per calendar year, by launch location",
            },
            "sites": sites,
        }
        self.snapshot_path.write_text(json.dumps(out, indent=2))
        self._snapshot = sites
        return sites


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import constants

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Re-fetch cadence from LL2")
    args = ap.parse_args()

    mgr = LaunchActivityManager()
    if args.refresh:
        mgr.refresh(constants.TARGETS)
        print(f"Snapshot updated: {SNAPSHOT_PATH}")
    else:
        for t in constants.TARGETS:
            print(f"{t['name']}: {mgr.get_cadence(t['name'], [2019, 2020, 2021, 2022, 2023, 2024])}")
