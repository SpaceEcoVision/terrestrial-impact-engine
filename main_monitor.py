"""
Terrestrial Impact Engine — main pipeline.

For each US spaceport, per year:
  - built-up surface % from Sentinel-2 (real via API, or labeled mock)
  - launch cadence from Launch Library 2 (committed snapshot)

It then summarises whether physical buildout leads launch-economy growth.
No credentials needed to run (mock + committed cadence); add Sentinel Hub
credentials for real Earth-observation numbers.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from managers.api_manager import SatelliteAPIManager
from managers.infrastructure_manager import InfrastructureManager
from managers.launch_manager import LaunchActivityManager
from managers.cache_manager import CacheManager
from config import config
import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TerrestrialImpactEngine")

load_dotenv()
CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET")
DASHBOARD_PATH_ENV = os.getenv("DASHBOARD_PATH")


def summarize(series):
    """Summarise a site's (buildout, cadence) time series. Pure + unit-testable.

    'buildout_leads_cadence' is a conservative flag: the site's ground footprint
    grew substantially while early launch cadence was still low, and cadence
    accelerated later — i.e. physical buildout preceded the launch ramp.
    """
    first, last = series[0], series[-1]
    n = len(series)
    half = max(1, n // 2)
    early_cad = sum(s["launches"] for s in series[:half]) / half
    late_cad = sum(s["launches"] for s in series[half:]) / (n - half)
    buildout_delta = round(last["buildout_pct"] - first["buildout_pct"], 1)
    return {
        "buildout_delta_pts": buildout_delta,
        "cadence_delta_launches": last["launches"] - first["launches"],
        "early_avg_cadence": round(early_cad, 1),
        "late_avg_cadence": round(late_cad, 1),
        "buildout_leads_cadence": bool(buildout_delta >= 15 and early_cad < 5 and late_cad > early_cad),
        "contains_mock": any(s.get("is_mock") for s in series),
    }


class TerrestrialImpactEngine:
    def __init__(self):
        self.infra_mgr = InfrastructureManager()
        self.launch_mgr = LaunchActivityManager()
        self.cache_mgr = CacheManager()
        self.cache_mgr.clean_stale()
        if CLIENT_ID and CLIENT_SECRET:
            self.sat_mgr = SatelliteAPIManager(CLIENT_ID, CLIENT_SECRET)
            logger.info("✅ Satellite API initialized (live Earth-observation mode)")
        else:
            logger.warning("⚠️  No Sentinel credentials — using labeled mock buildout data.")
            self.sat_mgr = None

    async def _get_buildout(self, site, year):
        """Return (built_up_pct, is_mock) for a site/year."""
        if self.sat_mgr:
            try:
                val = self.sat_mgr.fetch_urban_index(site["bbox"], f"{year}-01-01", f"{year}-04-30")
                return round(float(val), 2), False
            except Exception as e:
                logger.warning(f"      ⚠️ {site['name']} {year} live API failed: {e}")
        try:
            res = await self.infra_mgr.analyze_district_growth(
                region_name=site["name"],
                swir_path=f"data/{site['name']}_{year}_swir.tif",
                nir_path=f"data/{site['name']}_{year}_nir.tif",
                year=year,
            )
            return round(float(res["metrics"]["construction_index"]), 2), res.get("is_mock", False)
        except Exception as e:
            logger.warning(f"      ⏭️ {site['name']} {year} skipped — no data, mock disabled ({e})")
            return None, None

    async def run(self):
        logger.info("=== STARTING TERRESTRIAL IMPACT SCAN ===")
        end_year = int(os.getenv("END_YEAR", "2024"))
        start_year = max(2019, end_year - config.MAX_YEARS_HISTORY + 1)
        years = list(range(start_year, end_year + 1))

        report = []
        for site in constants.TARGETS:
            logger.info(f"📍 {site['name']} [{site['type']}]")
            cadence = self.launch_mgr.get_cadence(site["name"], years)
            series = []
            for year in years:
                buildout, is_mock = await self._get_buildout(site, year)
                if buildout is None:
                    continue
                series.append({
                    "year": year,
                    "buildout_pct": buildout,
                    "launches": cadence.get(year, 0),
                    "is_mock": is_mock,
                })
            if not series:
                continue
            report.append({
                "name": site["name"],
                "type": site["type"],
                "active": site["active"],
                "location_id": site["location_id"],
                "series": series,
                "summary": summarize(series),
            })

        output = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "country": constants.COUNTRY_NAMES.get("USA", "USA"),
                "years": years,
                "buildout_source": "Copernicus Sentinel-2 L2A",
                "cadence_source": "Launch Library 2 (The Space Devs)",
            },
            "sites": report,
        }

        base = Path(__file__).parent.resolve()
        out_path = base / "data" / "impact_analysis.json"
        os.makedirs(out_path.parent, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"\n💾 Saved: {out_path}")

        if DASHBOARD_PATH_ENV:
            dash = (base / DASHBOARD_PATH_ENV).resolve() if DASHBOARD_PATH_ENV.startswith(".") else Path(DASHBOARD_PATH_ENV)
            if dash.parent.exists():
                dash.write_text(json.dumps(output, indent=2))
                logger.info(f"🚀 Copied output to {dash}")

        logger.info("=== SCAN COMPLETE ===")
        return output


if __name__ == "__main__":
    asyncio.run(TerrestrialImpactEngine().run())
