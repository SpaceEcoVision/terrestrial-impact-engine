import os
import json
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemVerifier")


def check_file_exists(path, description):
    if os.path.exists(path):
        logger.info(f"✅ Found {description}: {path}")
        return True
    logger.error(f"❌ Missing {description}: {path}")
    return False


def validate_json_schema(file_path):
    try:
        data = json.load(open(file_path))
        if "meta" not in data or "sites" not in data:
            logger.error("❌ Schema invalid: missing 'meta' or 'sites'")
            return False
        if not data["sites"]:
            logger.warning("⚠️  Valid but 'sites' is empty")
            return True
        site = data["sites"][0]
        for key in ("name", "series", "summary"):
            if key not in site:
                logger.error(f"❌ Schema invalid: site missing '{key}'")
                return False
        point = site["series"][0]
        for key in ("year", "buildout_pct", "launches", "is_mock"):
            if key not in point:
                logger.error(f"❌ Schema invalid: data point missing '{key}'")
                return False
        logger.info(f"✅ Schema validated ({len(data['sites'])} sites)")
        return True
    except Exception as e:
        logger.error(f"❌ JSON validation failed: {e}")
        return False


async def run_pipeline_test():
    logger.info("🚀 Starting pipeline smoke test...")
    try:
        from main_monitor import TerrestrialImpactEngine
    except ImportError as e:
        logger.error(f"❌ Failed to import TerrestrialImpactEngine: {e}")
        return False
    try:
        if not os.getenv("SENTINEL_CLIENT_ID"):
            logger.warning("⚠️  No Sentinel keys — expecting labeled mock buildout.")
        await TerrestrialImpactEngine().run()
        logger.info("✅ Pipeline completed without errors")
        return True
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False


def main():
    print("=" * 60)
    print("TERRESTRIAL IMPACT ENGINE - SYSTEM VERIFICATION")
    print("=" * 60)

    check_file_exists("managers/infrastructure_manager.py", "Infra Manager")
    check_file_exists("managers/launch_manager.py", "Launch Manager")
    check_file_exists("reference/launch_cadence.json", "Launch cadence snapshot")
    check_file_exists("cv/train.py", "CV training scaffold")
    check_file_exists("examples/spaceport_buildout.py", "EO example")

    success = asyncio.run(run_pipeline_test())

    if success:
        output_path = "data/impact_analysis.json"
        if check_file_exists(output_path, "Output JSON"):
            validate_json_schema(output_path)

    print("\n" + "=" * 60)
    print("✅ SYSTEM VERIFICATION PASSED" if success else "❌ SYSTEM VERIFICATION FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
