import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger("InfrastructureManager")
logger.setLevel(getattr(logging, config.LOG_LEVEL))


class MockDataNotAllowedError(Exception):
    """Raised when mock data is requested but not allowed in production."""
    pass


class InfrastructureManager:
    """
    Analyzes infrastructure growth using Sentinel-2 satellite imagery.
    Calculates NDBI (Normalized Difference Built-up Index) to detect construction.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    - High NDBI (>0.2) = Built-up areas (concrete, asphalt)
    - Low NDBI (<0) = Vegetation

    Configuration loaded from environment variables via config module.
    """

    def __init__(self):
        self.ndbi_threshold = config.NDBI_THRESHOLD
        self.cloud_values = config.CLOUD_MASK_VALUES
        self.allow_mock_data = config.ALLOW_MOCK_DATA
        self.is_production = config.is_production()

        logger.info(f"InfrastructureManager initialized (NDBI threshold: {self.ndbi_threshold})")
        if self.is_production and self.allow_mock_data:
            logger.warning("⚠️  WARNING: Mock data allowed in PRODUCTION mode!")

    def calculate_ndbi(self, swir_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Built-up Index.

        Args:
            swir_band: Short-Wave Infrared (Sentinel-2 Band 11)
            nir_band: Near Infrared (Sentinel-2 Band 8)

        Returns:
            NDBI array with values between -1 and 1
        """
        # Convert to float for calculation
        swir = swir_band.astype(float)
        nir = nir_band.astype(float)

        # Calculate NDBI = (SWIR - NIR) / (SWIR + NIR)
        numerator = swir - nir
        denominator = swir + nir

        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ndbi = np.divide(numerator, denominator,
                             out=np.zeros_like(numerator),
                             where=denominator != 0)

        # Clip to valid range [-1, 1]
        ndbi = np.clip(ndbi, -1, 1)

        return ndbi

    def apply_cloud_mask(self, ndbi: np.ndarray, scl_band: Optional[np.ndarray]) -> np.ndarray:
        """
        Mask out clouds, shadows, and water using SCL (Scene Classification Layer).

        SCL Values:
        - 3: Cloud shadows
        - 8, 9: Cloud (medium/high probability)
        - 10: Thin cirrus
        - 11: Snow/Ice
        """
        if scl_band is None:
            logger.warning("No SCL band provided - skipping cloud masking")
            return ndbi

        # Create mask for invalid pixels
        invalid_mask = np.isin(scl_band, self.cloud_values)

        # Set invalid pixels to NaN
        masked_ndbi = ndbi.copy()
        masked_ndbi[invalid_mask] = np.nan

        pixels_masked = np.sum(invalid_mask)
        total_pixels = ndbi.size
        mask_percentage = (pixels_masked / total_pixels) * 100

        logger.info(f"Cloud masking: {pixels_masked:,} pixels masked ({mask_percentage:.1f}%)")

        return masked_ndbi

    def calculate_metrics(self, ndbi: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics from NDBI array.

        Returns:
            Dictionary with construction index, statistics, and interpretation
        """
        # Remove NaN values for calculation
        valid_ndbi = ndbi[~np.isnan(ndbi)]

        if valid_ndbi.size == 0:
            logger.error("No valid pixels after masking!")
            return {
                "construction_index": 0.0,
                "mean_ndbi": 0.0,
                "built_up_percentage": 0.0,
                "status": "No Data"
            }

        # Count pixels above threshold
        built_up_pixels = np.sum(valid_ndbi > self.ndbi_threshold)
        total_valid_pixels = valid_ndbi.size
        built_up_percentage = (built_up_pixels / total_valid_pixels) * 100

        # Calculate construction index (0-100 scale)
        construction_index = built_up_percentage

        # Determine status
        if construction_index > 25:
            status = "High Urban Development"
        elif construction_index > 15:
            status = "Moderate Growth"
        elif construction_index > 5:
            status = "Emerging Development"
        else:
            status = "Rural/Undeveloped"

        return {
            "construction_index": round(construction_index, 2),
            "mean_ndbi": round(float(np.mean(valid_ndbi)), 3),
            "median_ndbi": round(float(np.median(valid_ndbi)), 3),
            "std_ndbi": round(float(np.std(valid_ndbi)), 3),
            "built_up_percentage": round(built_up_percentage, 2),
            "total_pixels_analyzed": int(total_valid_pixels),
            "built_up_pixels": int(built_up_pixels),
            "status": status
        }

    async def analyze_district_growth(
        self,
        region_name: str,
        swir_path: str,
        nir_path: str,
        scl_path: Optional[str] = None,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main analysis function: Processes satellite bands and returns infrastructure metrics.

        Args:
            region_name: Name of district/region being analyzed
            swir_path: Path to SWIR band GeoTIFF (Band 11)
            nir_path: Path to NIR band GeoTIFF (Band 8)
            scl_path: Optional path to Scene Classification Layer

        Returns:
            Dictionary with analysis results

        Raises:
            MockDataNotAllowedError: If in production mode and satellite files missing
        """
        try:
            # Try to import rasterio (geo-spatial library)
            try:
                import rasterio
            except ImportError:
                logger.warning("rasterio not installed — falling back to mock. Run: pip install rasterio")
                return self._handle_missing_data(region_name, "rasterio not installed", year)

            # Validate file existence
            if not os.path.exists(swir_path) or not os.path.exists(nir_path):
                logger.warning(f"Satellite files not found for {region_name}")
                return self._handle_missing_data(
                    region_name,
                    f"Satellite files not found: {swir_path}, {nir_path}",
                    year
                )

            logger.info(f"Analyzing infrastructure for: {region_name}")
            logger.info(f"  SWIR: {swir_path}")
            logger.info(f"  NIR:  {nir_path}")

            # Read satellite bands
            with rasterio.open(swir_path) as swir_src:
                swir_band = swir_src.read(1)
                metadata = {
                    "crs": str(swir_src.crs),
                    "bounds": swir_src.bounds,
                    "resolution": swir_src.res
                }

            with rasterio.open(nir_path) as nir_src:
                nir_band = nir_src.read(1)

            # Read cloud mask if provided
            scl_band = None
            if scl_path and os.path.exists(scl_path):
                with rasterio.open(scl_path) as scl_src:
                    scl_band = scl_src.read(1)
                logger.info("  SCL:  Cloud mask loaded")

            # Validate dimensions
            if swir_band.shape != nir_band.shape:
                logger.error("SWIR and NIR bands have different dimensions!")
                return self._handle_missing_data(region_name, "Band dimension mismatch", year)

            # Calculate NDBI
            logger.info("Calculating NDBI...")
            ndbi = self.calculate_ndbi(swir_band, nir_band)

            # Apply cloud mask
            if scl_band is not None:
                ndbi = self.apply_cloud_mask(ndbi, scl_band)

            # Calculate metrics
            metrics = self.calculate_metrics(ndbi)

            # Build result
            result = {
                "region": region_name,
                "timestamp": datetime.now().isoformat(),
                "construction_index": metrics["construction_index"],
                "status": metrics["status"],
                "metrics": metrics,
                "metadata": metadata,
                "interpretation": self._interpret_results(metrics),
                "is_mock": False  # Real data indicator
            }

            logger.info(f"✓ Analysis complete: {metrics['construction_index']:.1f}% built-up")

            return result

        except Exception as e:
            logger.error(f"Analysis failed for {region_name}: {e}")
            return self._handle_missing_data(region_name, str(e), year)

    def _handle_missing_data(self, region_name: str, error: str,
                             year: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle missing satellite data.

        In production mode: Raises exception
        In development mode: Returns mock data if allowed
        """
        if self.is_production:
            logger.error(f"❌ PRODUCTION MODE: Cannot proceed without real data for {region_name}")
            raise MockDataNotAllowedError(
                f"Satellite data missing for {region_name} and mock data not allowed in production. "
                f"Error: {error}"
            )

        if not self.allow_mock_data:
            logger.error(f"❌ Mock data disabled. Cannot analyze {region_name}")
            raise MockDataNotAllowedError(
                f"Satellite data missing for {region_name} and ALLOW_MOCK_DATA=False. "
                f"Error: {error}"
            )

        logger.warning(f"⚠️  Using mock data for {region_name} (development mode)")
        return self._mock_result(region_name, error, year)

    def _interpret_results(self, metrics: Dict[str, float]) -> str:
        """
        Provide human-readable interpretation of results.
        """
        ci = metrics["construction_index"]
        mean_ndbi = metrics["mean_ndbi"]

        if ci > 25:
            return (f"Highly urbanized area with {ci:.1f}% built-up coverage. "
                   f"Strong infrastructure presence detected.")
        elif ci > 15:
            return (f"Moderate urban development ({ci:.1f}% built-up). "
                   f"Indicates growing construction activity.")
        elif ci > 5:
            return (f"Emerging development with {ci:.1f}% built-up area. "
                   f"Early-stage infrastructure growth.")
        else:
            return (f"Minimal construction detected ({ci:.1f}% built-up). "
                   f"Predominantly rural or undeveloped area.")

    def _mock_result(self, region_name: str, error: Optional[str] = None,
                     year: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate mock data for testing when satellite files unavailable.
        Only called in development mode when ALLOW_MOCK_DATA=True.

        Values are deterministic from (site name, year): the same site+year always
        yields the same number, and they trend upward over time so the downstream
        diagnosis sees a believable construction-growth signal instead of a flat line.
        Always flagged is_mock=True — never to be confused with real measurements.
        """
        logger.info(f"Generating mock data for {region_name}" + (f" ({year})" if year else ""))

        # Get mock values from config (treated as the site's "mature" build-out level)
        mock_values = config.get_mock_data_values()
        name_hash = sum(ord(c) for c in region_name)

        if region_name in mock_values:
            mature = float(mock_values[region_name])
        else:
            # Context-aware base level, stable per site name
            if any(k in region_name for k in ("Space", "Base", "Center", "Port", "Station", "Urban", "Capital", "City")):
                base = 45.0
            elif any(k in region_name for k in ("Launch", "Flight", "Test", "Infrastructure", "Airport")):
                base = 30.0
            elif "Rural" in region_name:
                base = 8.0
            else:
                base = 20.0
            mature = max(0.0, min(100.0, base + (name_hash % 100) / 10.0 - 5.0))

        # Deterministic upward trend toward the mature level (baseline year = 2019)
        if year is not None:
            years_in = max(0, year - 2019)
            jitter = ((name_hash + year) % 7) - 3            # -3..+3, stable per site+year
            mock_index = mature * 0.45 + (mature * 0.11 * years_in) + jitter
        else:
            mock_index = mature
        mock_index = round(max(0.0, min(100.0, mock_index)), 2)

        return {
            "region": region_name,
            "timestamp": datetime.now().isoformat(),
            "construction_index": mock_index,
            "status": "Mock Data (Satellite files not available)",
            "metrics": {
                "construction_index": mock_index,
                "mean_ndbi": 0.15,
                "median_ndbi": 0.12,
                "std_ndbi": 0.18,
                "built_up_percentage": mock_index,
                "total_pixels_analyzed": 1000000,
                "built_up_pixels": int(1000000 * mock_index / 100)
            },
            "metadata": {
                "note": "Using mock data - satellite files not found",
                "error": error,
                "environment": config.ENVIRONMENT
            },
            "interpretation": f"Mock analysis for {region_name}",
            "is_mock": True  # Clearly flag as mock data
        }


# ============================================================================
# TIME SERIES ANALYSIS (Multi-Year Comparison)
# ============================================================================

class TimeSeriesAnalyzer:
    """
    Analyzes infrastructure changes over multiple years.
    Essential for testing the capital-to-buildout hypothesis.
    """

    def __init__(self, infra_manager: InfrastructureManager):
        self.infra_manager = infra_manager

    async def analyze_multi_year(
        self,
        region_name: str,
        year_data: Dict[int, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze same region across multiple years.

        Args:
            region_name: District name
            year_data: {2019: {"swir": "path1", "nir": "path2"}, ...}

        Returns:
            Time series of construction indices
        """
        results = []

        for year in sorted(year_data.keys()):
            paths = year_data[year]

            logger.info(f"Processing {region_name} - Year {year}")

            analysis = await self.infra_manager.analyze_district_growth(
                region_name=f"{region_name}_{year}",
                swir_path=paths.get("swir", ""),
                nir_path=paths.get("nir", ""),
                scl_path=paths.get("scl")
            )

            results.append({
                "year": year,
                "construction_index": analysis["construction_index"],
                "status": analysis["status"],
                "metrics": analysis["metrics"],
                "is_mock": analysis.get("is_mock", False)
            })

        # Calculate growth rate
        if len(results) >= 2:
            first_ci = results[0]["construction_index"]
            last_ci = results[-1]["construction_index"]
            years_span = results[-1]["year"] - results[0]["year"]

            growth_rate = ((last_ci - first_ci) / first_ci * 100) if first_ci > 0 else 0
            annual_growth = growth_rate / years_span if years_span > 0 else 0
        else:
            growth_rate = 0
            annual_growth = 0

        # Check if any mock data was used
        has_mock_data = any(r.get("is_mock", False) for r in results)

        return {
            "region": region_name,
            "time_series": results,
            "summary": {
                "years_analyzed": len(results),
                "initial_year": results[0]["year"] if results else None,
                "final_year": results[-1]["year"] if results else None,
                "initial_construction_index": results[0]["construction_index"] if results else 0,
                "final_construction_index": results[-1]["construction_index"] if results else 0,
                "total_growth_percentage": round(growth_rate, 2),
                "annual_growth_rate": round(annual_growth, 2),
                "contains_mock_data": has_mock_data
            }
        }


# ============================================================================
# TEST BLOCK
# ============================================================================

if __name__ == "__main__":
    async def test_analysis():
        """Test the infrastructure manager"""

        mgr = InfrastructureManager()

        # Test 1: Single year analysis
        print("\n=== TEST 1: Single District Analysis ===")
        try:
            result = await mgr.analyze_district_growth(
                region_name="Starbase (Boca Chica), TX",
                swir_path="data/sentinel_swir.tif",
                nir_path="data/sentinel_nir.tif",
                scl_path="data/sentinel_scl.tif"
            )

            print(f"Region: {result['region']}")
            print(f"Construction Index: {result['construction_index']}%")
            print(f"Status: {result['status']}")
            print(f"Is Mock Data: {result.get('is_mock', False)}")
            print(f"Interpretation: {result['interpretation']}")
        except MockDataNotAllowedError as e:
            print(f"❌ Error: {e}")

        # Test 2: Multi-year analysis
        print("\n=== TEST 2: Time Series Analysis ===")

        time_analyzer = TimeSeriesAnalyzer(mgr)

        # Mock data structure (replace with real paths)
        year_data = {
            2019: {"swir": "data/2019_swir.tif", "nir": "data/2019_nir.tif"},
            2020: {"swir": "data/2020_swir.tif", "nir": "data/2020_nir.tif"},
            2021: {"swir": "data/2021_swir.tif", "nir": "data/2021_nir.tif"},
            2022: {"swir": "data/2022_swir.tif", "nir": "data/2022_nir.tif"},
            2023: {"swir": "data/2023_swir.tif", "nir": "data/2023_nir.tif"}
        }

        try:
            time_series = await time_analyzer.analyze_multi_year(
                region_name="Starbase (Boca Chica), TX",
                year_data=year_data
            )

            print(f"Years analyzed: {time_series['summary']['years_analyzed']}")
            print(f"Growth rate: {time_series['summary']['total_growth_percentage']}%")
            print(f"Annual growth: {time_series['summary']['annual_growth_rate']}% per year")
            print(f"Contains mock data: {time_series['summary']['contains_mock_data']}")

            print("\nYearly breakdown:")
            for entry in time_series['time_series']:
                mock_flag = " [MOCK]" if entry.get('is_mock') else ""
                print(f"  {entry['year']}: {entry['construction_index']:.1f}% ({entry['status']}){mock_flag}")
        except MockDataNotAllowedError as e:
            print(f"❌ Error: {e}")

    # Run tests
    asyncio.run(test_analysis())
