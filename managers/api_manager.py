import logging
import numpy as np
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS
)
from managers.cache_manager import CacheManager

# Configure Logger
logger = logging.getLogger("SatelliteAPIManager")

class SatelliteAPIManager:
    def __init__(self, client_id, client_secret):
        self.config = SHConfig()
        self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        
        # Initialize Cache
        self.cache = CacheManager("satellite")

    def fetch_urban_index(self, bbox_coords, start_date, end_date):
        # Check Cache
        cache_key = self.cache.generate_key("ndbi", bbox_coords, start_date, end_date)
        cached_val = self.cache.get(cache_key)
        if cached_val is not None:
            logger.info(f"      ✓ [CACHE] Loaded satellite data")
            return cached_val

        # Evalscript calculates NDBI (Normalized Difference Built-up Index)
        # Formula: (SWIR - NIR) / (SWIR + NIR)
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B11", "B8A", "SCL"],
            output: { bands: 3, sampleType: "FLOAT32" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.B11, sample.B8A, sample.SCL];
        }
        """

        copernicus_l2a = DataCollection.define_from(
            DataCollection.SENTINEL2_L2A,
            "COPERNICUS_L2A",
            service_url="https://sh.dataspace.copernicus.eu"
        )

        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=copernicus_l2a,
                    time_interval=(start_date, end_date),
                    mosaicking_order="leastCC",
                    # CRITICAL FIX: Harmonize old data with new 2022+ baseline
                    other_args={"processing": {"harmonizeValues": True}}
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox,
            size=(512, 512),
            config=self.config
        )

        try:
            data = request.get_data()[0]
        except Exception as e:
            logger.error(f"API Request Failed: {e}")
            raise e

        swir = data[:, :, 0]
        nir = data[:, :, 1]
        scl = data[:, :, 2]

        # Calculate NDBI
        ndbi = (swir - nir) / (swir + nir + 0.0001)

        # SCL Filter: 0=No Data, 1=Defect, 3=Cloud Shadow, 6=Water, 8-11=Cloud/Snow.
        # Water is excluded so coastal tides / open sea do not skew the built-up fraction.
        bad_data_mask = (scl == 0) | (scl == 1) | (scl == 3) | (scl == 6) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
        valid_pixels = ~bad_data_mask

        # NDBI > 0.05 usually indicates concrete/asphalt
        built_up_pixels = (ndbi > 0.05) & valid_pixels

        valid_count = np.sum(valid_pixels)
        if valid_count == 0:
            result = 0.0
        else:
            result = (np.sum(built_up_pixels) / valid_count) * 100
            
        # Save to Cache
        self.cache.set(cache_key, float(result))

        return result

    # ------------------------------------------------------------------
    # CV training data: raw multi-band tiles + WorldCover label masks
    # ------------------------------------------------------------------
    # fetch_urban_index() returns a single built-up percentage (the v1 NDBI
    # indicator). The CV model needs the underlying imagery and a per-pixel
    # label. The two methods below provide exactly that:
    #   - fetch_band_stack:        the model INPUT  (multi-band Sentinel-2 tile)
    #   - fetch_worldcover_builtup: the model TARGET (ESA WorldCover built-up mask)

    # Default bands: RGB (B02/B03/B04) for context, NIR (B08) and SWIR (B11/B12)
    # which carry most of the built-up / impervious-surface signal.
    DEFAULT_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

    def fetch_band_stack(self, bbox_coords, start_date, end_date,
                         size=512, bands=DEFAULT_BANDS):
        """Fetch a multi-band Sentinel-2 L2A tile as the CV model input.

        Returns (stack, valid) where:
          - stack: float32 (H, W, C) reflectance, one channel per band (0..1)
          - valid: bool   (H, W) — False where SCL marks cloud/shadow/water/no-data

        Like fetch_urban_index, this uses leastCC mosaicking over the window and
        harmonizes pre-2022 values to the new baseline.
        """
        band_list = list(bands)
        inputs = band_list + ["SCL"]
        out_bands = len(inputs)
        # Build the evalscript dynamically so callers can pick the band set.
        sample_fields = ", ".join(f"sample.{b}" for b in inputs)
        evalscript = f"""
        //VERSION=3
        function setup() {{
          return {{
            input: {inputs!r},
            output: {{ bands: {out_bands}, sampleType: "FLOAT32" }}
          }};
        }}
        function evaluatePixel(sample) {{
          return [{sample_fields}];
        }}
        """.replace("'", '"')

        copernicus_l2a = DataCollection.define_from(
            DataCollection.SENTINEL2_L2A,
            "COPERNICUS_L2A",
            service_url="https://sh.dataspace.copernicus.eu"
        )
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=copernicus_l2a,
                    time_interval=(start_date, end_date),
                    mosaicking_order="leastCC",
                    other_args={"processing": {"harmonizeValues": True}}
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=(size, size),
            config=self.config
        )
        data = request.get_data()[0]                 # (H, W, out_bands)
        stack = data[:, :, :len(band_list)].astype("float32")
        scl = data[:, :, len(band_list)]
        # SCL: 0=NoData 1=Defect 3=CloudShadow 6=Water 8/9/10=Cloud 11=Snow
        bad = np.isin(scl, [0, 1, 3, 6, 8, 9, 10, 11])
        valid = ~bad
        return stack, valid

    def fetch_worldcover_builtup(self, bbox_coords, size=512, year=2021):
        """Fetch the ESA WorldCover built-up label mask for the bbox.

        Returns float32 (H, W) in {0, 1}: 1 where WorldCover class == 50
        (built-up). Read directly from the public ESA WorldCover COGs on AWS
        (no auth), windowed to the bbox and resampled to `size` with nearest
        neighbour so it aligns pixel-for-pixel with fetch_band_stack output.

        WorldCover has two epochs only: 2020 (v100) and 2021 (v200).
        """
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds

        version = {2020: "v100", 2021: "v200"}.get(year)
        if version is None:
            raise ValueError("WorldCover year must be 2020 or 2021")

        min_lon, min_lat, max_lon, max_lat = bbox_coords
        # WorldCover tiles are 3x3 deg, named by SW corner floored to a multiple of 3.
        tile_lat = int(np.floor(min_lat / 3.0) * 3)
        tile_lon = int(np.floor(min_lon / 3.0) * 3)
        ns = f"N{tile_lat:02d}" if tile_lat >= 0 else f"S{-tile_lat:02d}"
        ew = f"E{tile_lon:03d}" if tile_lon >= 0 else f"W{-tile_lon:03d}"
        tile = f"{ns}{ew}"
        url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
               f"{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")

        dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, size, size)
        dst = np.zeros((size, size), dtype="uint8")
        with rasterio.open(f"/vsicurl/{url}") as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
            )
        return (dst == 50).astype("float32")
