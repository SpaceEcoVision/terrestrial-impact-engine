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
