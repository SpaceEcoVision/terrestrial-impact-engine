"""
Configuration management for the Terrestrial Impact Engine.
Loads settings from environment variables with sensible defaults.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """
    Central configuration class for the application.
    All settings are loaded from environment variables.
    """

    # Environment Settings
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # API Rate Limiting
    API_REQUEST_DELAY = float(os.getenv('API_REQUEST_DELAY', '0.5'))
    API_MAX_RETRIES = int(os.getenv('API_MAX_RETRIES', '3'))
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))

    # Data Directories
    SATELLITE_DATA_DIR = Path(os.getenv('SATELLITE_DATA_DIR', './data'))
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './data'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', './logs'))
    CACHE_DIR = Path(os.getenv('CACHE_DIR', './data/cache'))

    # Ensure directories exist
    SATELLITE_DATA_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    # Cache Settings
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
    CACHE_EXPIRY_HOURS = int(os.getenv('CACHE_EXPIRY_HOURS', '24'))

    # Analysis Settings
    NDBI_THRESHOLD = float(os.getenv('NDBI_THRESHOLD', '0.2'))
    CLOUD_MASK_VALUES = [
        int(x.strip())
        for x in os.getenv('CLOUD_MASK_VALUES', '3,8,9,10,11').split(',')
    ]
    DEFAULT_COUNTRY_CODE = os.getenv('DEFAULT_COUNTRY_CODE', 'USA')
    MAX_YEARS_HISTORY = int(os.getenv('MAX_YEARS_HISTORY', '10'))

    # Mock Data Settings
    ALLOW_MOCK_DATA = os.getenv('ALLOW_MOCK_DATA', 'True').lower() == 'true'

    @staticmethod
    def get_mock_data_values() -> Dict[str, float]:
        """Load mock data values from environment or use defaults."""
        mock_data_env = os.getenv('MOCK_DATA_VALUES')

        if mock_data_env:
            try:
                return json.loads(mock_data_env)
            except json.JSONDecodeError:
                pass

        # Default mock values (spaceport built-up index samples)
        return {
            "Starbase_Boca_Chica": 18.5,
            "Kennedy_Space_Center": 32.1,
            "Vandenberg": 12.3,
            "Spaceport_America": 9.4
        }

    # Statistical Settings
    CORRELATION_STRONG_THRESHOLD = float(os.getenv('CORRELATION_STRONG_THRESHOLD', '0.6'))
    CORRELATION_WEAK_THRESHOLD = float(os.getenv('CORRELATION_WEAK_THRESHOLD', '0.2'))
    MIN_SAMPLE_SIZE = int(os.getenv('MIN_SAMPLE_SIZE', '3'))

    # Advanced Settings
    RASTERIO_ENV_GDAL_CACHEMAX = os.getenv('RASTERIO_ENV_GDAL_CACHEMAX', '512')
    ASYNC_CONCURRENT_REQUESTS = int(os.getenv('ASYNC_CONCURRENT_REQUESTS', '5'))
    OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT', 'json')
    OUTPUT_INDENT = int(os.getenv('OUTPUT_INDENT', '2'))

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return cls.ENVIRONMENT.lower() == 'production'

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode."""
        return cls.ENVIRONMENT.lower() == 'development'

    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        Returns empty list if all valid.
        """
        issues = []

        # Check critical paths exist
        if not cls.SATELLITE_DATA_DIR.exists():
            issues.append(f"Satellite data directory not found: {cls.SATELLITE_DATA_DIR}")

        # Validate NDBI threshold
        if not 0 < cls.NDBI_THRESHOLD < 1:
            issues.append(f"Invalid NDBI_THRESHOLD: {cls.NDBI_THRESHOLD} (must be 0-1)")

        # Validate correlation thresholds
        if cls.CORRELATION_STRONG_THRESHOLD <= cls.CORRELATION_WEAK_THRESHOLD:
            issues.append(
                f"CORRELATION_STRONG_THRESHOLD ({cls.CORRELATION_STRONG_THRESHOLD}) "
                f"must be > CORRELATION_WEAK_THRESHOLD ({cls.CORRELATION_WEAK_THRESHOLD})"
            )

        # Warn about mock data in production
        if cls.is_production() and cls.ALLOW_MOCK_DATA:
            issues.append("WARNING: ALLOW_MOCK_DATA=True in production environment!")

        return issues

    @classmethod
    def print_config(cls):
        """Print current configuration (useful for debugging)."""
        print("=" * 60)
        print("TERRESTRIAL IMPACT ENGINE - CONFIGURATION")
        print("=" * 60)
        print(f"Environment: {cls.ENVIRONMENT}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Mock Data Allowed: {cls.ALLOW_MOCK_DATA}")
        print(f"Cache Enabled: {cls.CACHE_ENABLED}")
        print(f"NDBI Threshold: {cls.NDBI_THRESHOLD}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print("=" * 60)

        # Validation
        issues = cls.validate()
        if issues:
            print("\n⚠️  Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
            print()


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Test configuration loading
    config.print_config()
