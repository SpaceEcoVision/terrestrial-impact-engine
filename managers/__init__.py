"""
Terrestrial Impact Engine - Managers Package

Core logic managers:
1. Satellite API interaction (Sentinel Hub / Copernicus)
2. Infrastructure analysis (NDBI built-up from Sentinel-2)
3. Launch activity (cadence from Launch Library 2)
"""

from .api_manager import SatelliteAPIManager
from .infrastructure_manager import InfrastructureManager, TimeSeriesAnalyzer
from .launch_manager import LaunchActivityManager

__all__ = [
    'SatelliteAPIManager',
    'InfrastructureManager',
    'TimeSeriesAnalyzer',
    'LaunchActivityManager',
]
