"""
Geographic + launch targets for the Terrestrial Impact Engine.

Each target is a United States spaceport with:
  - bbox:        Sentinel-2 bounding box for the built-up (NDBI) measurement
  - location_id: Launch Library 2 location id, for launch cadence

The engine compares physical buildout (satellite) against launch cadence (LL2),
testing whether ground buildout leads launch-economy growth.
bbox = [min_lon, min_lat, max_lon, max_lat].
"""

TARGETS = [
    {"name": "Starbase (Boca Chica), TX", "code": "USA", "type": "Spaceport", "active": True,
     "bbox": [-97.18, 25.985, -97.145, 26.005], "location_id": 143},
    {"name": "Cape Canaveral SFS, FL", "code": "USA", "type": "Spaceport", "active": True,
     "bbox": [-80.60, 28.45, -80.52, 28.62], "location_id": 12},
    {"name": "Kennedy Space Center, FL", "code": "USA", "type": "Spaceport", "active": True,
     "bbox": [-80.65, 28.58, -80.58, 28.65], "location_id": 27},
    {"name": "Vandenberg SFB, CA", "code": "USA", "type": "Spaceport", "active": True,
     "bbox": [-120.65, 34.70, -120.55, 34.78], "location_id": 11},
    {"name": "Wallops Flight Facility, VA", "code": "USA", "type": "Launch site", "active": True,
     "bbox": [-75.52, 37.91, -75.44, 37.97], "location_id": 21},
    {"name": "Spaceport America, NM", "code": "USA", "type": "Spaceport", "active": False,
     "bbox": [-107.00, 32.97, -106.94, 33.01], "location_id": 31},
]

# Human-readable country names for logs/output.
COUNTRY_NAMES = {"USA": "United States"}
