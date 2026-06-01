import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger("CacheManager")

class CacheManager:
    """
    Manages file-based caching for API responses and expensive calculations.
    """
    def __init__(self, sub_dir: str = ""):
        self.cache_dir = config.CACHE_DIR / sub_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.CACHE_ENABLED
        self.expiry = timedelta(hours=config.CACHE_EXPIRY_HOURS)

    def generate_key(self, *args) -> str:
        """Generate a safe filename key from arguments using MD5."""
        key_str = "-".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache if it exists and is not expired."""
        if not self.enabled:
            return None
        
        file_path = self.cache_dir / f"{key}.json"
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check expiry
            timestamp = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - timestamp > self.expiry:
                logger.debug(f"Cache expired for {key}")
                return None
                
            logger.debug(f"Cache hit for {key}")
            return data['payload']
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None

    def set(self, key: str, payload: Any):
        """Save data to cache."""
        if not self.enabled:
            return
            
        file_path = self.cache_dir / f"{key}.json"
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'payload': payload
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache for {key}: {e}")

    def clean_stale(self) -> int:
        """Remove expired cache files to free up space."""
        if not self.enabled:
            return 0

        count = 0
        for file_path in self.cache_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                timestamp = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - timestamp > self.expiry:
                    file_path.unlink()
                    count += 1
            except Exception:
                # If file is corrupt or unreadable, delete it
                try:
                    file_path.unlink()
                    count += 1
                except:
                    pass
        
        if count > 0:
            logger.info(f"🧹 Cleaned {count} stale cache files from {self.cache_dir.name}")
        return count
