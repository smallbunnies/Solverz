import os
import tempfile
from pathlib import Path


TEMP_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("NUMBA_CACHE_DIR", str(TEMP_ROOT / "numba-cache"))
