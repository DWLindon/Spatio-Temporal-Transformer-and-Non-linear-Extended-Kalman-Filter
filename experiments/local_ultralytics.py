from __future__ import annotations

import sys
from pathlib import Path


def use_local_ultralytics() -> Path:
    """Put this repository root before site-packages so custom modules such as STDFM are importable."""
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    return root
