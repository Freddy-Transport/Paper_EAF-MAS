"""Script import bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
