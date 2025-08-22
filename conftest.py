"""
Pytest bootstrap for ToxiRAG
Ensures project root is on sys.path for imports like `from ingest...`.
"""

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()


