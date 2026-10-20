"""The eval harness and the scripts aren't installed packages, so tests that
import them need those directories on the path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("eval", "scripts"):
    sys.path.insert(0, str(ROOT / folder))
