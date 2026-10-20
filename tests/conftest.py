"""The eval harness lives in eval/, which isn't an installed package, so tests
that import it need that directory on the path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
