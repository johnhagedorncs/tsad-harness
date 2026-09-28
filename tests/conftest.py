"""Make ``src/`` importable so the suite runs against a checkout.

The package uses a src layout. Installing it (``pip install -e .``) is the
normal path and is what CI does, but putting ``src`` on ``sys.path`` here means
``pytest`` also works in a bare checkout with no install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
