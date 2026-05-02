"""Entry point shim — satisfies the AGENTS.md §6.1 contract.

The evaluator resolves the known entry point at ``code/main.py``.
All implementation lives in the root ``main.py`` and the ``code/`` package;
this file simply delegates to the root runner so both invocations work:

    python code/main.py      # evaluator path
    python main.py           # direct / terminal path
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``from code.xxx import ...`` resolves
# whether this file is run as ``python code/main.py`` or ``python -m code.main``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import main  # noqa: E402 — import after path fix

if __name__ == "__main__":
    main()
