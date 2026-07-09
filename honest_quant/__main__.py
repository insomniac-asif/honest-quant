"""Enable ``python -m honest_quant`` as an alias for ``python -m honest_quant.run``."""

from __future__ import annotations

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
