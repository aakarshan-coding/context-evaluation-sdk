"""`python -m contexteval` is the same entry point as the `contexteval` console script."""

import sys

from ._cli import main

sys.exit(main())
