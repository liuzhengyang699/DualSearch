#!/usr/bin/env python3
"""Deprecated alias for the unified local-only EVQA pipeline."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_search.data.evqa_pipeline import main


if __name__ == "__main__":
    main()
