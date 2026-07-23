#!/usr/bin/env python3
"""CLI entry point for the staged EVQA/iNaturalist data pipeline."""

import sys
from pathlib import Path


# Allow the documented ``python scripts/data_process/evqa_pipeline.py`` form
# without requiring an editable installation first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_search.data.evqa_pipeline import main


if __name__ == "__main__":
    main()
