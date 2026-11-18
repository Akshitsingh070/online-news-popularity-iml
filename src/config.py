"""Shared configuration for the P11 Online News Popularity project."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "OnlineNewsPopularity.csv"
FIG_DIR = ROOT / "reports" / "figures"
RESULT_DIR = ROOT / "reports" / "results"
MODEL_DIR = ROOT / "models"

for _d in (FIG_DIR, RESULT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "shares"

# Columns that are identifiers / leakage, not content signal.
#   url       : a per-article identifier; cannot generalise to an unpublished article
#   timedelta : days since the first article -> encodes publication date, i.e. time-travel
#               information that is unavailable for a *future* article and would let the model
#               memorise events of a specific week.
DROP_COLS = ["url", "timedelta"]

# UCI dataset description page. The CSV that ships with the archive is *not* the
# number quoted on that page; we always report the number of rows actually loaded.
UCI_PAGE_ROWS_CLAIMED = 39_797

FIGURE_DPI = 150
