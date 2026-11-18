"""Loading the Online News Popularity data and checking what we actually got."""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV exactly as distributed by UCI.

    Two quirks of this file are handled here:
      * every column name except the first carries a leading space (", timedelta"), which
        would otherwise give us columns called " timedelta";
      * there is no header quoting, so pandas' ``skipinitialspace`` plus an explicit strip
        is the safest combination.
    """
    path = path or config.DATA_RAW
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def row_count_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small table comparing the loaded row count with the UCI page figure."""
    loaded = len(df)
    claimed = config.UCI_PAGE_ROWS_CLAIMED
    return pd.DataFrame(
        {
            "quantity": [
                "rows loaded from CSV",
                "rows quoted on the UCI dataset page",
                "difference",
                "columns loaded",
                "predictive features after dropping url/timedelta",
                "target",
            ],
            "value": [
                loaded,
                claimed,
                claimed - loaded,
                df.shape[1],
                df.shape[1] - 1 - len(config.DROP_COLS),
                config.TARGET,
            ],
        }
    )


def sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Basic integrity checks worth quoting in the report."""
    shares = df[config.TARGET]
    return pd.DataFrame(
        {
            "check": [
                "duplicate rows",
                "missing values",
                "articles with 0 content tokens",
                "shares == 0 (would break log)",
                "min shares",
                "median shares",
                "mean shares",
                "max shares",
                "skew(shares)",
                "skew(log shares)",
            ],
            "value": [
                int(df.duplicated().sum()),
                int(df.isna().sum().sum()),
                int((df["n_tokens_content"] == 0).sum()),
                int((shares <= 0).sum()),
                int(shares.min()),
                float(shares.median()),
                round(float(shares.mean()), 1),
                int(shares.max()),
                round(float(shares.skew()), 2),
                round(float(shares.transform("log").skew()), 2),
            ],
        }
    )
