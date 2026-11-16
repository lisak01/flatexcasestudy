"""The common interface every data source implements.

Every adapter returns a DataFrame with exactly two columns:
    rate_date (datetime64, one row per day)
    rate      (float)

The ETL only talks to this interface. A new source - another REST API, a file in a
cloud bucket, a legacy file share, a database - gets its own small adapter class.
Quality checks, loading and the load log are reused unchanged.
"""

# allows type hints like "str | None" (only for hints)
from __future__ import annotations

# tools to define a "template" class that others must follow
from abc import ABC, abstractmethod

# tables with data
import pandas as pd


class SourceError(RuntimeError):
    """Our own error type: the source could not deliver data (network, driver, format)."""


class SourceAdapter(ABC):
    """Template for all sources: every adapter must have a fetch() method."""

    # name of the source; written to the load log, so every row can be traced to its source
    name: str = "source"

    # every adapter MUST write its own version of this method
    @abstractmethod
    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        """Return all rates from start_date on, as a table (rate_date, rate)."""


def empty_frame() -> pd.DataFrame:
    """Empty table with the right columns (used when the source has no data)."""
    # two empty columns: date and rate
    return pd.DataFrame({"rate_date": pd.Series(dtype="datetime64[ns]"),
                         "rate": pd.Series(dtype="float64")})
