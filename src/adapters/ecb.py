"""Adapter for the ECB Data Portal API (SDMX REST, CSV format).

Only fetching and parsing happens here; checks and loading are done by the ETL.
"""

# allows type hints like "str | None" (only for hints)
from __future__ import annotations

# lets pandas read a text as if it were a file
import io

# for the pause between attempts
import time

# tables with data
import pandas as pd

# sends requests to websites
import requests

# our settings
import config

# the template, our error type and the empty table
from src.adapters.base import SourceAdapter, SourceError, empty_frame


class EcbClient(SourceAdapter):
    """Downloads the USD/EUR rate from the ECB website."""

    # name written to the load log
    name = "ecb_data_portal"

    # runs once when the adapter is created; all values come from config
    def __init__(
        self,
        flow: str = config.ECB_FLOW,
        series_key: str = config.ECB_SERIES_KEY,
        base_url: str = config.ECB_BASE_URL,
        timeout: float = config.HTTP_TIMEOUT_SECONDS,
        max_retries: int = config.HTTP_MAX_RETRIES,
        backoff: float = config.HTTP_BACKOFF_SECONDS,
    ):
        # build the web address: base / EXR / D.USD.EUR.SP00.A
        self.url = f"{base_url}/{flow}/{series_key}"
        # how long to wait for an answer (30 s)
        self.timeout = timeout
        # how many attempts in total (3)
        self.max_retries = max_retries
        # first pause between attempts (2 s)
        self.backoff = backoff

    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        """Get the rates from start_date on, as a table (rate_date, rate)."""
        # ask the ECB for CSV
        params = {"format": "csvdata"}
        # and only from this date on
        if start_date:
            params["startPeriod"] = start_date
        # download (with retries)
        text = self._get_with_retry(params)
        # turn the CSV text into our two-column table
        return parse_ecb_csv(text)

    def _get_with_retry(self, params: dict) -> str:
        """Download with retries: wait 2 s, then 4 s, then give up.

        Retries only problems that can go away by themselves
        (timeouts, connection errors, HTTP 429 and 5xx).
        """
        # remembers the last problem, for the error message
        last_error = None
        # attempt 1, 2, 3
        for attempt in range(1, self.max_retries + 1):
            try:
                # send the request to the ECB
                resp = requests.get(self.url, params=params, timeout=self.timeout)
                # The ECB API answers 404 "No results found" when the period has no
                # observations (e.g. daily run before 16:00 CET). That is not an error.
                if resp.status_code == 404:
                    return ""
                # 429 = too many requests, 5xx = server problem -> try again
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                # everything else: success, or a wrong request
                else:
                    resp.raise_for_status()  # other 4xx: wrong request, do not retry
                    # success: give back the CSV text
                    return resp.text
            # no answer in time or no connection -> try again
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            # wrong request (e.g. 400) -> retrying won't help, stop
            except requests.HTTPError as exc:
                raise SourceError(f"ECB API rejected the request: {exc}") from exc

            # not the last attempt -> wait (2 s, then 4 s)
            if attempt < self.max_retries:
                time.sleep(self.backoff * 2 ** (attempt - 1))

        # all attempts failed -> clear error message
        raise SourceError(
            f"ECB API not reachable after {self.max_retries} attempts "
            f"({self.url}). Last error: {last_error}"
        )


def parse_ecb_csv(text: str) -> pd.DataFrame:
    """Turn the ECB CSV text into our table with two columns: rate_date, rate.

    The response has many metadata columns; we only need TIME_PERIOD and OBS_VALUE.
    Dates and values are parsed here; checking them is the job of the ETL.
    """
    # empty answer -> empty table
    if not text.strip():
        return empty_frame()

    # read the CSV text as a table, everything as text first
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    # do the two columns we need exist?
    missing = {"TIME_PERIOD", "OBS_VALUE"} - set(raw.columns)
    # if not -> the format changed, stop with a clear error
    if missing:
        raise SourceError(f"Unexpected ECB CSV format, missing columns: {missing}")

    # Until 2012 the ECB also sends TARGET holidays as rows without a value and
    # OBS_STATUS = "H" (holiday). They are not observations, so they are dropped here.
    # Any other empty value stays and is stopped by the quality checks in the ETL.
    if "OBS_STATUS" in raw.columns:
        # keep only rows that are not holidays
        raw = raw[raw["OBS_STATUS"] != "H"]

    # build the final table: date and rate
    return pd.DataFrame({
        # errors="coerce": invalid dates/values become NaT/NaN and are reported
        # by the quality checks instead of crashing here.
        # text -> real date
        "rate_date": pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m-%d", errors="coerce"),
        # text -> number
        "rate": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
    })
