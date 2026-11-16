"""One adapter for files anywhere: local disk, network share, Amazon S3, Azure, Google Cloud.

The location is a URI; fsspec picks the right driver from its prefix:

    data/usd_eur.csv                              local file or network share
    s3://bank-raw/ecb/usd_eur.csv                 Amazon S3              (pip install s3fs)
    az://raw/ecb/usd_eur.csv, abfs://...          Azure Blob / ADLS Gen2 (pip install adlfs)
    gs://bank-raw/ecb/usd_eur.csv                 Google Cloud Storage   (pip install gcsfs)
    memory://bucket/usd_eur.csv                   in-memory "bucket", used in the tests

Credentials never appear in the code: each driver uses the cloud's standard
credential chain (environment variables, managed identity, IAM role, CLI login).
The file is expected in the ECB CSV format (e.g. a copy in a landing zone).
"""

# allows type hints like "str | None" (only for hints)
from __future__ import annotations

# opens files anywhere: local disk, S3, Azure, Google Cloud
import fsspec

# tables with data
import pandas as pd

# the template and our error type
from src.adapters.base import SourceAdapter, SourceError

# reuse the ECB CSV reader (the file has the same format)
from src.adapters.ecb import parse_ecb_csv


class FileAdapter(SourceAdapter):
    """Reads the rates from a file: local or in the cloud."""

    # runs once when the adapter is created
    def __init__(self, uri: str):
        # the file path, e.g. data/rates.csv or s3://bucket/x.csv
        self.uri = str(uri)
        # the part before "://" (s3, az, gs) or "local" for a normal path
        protocol = self.uri.split("://")[0] if "://" in self.uri else "local"
        # name for the load log
        self.name = f"file_{protocol}"  # e.g. file_s3, file_az, file_local

    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        """Read the file and return the rates from start_date on."""
        try:
            # open the file (fsspec picks local or cloud by itself)
            with fsspec.open(self.uri, "rt", encoding="utf-8") as f:
                # read the whole file as text
                text = f.read()
        # library for this cloud not installed
        except ImportError as exc:  # driver for this cloud not installed
            raise SourceError(f"Missing driver for {self.uri}: {exc}") from exc
        # file does not exist
        except FileNotFoundError as exc:
            raise SourceError(f"File not found: {self.uri}") from exc
        # other problems: network, no permission, no login
        except OSError as exc:  # network, permission, credentials
            raise SourceError(f"Cannot read {self.uri}: {exc}") from exc

        # turn the CSV text into our table (rate_date, rate)
        df = parse_ecb_csv(text)
        # keep only rows from start_date on
        if start_date:
            df = df[df["rate_date"] >= pd.Timestamp(start_date)]
        # renumber the rows 0, 1, 2, ... and give back the table
        return df.reset_index(drop=True)
