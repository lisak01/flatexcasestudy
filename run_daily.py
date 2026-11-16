"""The scheduled daily job: ETL (daily) + rebuild the portal. No browser, no tests.

    python run_daily.py                         # source from config.FX_SOURCE (ECB API)
    python run_daily.py --source s3://b/x.csv   # any other source, same flow

For everything at once (incl. tests, opening the page) use: python run_all.py

If the database is still empty, the initial load runs first. Exit code 1 if the
ETL fails; the dashboard is then still rebuilt from the last good data, and its
status badge shows the problem.
"""

# reads options from the command line, e.g. --source
import argparse

# needed to return the exit code
import sys

# the database
import duckdb

# our settings
import config

# our modules: loading and web page
from src import etl, portal

# picks the source adapter (ECB or file)
from src.adapters import make_adapter


def has_data() -> bool:
    """True if the database already has at least one rate."""
    # no database file yet -> no data
    if not config.DUCKDB_PATH.exists():
        return False
    # open the database, only for reading
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        # count the rates; more than 0 -> True
        return con.execute("SELECT count(*) FROM fx_rates").fetchall()[0][0] > 0
    # table does not exist yet -> no data
    except duckdb.Error:
        return False
    # always close the database
    finally:
        con.close()


def main() -> int:
    """Daily job: load new rates, then rebuild the dashboard."""
    # describes the command-line options
    parser = argparse.ArgumentParser(description=__doc__)
    # --source: other source, default from config (ecb)
    parser.add_argument("--source", default=config.FX_SOURCE,
                        help="'ecb' (API) or a file URI: local path, s3://, az://, gs://")
    # read what the user typed
    args = parser.parse_args()
    # pick the adapter: ECB or file
    adapter = make_adapter(args.source)

    # database empty -> full load, otherwise only new days
    mode = "daily" if has_data() else "initial"
    # run the ETL
    log = etl.run(mode, adapter)
    # print the result
    print(f"ETL [{log['status']}] mode={mode}: {log['message']}")

    # still no rates -> no dashboard possible
    if not has_data():
        print("No data in the database - dashboard not built.")
        return 1
    # rebuild dashboard.html and print where it is
    print(f"Portal written to {portal.build()}")
    # exit code: 1 if the load failed (scheduler can alert), otherwise 0
    return 1 if log["status"] == "FAILED" else 0


# run main() only when the file is started directly
if __name__ == "__main__":
    sys.exit(main())
