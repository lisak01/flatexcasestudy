"""ETL for the ECB USD/EUR reference rate into a local DuckDB file.

Usage:
    python -m src.etl --initial      # full history, table rebuilt
    python -m src.etl --daily        # only new days, idempotent upsert
    python -m src.etl --daily --source s3://bucket/usd_eur.csv   # any other source, same flow

Flow of one run:
    1. extract   adapter.fetch(start_date)             -> DataFrame(rate_date, rate)
    2. check     hard checks on the new batch          -> stop and log FAILED if broken
    3. load      one transaction: rebuild or upsert    -> fx_rates
    4. check     soft checks on the full stored series -> warnings (gaps, jumps)
    5. log       one row per run                       -> etl_load_log
"""

# allows type hints like "str | None" (only for hints, no effect on the logic)
from __future__ import annotations

# reads options from the command line, e.g. --initial / --daily
import argparse

# turns the check results into text for the log table
import json

# needed to return the exit code (0 = OK, 1 = failed)
import sys

# dates and times: now, minus 5 days, etc.
from datetime import UTC, datetime, timedelta

# file paths
from pathlib import Path

# the database
import duckdb

# tables with data
import pandas as pd

# calculates the date of Easter (for ECB holidays)
from dateutil.easter import easter

# our settings
import config

# the source adapters: common type, error type, and the function that picks one
from src.adapters import SourceAdapter, SourceError, make_adapter

# --- Database schema ---------------------------------------------------------

# SQL that creates the two tables if they do not exist yet
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fx_rates (
    series_id  VARCHAR   NOT NULL,   -- e.g. EXR.D.USD.EUR.SP00.A
    rate_date  DATE      NOT NULL,   -- ECB reference date
    rate       DOUBLE    NOT NULL,   -- USD per 1 EUR
    source     VARCHAR   NOT NULL,   -- adapter that delivered the value
    loaded_at  TIMESTAMP NOT NULL,   -- when this row was written (UTC)
    PRIMARY KEY (series_id, rate_date)  -- guarantees: one rate per day
);

CREATE TABLE IF NOT EXISTS etl_load_log (
    run_at       TIMESTAMP NOT NULL,  -- UTC
    mode         VARCHAR   NOT NULL,  -- initial | daily
    series_id    VARCHAR   NOT NULL,
    source       VARCHAR   NOT NULL,
    rows_fetched INTEGER,
    rows_new     INTEGER,             -- dates that did not exist before
    min_date     DATE,                -- date range of the fetched batch
    max_date     DATE,
    status       VARCHAR   NOT NULL,  -- OK | OK_WITH_WARNINGS | NO_NEW_DATA | FAILED
    checks       VARCHAR,             -- JSON with the result of every check
    message      VARCHAR
);
"""


def connect(db_path: Path = config.DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the database and make sure both tables exist."""
    # make sure it is a path
    db_path = Path(db_path)
    # create the data folder if it is missing
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # open the database file (creates it if new)
    con = duckdb.connect(str(db_path))
    # create the tables if they are missing
    con.execute(SCHEMA_SQL)
    # give back the open database
    return con


# --- Quality checks ----------------------------------------------------------

def check_batch(df: pd.DataFrame) -> dict:
    """Hard checks on the new data. If one fails, nothing is loaded."""
    # each line answers True (OK) or False (problem)
    return {
        # no empty dates
        "valid_dates": int(df["rate_date"].isna().sum()) == 0,
        # no empty rates
        "valid_values": int(df["rate"].isna().sum()) == 0,
        # every rate is above 0
        "rate_positive": bool((df["rate"].dropna() > 0).all()),
        # no date appears twice
        "no_duplicate_dates": not df["rate_date"].duplicated().any(),
    }


def target_holidays(years) -> set[pd.Timestamp]:
    """Days without an ECB rate (holidays), so they don't count as missing.

    New Year, Good Friday, Easter Monday, 1 May, 25 and 26 December;
    in 1999-2001 also 31 December (confirmed by the ECB holiday rows, OBS_STATUS = H).
    """
    # empty set, holidays are added below
    days = set()
    # go through every year
    for y in years:
        # Easter Sunday of this year
        e = pd.Timestamp(easter(y))
        # add the fixed holidays and the two Easter days
        days |= {
            # 1 Jan, Good Friday (Easter - 2), Easter Monday (Easter + 1)
            pd.Timestamp(y, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
            # 1 May, 25 Dec, 26 Dec
            pd.Timestamp(y, 5, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26),
        }
        # until 2001 the ECB was also closed on 31 Dec
        if y <= 2001:
            days.add(pd.Timestamp(y, 12, 31))
    # all holidays
    return days


def check_series(dates: pd.Series, rates: pd.Series) -> dict:
    """Soft checks on all stored data. Only warnings, the load is not stopped.

    Gaps: ECB publishes on TARGET business days only, so weekends and TARGET
    holidays are expected gaps. Any other missing weekday is reported.
    """
    # dates as real dates, sorted
    dates = pd.to_datetime(dates).sort_values()
    # no data at all -> nothing to warn about
    if dates.empty:
        return {"unexplained_missing_weekdays": [], "weekend_dates": [], "large_jumps": []}

    # all Mon-Fri days from first to last date
    weekdays = pd.bdate_range(dates.min(), dates.max())  # Mon-Fri
    # all ECB holidays in these years
    holidays = target_holidays(range(dates.min().year, dates.max().year + 1))
    # days that should have a rate: weekdays minus holidays
    expected = {d for d in weekdays if d not in holidays}
    # days that should have a rate but don't
    missing = sorted(expected - set(dates))
    # dates on Saturday or Sunday (should not exist)
    weekend = sorted(d for d in dates if d.weekday() >= 5)

    # rates with their dates, sorted
    s = pd.Series(rates.values, index=dates.values).sort_index()
    # daily change in %, without sign
    change = s.pct_change().abs()
    # days where the rate moved more than 5%
    jumps = change[change > config.MAX_ABS_DAILY_CHANGE]

    # date format for the log, e.g. 2026-11-13
    fmt = "%Y-%m-%d"
    # three lists of dates as text
    return {
        "unexplained_missing_weekdays": [d.strftime(fmt) for d in missing],
        "weekend_dates": [d.strftime(fmt) for d in weekend],
        "large_jumps": [pd.Timestamp(d).strftime(fmt) for d in jumps.index],
    }


# --- Main run ----------------------------------------------------------------

def run(mode: str, adapter: SourceAdapter, db_path: Path = config.DUCKDB_PATH,
        series_id: str = config.SERIES_ID) -> dict:
    """One ETL run: fetch -> check -> load -> check -> log. Returns the log row."""
    # time of this run (UTC)
    run_at = datetime.now(UTC).replace(tzinfo=None)
    # open the database
    con = connect(db_path)
    # the log row; starts as FAILED and is set to OK only if everything works
    log = {"run_at": run_at, "mode": mode, "series_id": series_id, "source": adapter.name,
           "rows_fetched": None, "rows_new": None, "min_date": None, "max_date": None,
           "status": "FAILED", "checks": {}, "message": ""}
    # remembers if we are inside the transaction (needed to undo on error)
    in_tx = False  # True while the load transaction is open
    try:
        # 1. Decide from which date to fetch.
        # initial: everything since 1999
        if mode == "initial":
            start = config.INITIAL_START_DATE
        # daily: only recent days
        elif mode == "daily":
            # last date we already have
            last = con.execute("SELECT max(rate_date) FROM fx_rates WHERE series_id = ?",
                               [series_id]).fetchall()[0][0]
            # nothing stored yet -> daily is not possible, initial first
            if last is None:
                raise RuntimeError("No data stored yet. Run 'python -m src.etl --initial' first.")
            # start 5 days before the last date (catches ECB corrections)
            start = (last - timedelta(days=config.DAILY_LOOKBACK_DAYS)).isoformat()
        # any other mode is a mistake
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Fetch the data from the source (ECB or file).
        batch = adapter.fetch(start)
        # how many rows we got
        log["rows_fetched"] = len(batch)
        # remember first and last date of the new data
        if not batch.empty:
            log["min_date"] = batch["rate_date"].min()
            log["max_date"] = batch["rate_date"].max()

        # nothing came back
        if batch.empty:
            # for initial this is an error
            if mode == "initial":
                raise RuntimeError("Source returned no data for the initial load.")
            # for daily it is normal (e.g. weekend): log NO_NEW_DATA and stop
            log.update(status="NO_NEW_DATA", rows_new=0,
                       message=f"No observations since {start} (not yet published?).")
            return log

        # 3. Hard checks on the new data.
        hard = check_batch(batch)
        # save the check results in the log
        log["checks"]["batch"] = hard
        # if any check failed -> stop, load nothing
        if not all(hard.values()):
            # names of the failed checks
            failed = [k for k, ok in hard.items() if not ok]
            raise RuntimeError(f"Quality checks failed, nothing loaded: {failed}")

        # 4. Load in one transaction: if anything fails, the old data stays untouched.
        # add the extra columns: series name, source, load time
        rows = batch.assign(series_id=series_id, source=adapter.name, loaded_at=run_at)
        # make the table visible to SQL under the name new_rows
        con.register("new_rows", rows)
        # start the transaction: all or nothing
        con.execute("BEGIN TRANSACTION")
        in_tx = True
        # initial: delete the old data first (full rebuild)
        if mode == "initial":
            con.execute("DELETE FROM fx_rates WHERE series_id = ?", [series_id])
        # count rows whose date we did not have before (= really new)
        rows_new = con.execute(
            "SELECT count(*) FROM new_rows n WHERE NOT EXISTS ("
            " SELECT 1 FROM fx_rates f"
            " WHERE f.series_id = n.series_id AND f.rate_date = n.rate_date)"
        ).fetchall()[0][0]
        # Upsert on (series_id, rate_date): a second run with the same data
        # overwrites the same rows instead of creating duplicates.
        # new date -> insert; existing date -> update the rate
        con.execute("""
            INSERT INTO fx_rates (series_id, rate_date, rate, source, loaded_at)
            SELECT series_id, CAST(rate_date AS DATE), rate, source, loaded_at FROM new_rows
            ON CONFLICT (series_id, rate_date) DO UPDATE SET
                rate = excluded.rate, source = excluded.source, loaded_at = excluded.loaded_at
        """)
        # save everything for good
        con.execute("COMMIT")
        in_tx = False
        # remove the temporary name new_rows
        con.unregister("new_rows")
        # store the number of new rows in the log
        log["rows_new"] = int(rows_new)

        # 5. Soft checks on all stored data.
        # read all stored rates back
        stored = con.execute("SELECT rate_date, rate FROM fx_rates WHERE series_id = ? "
                             "ORDER BY rate_date", [series_id]).df()
        # gaps, weekend dates, big jumps
        soft = check_series(stored["rate_date"], stored["rate"])
        # in the log: count + first 20 dates of each warning
        log["checks"]["series"] = {k: {"count": len(v), "dates": v[:20]} for k, v in soft.items()}
        # True if at least one warning list is not empty
        has_warnings = any(soft.values())
        # status: OK or OK with warnings
        log["status"] = "OK_WITH_WARNINGS" if has_warnings else "OK"
        # short summary text
        log["message"] = (f"{rows_new} new rows, {len(stored)} rows stored, "
                          f"last date {stored['rate_date'].max():%Y-%m-%d}.")
        return log

    # any expected error lands here
    except (SourceError, RuntimeError, ValueError, duckdb.Error) as exc:
        # undo the half-finished load, old data stays
        if in_tx:
            con.execute("ROLLBACK")
        # mark the run as failed with the error text
        log["status"] = "FAILED"
        log["message"] = str(exc)
        return log

    # runs always, success or error
    finally:
        # write one row into etl_load_log
        _write_log(con, log)
        # close the database
        con.close()


def _write_log(con: duckdb.DuckDBPyConnection, log: dict) -> None:
    """Write one row into etl_load_log."""
    # 11 values for the 11 columns of the log table
    con.execute(
        "INSERT INTO etl_load_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [log["run_at"], log["mode"], log["series_id"], log["source"], log["rows_fetched"],
         log["rows_new"],
         # dates only if there are any, otherwise empty
         log["min_date"].date() if log["min_date"] is not None else None,
         log["max_date"].date() if log["max_date"] is not None else None,
         # check results saved as JSON text
         log["status"], json.dumps(log["checks"]), log["message"]],
    )


def main(argv: list[str] | None = None) -> int:
    """Command line: python -m src.etl --initial / --daily [--source ...]."""
    # describes the command-line options
    parser = argparse.ArgumentParser(description="Load the ECB USD/EUR rate into DuckDB.")
    # exactly one of --initial or --daily must be given
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initial", action="store_true", help="full history, rebuild table")
    group.add_argument("--daily", action="store_true", help="only new days (upsert)")
    # optional: other source, default from config (ecb)
    parser.add_argument("--source", default=config.FX_SOURCE,
                        help="'ecb' (API) or a file URI: local path, s3://, az://, gs://")
    # optional: other database file
    parser.add_argument("--db", default=str(config.DUCKDB_PATH), help="DuckDB file path")
    # read what the user typed
    args = parser.parse_args(argv)

    # pick the adapter: ECB or file
    adapter = make_adapter(args.source)
    # run the ETL in the chosen mode
    log = run("initial" if args.initial else "daily", adapter, Path(args.db))

    # print a one-line summary in the terminal
    print(f"[{log['status']}] mode={log['mode']} source={log['source']} "
          f"fetched={log['rows_fetched']} new={log['rows_new']} :: {log['message']}")
    # print the warnings, if any
    series_checks = log["checks"].get("series", {})
    for name, result in series_checks.items():
        if result["count"]:
            print(f"  warning {name}: {result['count']} (first: {result['dates'][:5]})")
    # exit code: 1 = failed (the scheduler can send an alert), 0 = OK
    return 1 if log["status"] == "FAILED" else 0


# run main() only when the file is started directly
if __name__ == "__main__":
    sys.exit(main())
