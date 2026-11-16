"""One command for everything:

    python run_all.py              # load data, compute, test, build the portal, open it
    python run_all.py --no-open    # same, without opening the browser

Steps:
    1. ETL       ECB rates into DuckDB (initial load if the database is empty, else daily)
    2. Task 2    index analysis (if the supplied files are in data/)
    3. Quality   pytest (results -> output/test_results.xml) and ruff (-> output/ruff.json)
    4. Portal    output/dashboard.html with all tabs, opened in the browser
"""

# reads options from the command line, e.g. --no-open
import argparse

# starts other programs (pytest, ruff) from Python
import subprocess

# exit code and path to the current Python
import sys

# opens the finished page in the browser
import webbrowser

# our settings
import config

# checks if the database already has rates
from run_daily import has_data

# our modules: loading, Task 2, web page
from src import etl, index_analysis, portal

# picks the source adapter (ECB or file)
from src.adapters import make_adapter


def run_quality_checks() -> None:
    """Run the tests and the code check; their results are shown on the Quality tab."""
    # create the output folder if it is missing
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # run all tests, save the results as an XML file
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={portal.TEST_RESULTS}"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    # print the last line of the test output, e.g. "29 passed"
    print("   " + (tests.stdout.strip().splitlines() or ["no output"])[-1])
    # run the code check (ruff), result as JSON
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format", "json"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    # save the ruff result to a file for the portal
    portal.RUFF_RESULTS.write_text(ruff.stdout or "[]", encoding="utf-8")
    # short summary in the terminal
    print(f"   ruff: {'no issues' if ruff.returncode == 0 else 'issues found'}")


def main() -> int:
    """Run all four steps one after another."""
    # describes the command-line options
    parser = argparse.ArgumentParser(description=__doc__)
    # --no-open: don't open the browser at the end
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    # --skip-tests: skip tests and code check (faster)
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest and ruff")
    # read what the user typed
    args = parser.parse_args()

    # --- Step 1: load the ECB rates ---
    print("1/4 ETL")
    # database empty -> full load, otherwise only new days
    mode = "daily" if has_data() else "initial"
    # run the ETL with the source from config (ecb or a file)
    log = etl.run(mode, make_adapter(config.FX_SOURCE))
    # print the result, e.g. "[OK] daily: 0 new rows ..."
    print(f"   [{log['status']}] {mode}: {log['message']}")
    # still no rates at all -> nothing to show, stop
    if not has_data():
        print("No FX data in the database - cannot build the portal.")
        return 1

    # --- Step 2: index analysis (Task 2) ---
    print("2/4 Index analysis")
    # stays empty if the input files are missing
    index_res = None
    # run only if price_hist.parquet is in data/
    if config.PRICE_HIST_PATH.exists():
        # calculate all correlations
        index_res = index_analysis.analyse()
        # save report and CSV tables in output/
        index_analysis.save_outputs(index_res)
        # the daily results
        d = index_res["daily"]
        # print e.g. "normal 0.48 -> crisis 0.60"
        print(f"   avg correlation normal {d['avg_normal']:.2f} -> crisis {d['avg_crisis']:.2f}")
    # files missing -> skip Task 2, the rest still works
    else:
        print(f"   skipped: {config.PRICE_HIST_PATH} not found")

    # --- Step 3: tests and code check ---
    print("3/4 Tests and code check")
    # skipped if --skip-tests was given
    if args.skip_tests:
        print("   skipped")
    # otherwise run pytest and ruff
    else:
        run_quality_checks()

    # --- Step 4: build the web page ---
    print("4/4 Portal")
    # build dashboard.html (with the Task 2 tab if we have results)
    out = portal.build(index_res=index_res, include_index=index_res is not None)
    # print where the page was saved
    print(f"   {out}")
    # open the page in the browser (unless --no-open)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    # exit code: 1 if the load failed, otherwise 0
    return 1 if log["status"] == "FAILED" else 0


# run main() only when the file is started directly
if __name__ == "__main__":
    sys.exit(main())
