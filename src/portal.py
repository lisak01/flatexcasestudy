"""The portal: one web page that shows everything, built by `python run_all.py`.

Tabs (sidebar):
    Dashboard        board view: status bar, KPI cards, charts, one plain sentence
    Risk Analytics   method comparison, backtest chart, assumptions and limits
    Index Analysis   Task 2 report
    Data Pipeline    sources and cloud adapters, architecture, load log, database tables
    Quality          test results (pytest) and code check (ruff)

The page is one self-contained HTML file (Plotly embedded): no server, works offline,
can be sent to the board as a file.
"""

# allows type hints like "dict | None" (only for hints)
from __future__ import annotations

# turns the logo file into text, so it can sit inside the HTML
import base64

# makes text safe to put into HTML
import html

# reads the ruff result (JSON file)
import json

# reads the test result (XML file)
import xml.etree.ElementTree as ET

# current date and time (for "Generated ...")
from datetime import UTC, datetime

# file paths
from pathlib import Path

# the database
import duckdb

# tables with data (here: to check for empty values)
import pandas as pd

# the chart library code, embedded into the page
from plotly.offline import get_plotlyjs

# our settings
import config

# our modules: list of sources, FX numbers and charts
from src import adapters, dashboard

# the HTML template with {{PLACEHOLDERS}}
TEMPLATE = Path(__file__).parent / "portal_template.html"
# test result file, written by run_all.py (pytest)
TEST_RESULTS = config.OUTPUT_DIR / "test_results.xml"  # written by run_all.py (pytest)
# code check result file, written by run_all.py (ruff)
RUFF_RESULTS = config.OUTPUT_DIR / "ruff.json"  # written by run_all.py (ruff)

# readable titles for the three test files (Quality tab)
TEST_GROUPS = {"test_etl": "Task 1a - ETL and source adapters",
               "test_fx_risk": "Task 1b - FX risk and dashboard",
               "test_index_analysis": "Task 2 - index analysis"}


def esc(x) -> str:
    """Make any text safe for HTML (e.g. "<" becomes "&lt;")."""
    return html.escape(str(x))


# --- Data for the tabs -------------------------------------------------------


def pipeline_data(db_path: Path) -> dict:
    """Load log and table sizes from the database (Data Pipeline tab)."""
    # open the database, only for reading
    con = duckdb.connect(str(db_path), read_only=True)
    # the last 10 loads, newest first
    log = con.execute("""
        SELECT run_at, mode, source, rows_fetched, rows_new, min_date, max_date, status, message
        FROM etl_load_log ORDER BY run_at DESC LIMIT 10""").df()
    # number of rows per table
    tables = {}
    for name in ("fx_rates", "etl_load_log", "fx_risk_history"):
        try:
            # count the rows of this table
            tables[name] = con.execute(f"SELECT count(*) FROM {name}").fetchall()[0][0]
        # table does not exist yet -> empty
        except duckdb.Error:
            tables[name] = None
    # first and last date of the rates
    span = con.execute("SELECT min(rate_date), max(rate_date) FROM fx_rates").fetchall()[0]
    # close the database
    con.close()
    # give back: load log, table sizes, date range
    return {"log": log, "tables": tables, "span": span}


def test_results(path: Path = TEST_RESULTS) -> dict | None:
    """Read the test result file. None if the tests were not run yet."""
    # no file -> tests not run yet
    if not path.exists():
        return None
    # read the XML file
    root = ET.parse(path).getroot()
    # find the part with the test results
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    # one entry per test
    cases = []
    for tc in suite.iter("testcase"):
        # failed or error -> "failed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        # skipped (e.g. index files missing) -> "skipped"
        elif tc.find("skipped") is not None:
            status = "skipped"
        # otherwise -> "passed"
        else:
            status = "passed"
        # test file, test name, time, result
        cases.append({"module": tc.get("classname", "").split(".")[-1], "name": tc.get("name"),
                      "time": float(tc.get("time", 0)), "status": status})
    # count passed / failed / skipped
    count = {k: sum(c["status"] == k for c in cases) for k in ("passed", "failed", "skipped")}
    # give back all tests, the counts, duration and time of the run
    return {"cases": cases, "total": len(cases), **count,
            "time": float(suite.get("time", 0)), "timestamp": suite.get("timestamp", "")[:16]}


def ruff_results(path: Path = RUFF_RESULTS) -> dict | None:
    """Read the code check result. None if ruff was not run yet."""
    # no file -> not run yet
    if not path.exists():
        return None
    # list of problems found by ruff
    issues = json.loads(path.read_text(encoding="utf-8") or "[]")
    # when the file was written
    run_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d %H:%M")
    # problems, how many, when
    return {"issues": issues, "count": len(issues), "run_at": run_at}


# --- HTML pieces -------------------------------------------------------------


def status_item(ok: bool | None, label: str, value: str) -> str:
    """One field of the status bar. ok=None means 'not available' (grey)."""
    # colour and icon: green tick, red "!", or grey dash
    cls, icon = {True: ("good", "✓"), False: ("bad", "!"), None: ("muted", "–")}[ok]
    # HTML for one field: icon, label, value
    return (f'<div class="stat"><span class="stat-icon {cls}">{icon}</span>'
            f'<div><div class="label">{esc(label)}</div><div class="stat-val">{esc(value)}</div>'
            f'</div></div>')


def status_bar(fx: dict, tests: dict | None, ruff: dict | None) -> str:
    """The five status fields at the top of the dashboard."""
    items = [
        # 1. are the rates fresh?
        status_item(fx["_fresh"], "Market data", f"Up to date ({fx['AS_OF']})" if fx["_fresh"]
                    else f"{fx['_age']} days old"),
        # 2. did the last load work?
        status_item(fx["_etl_ok"], "Last data load", fx["_etl_status"]),
        # 3. did all tests pass?
        status_item(None if tests is None else tests["failed"] == 0, "Automated tests",
                    "not run yet" if tests is None
                    else f"{tests['passed']}/{tests['total'] - tests['skipped']} passed"),
        # 4. is the code clean (ruff)?
        status_item(None if ruff is None else ruff["count"] == 0, "Code check (ruff)",
                    "not run yet" if ruff is None else f"{ruff['count']} issues"),
        # 5. is the backtest green?
        status_item(fx["_zone"] == "Green", "VaR backtest", f"Basel {fx['_zone']}"),
    ]
    # glue the five fields together
    return "".join(items)


def table(headers: list[str], rows: list[list], num_cols: tuple = ()) -> str:
    """Build an HTML table. num_cols = columns aligned to the right (numbers)."""
    # header row
    head = "".join(f"<th{' class=num' if i in num_cols else ''}>{esc(h)}</th>"
                   for i, h in enumerate(headers))
    # one <tr> per row, one <td> per cell
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i in num_cols else ''}>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    # the whole table
    return f"<table><tr>{head}</tr>{body}</table>"


def pill(text: str, kind: str) -> str:
    """Small coloured label, e.g. a green "OK"."""
    return f'<span class="pill {kind}">{esc(text)}</span>'


def sources_html() -> str:
    """Table of all connectable sources (Data Pipeline tab)."""
    # one row per source
    rows = []
    for a in adapters.catalogue():
        # needed library, with a note if it is not installed
        driver = esc(a["driver"]) + ("" if a["driver_installed"] else " (not installed)")
        # True for the source currently set in config.FX_SOURCE
        live = a["example"] == config.FX_SOURCE
        # name (+ green "in use" label), type, example address, driver, use, status
        rows.append([f"<b>{esc(a['name'])}</b>" + (" " + pill("in use", "good") if live else ""),
                     esc(a["kind"]), f"<code>{esc(a['example'])}</code>", driver,
                     esc(a["use"]), esc(a["status"])])
    # as an HTML table
    return table(["Source", "Type", "Address in config", "Driver", "Typical use", "Status"], rows)


def load_log_html(log) -> str:
    """Table of the last 10 loads (Data Pipeline tab)."""
    # colour of the status label
    kind = {"OK": "good", "OK_WITH_WARNINGS": "warn", "NO_NEW_DATA": "muted", "FAILED": "bad"}
    # one row per load; empty values (e.g. after a FAILED run) are shown as ""
    rows = [[f"{r.run_at:%Y-%m-%d %H:%M}", esc(r.mode), esc(r.source),
             "" if pd.isna(r.rows_fetched) else f"{int(r.rows_fetched):,}",
             "" if pd.isna(r.rows_new) else f"{int(r.rows_new):,}",
             f"{str(r.min_date)[:10]} – {str(r.max_date)[:10]}"
             if pd.notna(r.min_date) else "",
             pill(r.status, kind.get(r.status, "muted")), esc(str(r.message)[:80])]
            for r in log.itertuples()]
    # as an HTML table (columns 3 and 4 are numbers)
    return table(["Run (UTC)", "Mode", "Source", "Fetched", "New", "Date range", "Status",
                  "Message"], rows, num_cols=(3, 4))


def tests_html(tests: dict | None) -> str:
    """One card per test file with every test and its result (Quality tab)."""
    # no results yet -> short note
    if tests is None:
        return '<p class="note">No test results yet. Run <code>python run_all.py</code>.</p>'
    # one card per test file
    out = []
    for module, title in TEST_GROUPS.items():
        # the tests of this file
        cases = [c for c in tests["cases"] if c["module"] == module]
        # no tests in this file -> skip
        if not cases:
            continue
        # result label, readable test name ("test_a_b" -> "a b"), time in ms
        rows = [[pill(c["status"], {"passed": "good", "failed": "bad"}.get(c["status"], "muted")),
                 esc(c["name"].removeprefix("test_").replace("_", " ")),
                 f"{c['time'] * 1000:.0f} ms"] for c in cases]
        # how many passed
        ok = sum(c["status"] == "passed" for c in cases)
        # the card: title, "(x/y passed)", table
        out.append(f'<div class="card"><h2>{esc(title)} <span class="note">({ok}/{len(cases)} '
                   f'passed)</span></h2>{table(["Result", "What is tested", "Time"], rows, (2,))}'
                   f'</div>')
    # all cards together
    return "".join(out)


def ruff_html(ruff: dict | None) -> str:
    """Result of the code check (Quality tab)."""
    # not run yet
    if ruff is None:
        return '<p class="note">Not run yet.</p>'
    # no problems -> green "0 issues"
    if ruff["count"] == 0:
        return (f'<p>{pill("0 issues", "good")} All files pass the configured rules '
                f'(pycodestyle, pyflakes, import order, bugbear, pyupgrade). '
                f'Checked {ruff["run_at"]} UTC.</p>')
    # problems -> table: file, line, rule, message (max 30)
    rows = [[esc(i["filename"].split("/")[-1]), str(i["location"]["row"]), esc(i["code"]),
             esc(i["message"])] for i in ruff["issues"][:30]]
    return table(["File", "Line", "Rule", "Message"], rows)


def logo_html() -> str:
    """Logo file if present (config.LOGO_PATH), else the company name as text."""
    # path to the optional logo
    path = Path(config.LOGO_PATH)
    # logo exists -> put the picture directly into the HTML
    if path.exists():
        # file type: svg or png
        mime = "image/svg+xml" if path.suffix == ".svg" else f"image/{path.suffix[1:]}"
        # picture as text (base64)
        data = base64.b64encode(path.read_bytes()).decode()
        # alternative text: company name
        alt = esc(config.BRAND_NAME)
        return f'<img class="logo-img" src="data:{mime};base64,{data}" alt="{alt}">'
    # no logo -> blue bar + company name as text
    return f'<span class="bar"></span>{esc(config.BRAND_NAME)}'


# --- Build -------------------------------------------------------------------


def build(db_path: Path = config.DUCKDB_PATH, out: Path = config.DASHBOARD_PATH,
          index_res: dict | None = None, include_index: bool = True) -> Path:
    """Write the portal. index_res: results of index_analysis.analyse(), computed if missing."""
    # all FX numbers and charts for the dashboard
    fx = dashboard.fx_values(dashboard.collect(db_path))
    # load log and table sizes
    pipe = pipeline_data(db_path)
    # test and code check results
    tests, ruff = test_results(), ruff_results()

    # no Task 2 results given, but the files exist -> calculate them now
    if index_res is None and include_index and config.PRICE_HIST_PATH.exists():
        from src import index_analysis
        index_res = index_analysis.analyse()
    # Task 2 results available -> report and the "0.48 -> 0.60" number
    if index_res is not None:
        from src import index_analysis
        index_body = index_analysis.report_body(index_res)
        d = index_res["daily"]
        index_kpi = f"{d['avg_normal']:.2f} → {d['avg_crisis']:.2f}"
    # no Task 2 data -> short note instead of the report
    else:
        index_body = ('<div class="card"><p>Index data not found. Put dim_master.csv and '
                      'price_hist.parquet into data/ and run '
                      '<code>python run_all.py</code>.</p></div>')
        index_kpi = "n/a"

    # table sizes, short name
    t = pipe["tables"]
    # every {{PLACEHOLDER}} in the template and what goes in
    values = {
        # all FX numbers (except internal ones starting with "_")
        **{k: v for k, v in fx.items() if not k.startswith("_")},
        # sidebar: company name, unit, logo; page title
        "BRAND": esc(config.BRAND_NAME), "UNIT": esc(config.BRAND_UNIT), "LOGO": logo_html(),
        "TITLE": esc(config.DASHBOARD_TITLE),
        # five status fields
        "STATUS_BAR": status_bar(fx, tests, ruff),
        # Task 2 tab
        "INDEX_BODY": index_body, "INDEX_KPI": index_kpi,
        # Data Pipeline tab: sources, current source, load log
        "SOURCES": sources_html(), "FX_SOURCE": esc(config.FX_SOURCE),
        "LOAD_LOG": load_log_html(pipe["log"]),
        # row counts of the three tables
        "ROWS_RATES": f"{t['fx_rates']:,}" if t["fx_rates"] else "0",
        "ROWS_LOG": f"{t['etl_load_log']:,}" if t["etl_load_log"] else "0",
        "ROWS_RISK": f"{t['fx_risk_history']:,}" if t["fx_risk_history"] else "0",
        # first and last date of the rates
        "SPAN": f"{pipe['span'][0]} to {pipe['span'][1]}",
        # Quality tab: test cards and ruff result
        "TESTS": tests_html(tests), "RUFF": ruff_html(ruff),
        # Quality tab: numbers ("–" if not run yet)
        "TESTS_TOTAL": "–" if tests is None else str(tests["total"]),
        "TESTS_PASSED": "–" if tests is None else str(tests["passed"]),
        "TESTS_FAILED": "–" if tests is None else str(tests["failed"]),
        "TESTS_TIME": "–" if tests is None else f"{tests['time']:.1f} s",
        "TESTS_AT": "not run yet" if tests is None else f"{tests['timestamp'].replace('T', ' ')}",
        "RUFF_COUNT": "–" if ruff is None else str(ruff["count"]),
        # time the page was built
        "GENERATED": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    # read the template
    page = TEMPLATE.read_text(encoding="utf-8")
    # replace every {{NAME}} with its value
    for key, val in values.items():
        page = page.replace("{{" + key + "}}", str(val))
    # Plotly's JavaScript last, so the placeholders above are never searched inside it.
    page = page.replace("{{PLOTLY_JS}}", get_plotlyjs())

    # where to save (output/dashboard.html)
    out = Path(out)
    # create the output folder if it is missing
    out.parent.mkdir(parents=True, exist_ok=True)
    # save the finished page
    out.write_text(page, encoding="utf-8")
    # give back the path of the page
    return out


# run build() only when the file is started directly
if __name__ == "__main__":
    print(f"Portal written to {build()}")
