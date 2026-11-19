"""Task 2: Are equity indices more correlated in times of market distress?

    python -m src.index_analysis

Steps:
    1. load      dim_master.csv + price_hist.parquet via DuckDB, run data checks
    2. clean     split the NASDAQ duplicate key, drop weekend rows
    3. returns   daily log returns (own trading days) and weekly Friday-to-Friday returns
    4. outliers  flag suspicious returns (not deleted; excluded from the main result)
    5. regimes   crisis = VIX close above 30 (sensitivity: other levels)
    6. compare   correlations crisis vs. normal, average pairwise, Forbes-Rigobon adjusted
    7. report    output/index_analysis.html + CSV tables in output/
"""

# allows type hints like "str | None" (only for hints)
from __future__ import annotations

# makes text safe to put into HTML
import html

# current date and time (for "Generated ...")
from datetime import UTC, datetime

# the database, used here to read the files with SQL
import duckdb

# fast math on arrays
import numpy as np

# tables with data
import pandas as pd

# charts
import plotly.graph_objects as go

# the chart library code, embedded into the HTML page
from plotly.offline import get_plotlyjs

# charts with several panels
from plotly.subplots import make_subplots

# our settings
import config

# --- 1. Load and check -------------------------------------------------------


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read both files and check them for problems (before fixing anything)."""
    # database in memory, only to run SQL on the files
    con = duckdb.connect()
    # read the index list (name, key, country)
    dim = con.execute(f"SELECT * FROM read_csv('{config.DIM_MASTER_PATH}')").df()
    # read the daily prices
    px = con.execute(f"""
        SELECT asset_key, CAST(date AS DATE) AS date, close, adjusted_close, volume
        FROM read_parquet('{config.PRICE_HIST_PATH}')""").df()
    # make both tables visible to SQL under short names
    con.register("dim", dim)
    con.register("px", px)

    # all data checks, each one SQL query
    checks = {
        # number of rows in the index list
        "dim_rows": len(dim),
        # number of price rows
        "price_rows": len(px),
        # A key that appears twice in the master table duplicates every joined price row.
        "duplicate_keys_in_master": con.execute("""
            SELECT asset_key, string_agg(ticker, ', ') AS tickers FROM dim
            GROUP BY asset_key HAVING count(*) > 1""").df().to_dict("records"),
        # rows after a simple join (shows the NASDAQ problem: too many rows)
        "naive_join_rows": con.execute(
            "SELECT count(*) FROM px JOIN dim USING (asset_key)").fetchall()[0][0],
        # keys with two prices on the same day
        "keys_with_two_prices_per_day": con.execute("""
            SELECT asset_key, count(*) AS days FROM (
                SELECT asset_key, date FROM px GROUP BY ALL HAVING count(*) > 1)
            GROUP BY asset_key""").df().to_dict("records"),
        # prices on Saturday or Sunday
        "weekend_rows": con.execute(
            "SELECT count(*) FROM px WHERE dayofweek(date) IN (0, 6)").fetchall()[0][0],
        # share of rows with volume 0
        "zero_volume_share": con.execute(
            "SELECT avg((volume = 0)::INT) FROM px").fetchall()[0][0],
        # rows where close and adjusted close differ
        "close_differs_from_adjusted": con.execute(
            "SELECT count(*) FROM px WHERE close <> adjusted_close").fetchall()[0][0],
        # prices that are 0, negative or empty
        "non_positive_close": con.execute(
            "SELECT count(*) FROM px WHERE close <= 0 OR close IS NULL").fetchall()[0][0],
        # indices that start later than January 2008
        "start_dates": con.execute("""
            SELECT asset_key, min(date) AS start FROM px GROUP BY 1
            HAVING min(date) > DATE '2008-01-31'""").df().to_dict("records"),
    }
    # close the database
    con.close()
    # dates as real dates
    px["date"] = pd.to_datetime(px["date"])
    # give back: index list, prices, check results
    return dim, px, checks


# --- 2. Clean ----------------------------------------------------------------


def split_nasdaq(px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Separate NASDAQ Composite (IXIC) and NASDAQ 100 (NDX), which share one asset_key.

    Each day has two prices. The level tells them apart: until 2020 the Composite is
    higher, from mid-2021 the NASDAQ 100. In between the two cross several times and
    are often less than 1% apart, so the identity cannot be proven from the data.
    Those dates are set to missing for both series instead of guessing.
    """
    # only the NASDAQ rows (the shared key)
    nas = px[px["asset_key"] == config.NASDAQ_KEY]
    # higher of the two prices per day
    hi = nas.groupby("date")["close"].max()
    # lower of the two prices per day
    lo = nas.groupby("date")["close"].min()
    # days where both prices are less than 1% apart (can't tell them apart)
    close_call = hi / lo < config.NASDAQ_AMBIGUOUS_RATIO
    # first and last of these unclear days
    first, last = close_call[close_call].index.min(), close_call[close_call].index.max()

    # days before / after the unclear period
    before, after = hi.index < first, hi.index > last
    # Composite: higher price before, lower price after, empty in between
    ixic = pd.Series(np.where(before, hi, np.where(after, lo, np.nan)), index=hi.index)
    # NASDAQ 100: the opposite
    ndx = pd.Series(np.where(before, lo, np.where(after, hi, np.nan)), index=hi.index)

    # Check against published closing values.
    # compare with the 3 known real closing prices (tolerance 0.05 points)
    anchors_ok = all(
        abs(ixic[pd.Timestamp(d)] - a) < 0.05 and abs(ndx[pd.Timestamp(d)] - b) < 0.05
        for d, (a, b) in config.NASDAQ_ANCHORS.items()
    )
    # two clean tables (ticker, date, close), empty days dropped
    rows = [pd.DataFrame({"ticker": t, "date": s.index, "close": s.values}).dropna()
            for t, s in (("IXIC.INDX", ixic), ("NDX.INDX", ndx))]
    # info for the report: unclear period, number of days, check result
    info = {"ambiguous_from": first, "ambiguous_to": last,
            "days_set_missing": int((~before & ~after).sum()), "anchors_ok": anchors_ok,
            "hi": hi, "lo": lo}
    # give back both NASDAQ series in one table, plus the info
    return pd.concat(rows), info


def clean_prices(dim: pd.DataFrame, px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Wide table: one column per index, one row per date (close prices) without weekends."""
    # split the two NASDAQ indices
    nasdaq, info = split_nasdaq(px)
    # names for all other indices (without the shared NASDAQ key)
    names = dim[dim["asset_key"] != config.NASDAQ_KEY][["asset_key", "ticker"]]
    # all other prices with their ticker name
    other = px[px["asset_key"] != config.NASDAQ_KEY].merge(names, on="asset_key")
    # put the other indices and the two NASDAQ series together
    long = pd.concat([other[["ticker", "date", "close"]], nasdaq])

    # Weekend rows (58, mostly special sessions in India/Taiwan) are dropped.
    # The next weekday's close-to-close return still covers the move.
    # keep only Monday-Friday (weekday 0-4)
    long = long[long["date"].dt.weekday < 5]
    # turn into a wide table: rows = dates, columns = indices
    wide = long.pivot(index="date", columns="ticker", values="close").sort_index()
    # give back the table and the NASDAQ info
    return wide, info


# --- 3. Returns --------------------------------------------------------------


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily change of each index, on its own trading days (holidays differ by country).

    No forward-filling: a holiday would create a 0 return followed by a catch-up.
    """
    # for each index: drop empty days, log of the price, difference to the day before
    return prices.apply(lambda s: np.log(s.dropna()).diff())


def weekly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Friday-to-Friday returns. Asia, Europe and the US all close within the same week,
    so the time-zone problem of daily data almost disappears."""
    # last price of each week (Friday), log, difference to last week, drop the first empty row
    return np.log(prices.resample("W-FRI").last()).diff().iloc[1:]


# --- 4. Outliers -------------------------------------------------------------


def flag_outliers(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Find suspicious daily returns and give the reason (they are not deleted).

    jump: |z| > OUTLIER_Z, z = return / trailing std (excluding the day itself),
          and no similar move elsewhere: the median |z| of the other indices on the
          same and the next day is below OUTLIER_MARKET_Z (next day catches Asia
          reacting to a US move).
    gap:  the return spans more than GAP_DAYS calendar days of missing data.
    """
    # usual daily move of each index over the last 250 days (without today)
    std = returns.rolling(config.OUTLIER_VOL_WINDOW, min_periods=60).std().shift(1)
    # today's move in "usual moves" (z = 5 means 5x the usual move)
    z = returns / std
    # how much the whole market moved today (median over all indices)
    market = z.abs().median(axis=1)
    # take the bigger of today and tomorrow (Asia reacts a day later)
    market = pd.concat([market, market.shift(-1)], axis=1).max(axis=1)
    # jump: this index moved more than 5x, while the market stayed calm
    jump = (z.abs() > config.OUTLIER_Z) & (market < config.OUTLIER_MARKET_Z).to_numpy()[:, None]

    # number of calendar days since the previous price, per index
    gap_days = prices.apply(lambda s: s.dropna().index.to_series().diff().dt.days)
    # gap: more than 14 days without a price
    gap = gap_days.reindex_like(returns) > config.GAP_DAYS

    # collect all flagged days in a list
    rows = []
    # first gaps, then jumps (a gap is the better explanation)
    for kind, mask in (("gap", gap), ("jump", jump)):  # gap first: the better explanation
        # all (date, index) pairs where the rule is True
        hits = mask.stack()
        # one row per flagged day
        for date, ticker in hits[hits].index:
            rows.append({"date": date.date(), "ticker": ticker, "reason": kind,
                         "ret": round(float(returns.at[date, ticker]), 4),
                         "z": round(float(z.at[date, ticker]), 1),
                         "market_abs_z": round(float(market[date]), 2)})
    # turn the list into a table
    table = pd.DataFrame(rows, columns=["date", "ticker", "reason", "ret", "z", "market_abs_z"])
    # each day only once (keep the first reason), sorted by date
    return table.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])


def drop_flagged(returns: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Copy of the returns with the flagged days set to empty."""
    # copy, so the original stays untouched
    out = returns.copy()
    # set each flagged day to empty
    for row in flags.itertuples():
        out.at[pd.Timestamp(row.date), row.ticker] = np.nan
    # give back the cleaned table
    return out


# --- 5./6. Regimes and correlations ------------------------------------------


def average_pairwise(corr: pd.DataFrame) -> float:
    """Average correlation over all pairs (each pair once, without an index with itself)."""
    # values above the diagonal of the matrix (= each pair once)
    upper = corr.to_numpy()[np.triu_indices(len(corr), k=1)]
    # their average, ignoring empty values
    return float(np.nanmean(upper))


def forbes_rigobon(rho: float, delta: float) -> float:
    """Correct a crisis correlation for higher volatility (Forbes & Rigobon 2002).

    delta = variance(crisis) / variance(normal) - 1 of the source market.
    Higher volatility alone raises measured correlation; this removes that effect.
    """
    # the Forbes-Rigobon formula
    return rho / np.sqrt(1 + delta * (1 - rho**2))


def regime_correlations(returns: pd.DataFrame, crisis: pd.Series) -> dict:
    """Correlations on crisis days vs. normal days, plus the corrected crisis value."""
    # crisis flag for every day (missing days = not crisis)
    crisis = crisis.reindex(returns.index).fillna(False).astype(bool)
    # split the returns: crisis days and normal days
    r_c, r_n = returns[crisis], returns[~crisis]
    # correlation matrix for each group (a pair needs at least 20 common days)
    corr_c, corr_n = r_c.corr(min_periods=20), r_n.corr(min_periods=20)

    # Forbes-Rigobon with the S&P 500 as the source of global stress.
    spx = "GSPC.INDX"
    # how much more the S&P 500 moves in crisis (variance ratio minus 1)
    delta = r_c[spx].var() / r_n[spx].var() - 1
    # corrected crisis correlation for every pair
    corr_fr = corr_c.apply(lambda col: forbes_rigobon(col, delta))
    # all results: matrices, delta, number of days, averages
    return {"crisis": corr_c, "normal": corr_n, "fr": corr_fr, "delta": delta,
            "n_crisis": int(crisis.sum()), "n_normal": int((~crisis).sum()),
            "avg_crisis": average_pairwise(corr_c), "avg_normal": average_pairwise(corr_n),
            "avg_fr": average_pairwise(corr_fr)}


def rolling_average_corr(returns: pd.DataFrame, window: int = config.ROLLING_CORR_DAYS,
                         step: int = 5) -> pd.Series:
    """Average correlation over a moving 63-day window (every 5th day, for the chart)."""
    # results: end date -> average correlation
    out = {}
    # move the window forward 5 days at a time
    for end in range(window, len(returns) + 1, step):
        # the last 63 days up to this point
        chunk = returns.iloc[end - window:end]
        # average correlation in this window
        out[returns.index[end - 1]] = average_pairwise(chunk.corr(min_periods=window // 2))
    # as a series: date -> value
    return pd.Series(out)


def sensitivity(returns: pd.DataFrame, vix: pd.Series, levels: list[float]) -> pd.DataFrame:
    """Repeat the crisis/normal comparison for several VIX thresholds."""
    # one row per threshold
    rows = []
    # for VIX 20, 24.4, 25, 30, 35, 40
    for level in levels:
        # crisis = VIX above this level
        res = regime_correlations(returns, vix > level)
        # keep the key numbers
        rows.append({"vix_level": round(level, 1), "crisis_obs": res["n_crisis"],
                     "avg_crisis": res["avg_crisis"], "avg_normal": res["avg_normal"],
                     "avg_crisis_fr": res["avg_fr"]})
    # as a table
    return pd.DataFrame(rows)


# --- Run everything ----------------------------------------------------------


def analyse() -> dict:
    """Run all steps of Task 2 and return all results."""
    # 1. read files and run checks
    dim, px, checks = load_raw()
    # 2. clean: split NASDAQ, drop weekends, wide table
    prices, nasdaq = clean_prices(dim, px)
    # take the VIX out: it is the crisis signal, not an equity index
    vix_px = prices.pop(config.VIX_TICKER)  # crisis signal, not an equity index

    # 3. daily returns
    daily = daily_log_returns(prices)
    # 4. find outliers
    flags = flag_outliers(daily, prices)
    # daily returns without the outliers (main result)
    daily_clean = drop_flagged(daily, flags)
    # weekly returns (time-zone check)
    weekly = weekly_log_returns(prices)

    # VIX for every day (non-US trading days take the last VIX)
    vix_daily = vix_px.reindex(daily.index).ffill()  # non-US trading days take last VIX
    # VIX per week: weekly average
    vix_weekly = vix_px.resample("W-FRI").mean().reindex(weekly.index)
    # 80% quantile of the VIX (24.4), extra threshold
    q80 = float(vix_px.quantile(0.8))
    # all thresholds for the sensitivity check, sorted
    levels = sorted(set(config.VIX_SENSITIVITY_LEVELS) | {q80})

    # main threshold: VIX 30
    level = config.VIX_CRISIS_LEVEL
    # 5./6. all results in one dictionary
    return {
        # data checks, NASDAQ info, outliers, VIX, quantile
        "checks": checks, "nasdaq": nasdaq, "flags": flags, "vix": vix_px, "q80": q80,
        # number of indices (30)
        "n_indices": prices.shape[1],
        # main result: daily, without outliers
        "daily": regime_correlations(daily_clean, vix_daily > level),
        # same with outliers (to show they don't change the result)
        "daily_with_outliers": regime_correlations(daily, vix_daily > level),
        # weekly
        "weekly": regime_correlations(weekly, vix_weekly > level),
        # sensitivity: daily and weekly
        "sens_daily": sensitivity(daily_clean, vix_daily, levels),
        "sens_weekly": sensitivity(weekly, vix_weekly, levels),
        # rolling correlation for the chart
        "rolling": rolling_average_corr(daily_clean),
    }


# --- 7. Report ---------------------------------------------------------------

# colours for the charts (dark theme)
BLUE, ORANGE, GREY = "#3987e5", "#d95926", "#8a94a6"
TEXT, MUTED, GRID = "#e6eaf2", "#9aa4b5", "#1f2a3c"


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Same look for every chart: dark background, fonts, grid."""
    # background, font, margins, height, legend, hover box
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color=MUTED, size=12),
                      margin=dict(l=8, r=8, t=30, b=8), height=height,
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right",
                                  font=dict(color=TEXT)),
                      hoverlabel=dict(bgcolor="#0f1726", font=dict(color=TEXT)))
    # grid lines on both axes
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _short(names) -> list[str]:
    """Shorter names for charts: "GDAXI.INDX" -> "GDAXI"."""
    return [n.replace(".INDX", "") for n in names]


def heatmap(corr: pd.DataFrame, title: str, diverging: bool = False) -> go.Figure:
    """Coloured correlation matrix (heatmap)."""
    # same order of indices in every heatmap
    order = corr.columns  # same order in every heatmap
    # colours: blue-grey-orange for differences, dark-to-blue for normal values
    scale = ([[0, "#2a78d6"], [0.5, "#2a2f3a"], [1, "#d95926"]] if diverging
             else [[0, "#0f1726"], [1, "#3987e5"]])
    # draw the matrix
    fig = go.Figure(go.Heatmap(
        z=corr.loc[order, order].values, x=_short(order), y=_short(order),
        colorscale=scale, zmin=-0.4 if diverging else 0, zmax=0.4 if diverging else 1,
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>", xgap=1, ygap=1))
    # common style, height 560
    fig = _style(fig, 560)
    # title
    fig.update_layout(title=dict(text=title, font=dict(color=TEXT, size=14)))
    # first index on top
    fig.update_yaxes(autorange="reversed")
    return fig


def report_body(res: dict) -> str:
    """The report content (cards and charts), used standalone and in the portal."""
    # short names: daily, weekly, daily with outliers
    d, w, dwo = res["daily"], res["weekly"], res["daily_with_outliers"]
    # order indices from least to most connected
    order = d["normal"].mean().sort_values().index  # least to most connected

    # put a matrix into this order
    def reorder(c):
        return c.loc[order, order]

    # list of all charts
    figs = []
    # Rolling correlation and VIX: two panels, one shared time axis (no dual y-axis).
    roll, vix = res["rolling"], res["vix"]
    # chart 0: two panels on top of each other
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                      row_heights=[0.6, 0.4],
                      subplot_titles=(f"Average pairwise correlation, rolling "
                                      f"{config.ROLLING_CORR_DAYS} days (daily returns)",
                                      "VIX close"))
    # top: rolling correlation (blue)
    f.add_trace(go.Scatter(x=roll.index, y=roll, name="Avg correlation",
                           line=dict(color=BLUE, width=2)), 1, 1)
    # bottom: VIX (orange)
    f.add_trace(go.Scatter(x=vix.index, y=vix, name="VIX", line=dict(color=ORANGE, width=1.5)),
                2, 1)
    # dotted line at VIX 30
    f.add_hline(y=config.VIX_CRISIS_LEVEL, line=dict(color=GREY, dash="dot"), row=2, col=1)
    figs.append(_style(f, 520))

    # charts 1-3: heatmaps normal, crisis, difference
    figs.append(heatmap(reorder(d["normal"]), f"Normal days (VIX ≤ {config.VIX_CRISIS_LEVEL})"))
    figs.append(heatmap(reorder(d["crisis"]), f"Crisis days (VIX > {config.VIX_CRISIS_LEVEL})"))
    figs.append(heatmap(reorder(d["crisis"] - d["normal"]),
                        "Difference: crisis minus normal", diverging=True))

    # Sensitivity to the VIX level.
    # chart 4: correlation for each VIX threshold (daily solid, weekly dashed)
    s_d, s_w = res["sens_daily"], res["sens_weekly"]
    f = go.Figure([
        go.Scatter(x=s_d.vix_level, y=s_d.avg_crisis, name="Crisis, daily",
                   mode="lines+markers", line=dict(color=ORANGE, width=2)),
        go.Scatter(x=s_d.vix_level, y=s_d.avg_normal, name="Normal, daily",
                   mode="lines+markers", line=dict(color=BLUE, width=2)),
        go.Scatter(x=s_w.vix_level, y=s_w.avg_crisis, name="Crisis, weekly",
                   mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dash")),
        go.Scatter(x=s_w.vix_level, y=s_w.avg_normal, name="Normal, weekly",
                   mode="lines+markers", line=dict(color=BLUE, width=2, dash="dash")),
    ])
    f = _style(f)
    f.update_xaxes(title="VIX threshold for 'crisis'")
    f.update_yaxes(title="Average pairwise correlation")
    figs.append(f)

    # NASDAQ duplicate: the two prices behind one key.
    # chart 5: ratio of the two NASDAQ prices, unclear period shaded orange
    n = res["nasdaq"]
    f = go.Figure([
        go.Scatter(x=n["hi"].index, y=n["hi"] / n["lo"], name="Higher / lower price",
                   line=dict(color=BLUE, width=1.5)),
    ])
    f.add_vrect(x0=n["ambiguous_from"], x1=n["ambiguous_to"], fillcolor=ORANGE, opacity=0.15,
                line_width=0, annotation_text="set to missing", annotation_font_color=TEXT)
    f = _style(f, 300)
    f.update_yaxes(title="Price ratio of the two rows")
    figs.append(f)

    # turn every chart into an HTML block
    divs = [fg.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False,
                                                                     "responsive": True})
            for fg in figs]

    # one row of the summary table (days, normal, crisis, corrected)
    def row(label, r):
        return (f"<tr><td>{label}</td><td class='num'>{r['n_normal']:,}</td>"
                f"<td class='num'>{r['n_crisis']:,}</td><td class='num'>{r['avg_normal']:.2f}</td>"
                f"<td class='num'>{r['avg_crisis']:.2f}</td>"
                f"<td class='num'>{r['avg_fr']:.2f}</td></tr>")

    # data checks and outliers
    c = res["checks"]
    flags = res["flags"]
    # one HTML table row per outlier
    flag_rows = "".join(
        f"<tr><td>{r.date}</td><td>{r.ticker.replace('.INDX', '')}</td><td>{r.reason}</td>"
        f"<td class='num'>{r.ret:+.1%}</td><td class='num'>{'' if pd.isna(r.z) else f'{r.z:.1f}'}"
        "</td></tr>"
        for r in flags.itertuples())
    # the duplicate key (NASDAQ)
    dup = c["duplicate_keys_in_master"][0]
    # crisis minus normal (e.g. +0.13)
    gap_low = d["avg_crisis"] - d["avg_normal"]
    # True if the corrected value is still above normal
    fr_above = d["avg_fr"] > d["avg_normal"]
    # date of the highest rolling correlation
    peak_date = roll.idxmax()

    # Other patterns, all computed from the data.
    # indices sorted from least to most connected
    connected = d["normal"].mean().sort_values()  # mean incl. the diagonal: ranking only
    # indices sorted by how much they rise in crisis
    rise = (d["crisis"] - d["normal"]).mean().sort_values()
    # all pairs once (upper triangle)
    pairs = d["normal"].where(np.triu(np.ones(d["normal"].shape, bool), k=1)).stack().dropna()
    # the 3 most similar pairs, as text
    top_pairs = ", ".join(f"{_short([a])[0]}/{_short([b])[0]} {v:.2f}"
                          for (a, b), v in pairs.sort_values().tail(3).items())
    # four sentences for "Other patterns"
    patterns = [
        f"Least connected on normal days: {', '.join(_short(connected.index[:4]))} (Asia); "
        f"most connected: {', '.join(_short(connected.index[-4:]))} (Europe).",
        f"Largest rise in crisis: {', '.join(_short(rise.index[-4:]))}.",
        f"Near-duplicate indices (normal-day correlation): {top_pairs}. They measure the same "
        "market and lift the average a little, in both regimes.",
        f"Daily vs. weekly: weekly correlations are higher in both regimes "
        f"({w['avg_normal']:.2f} vs. {d['avg_normal']:.2f} normal) - the time-zone effect.",
    ]
    # as an HTML list
    pattern_html = "".join(f"<li>{p}</li>" for p in patterns)

    # the whole report as HTML text (answer, table, charts, data traps, limitations)
    body = f"""
<h1>Index analysis: correlations in market distress</h1>
<p class="sub">{res['n_indices']} equity indices, 2008-2026. Crisis = VIX close above
{config.VIX_CRISIS_LEVEL}. Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}.</p>

<div class="card take"><h2>Answer</h2>
<p>{'Yes' if gap_low > 0 else 'No'}: on crisis days the average pairwise correlation
{'rises' if gap_low > 0 else 'changes'} from <b>{d['avg_normal']:.2f}</b> to
<b>{d['avg_crisis']:.2f}</b> (daily) and from <b>{w['avg_normal']:.2f}</b> to
<b>{w['avg_crisis']:.2f}</b> (weekly), for every VIX threshold tested.
After the Forbes-Rigobon volatility adjustment the crisis value is
<b>{d['avg_fr']:.2f}</b> (daily) and <b>{w['avg_fr']:.2f}</b> (weekly),
{'still above' if fr_above else 'below'} normal:
{'the link between markets itself gets stronger.' if fr_above else
 'the rise can be explained by higher volatility alone (interdependence, not contagion).'}
For risk management the raw number is what counts: diversification across regions is
weakest exactly when losses are largest.</p></div>

<div class="card"><h2>Summary table (average pairwise correlation)</h2>
<table><tr><th></th><th class='num'>normal obs</th><th class='num'>crisis obs</th>
<th class='num'>normal</th><th class='num'>crisis</th><th class='num'>crisis, FR-adjusted</th></tr>
{row('Daily returns, outliers excluded (main)', d)}
{row('Daily returns, outliers included', dwo)}
{row('Weekly returns (Friday to Friday)', w)}
</table>
<p class="note">Variance ratio of the S&amp;P 500 crisis vs. normal (daily): {d['delta'] + 1:.1f}x.
Crisis minus normal (daily): {gap_low:+.2f}.</p></div>

<div class="card">{divs[0]}<p class="note">The rolling correlation moves with the VIX.
Highest value {roll.max():.2f} in {peak_date:%B %Y}, lowest {roll.min():.2f} in
{roll.idxmin():%B %Y}.</p></div>
<div class="card"><h2>Other patterns</h2><ul>{pattern_html}</ul></div>
<div class="grid3"><div class="card">{divs[1]}</div><div class="card">{divs[2]}</div>
<div class="card">{divs[3]}</div></div>
<div class="card"><h2>Sensitivity to the crisis threshold</h2>{divs[4]}
<p class="note">The 80% quantile of the VIX is {res['q80']:.1f}. The conclusion does not depend
on the exact threshold. VIX &gt; 30 is used as the main level because it is a fixed,
widely used stress level and does not change when the sample changes.</p></div>

<div class="card"><h2>Data traps and how they were handled</h2><ol>
<li><b>Duplicate key.</b> {html.escape(dup['tickers'])} share asset_key {dup['asset_key']}.
A naive join gives {c['naive_join_rows']:,} rows instead of {c['price_rows']:,} and two prices
per day. Split by price level (Composite higher until 2020, NASDAQ 100 higher from mid-2021);
{res['nasdaq']['days_set_missing']} days between {res['nasdaq']['ambiguous_from']:%Y-%m-%d} and
{res['nasdaq']['ambiguous_to']:%Y-%m-%d}, where the two cross, are set to missing.
Check against published closes: {'passed' if res['nasdaq']['anchors_ok'] else 'FAILED'}.
{divs[5]}</li>
<li><b>Time zones.</b> Asia closes before Europe opens, Europe before the US closes.
Daily correlations between regions are therefore understated; weekly returns fix most of this.</li>
<li><b>Outliers</b> ({len(flags)} flagged, excluded from the main result, shown above with and
without): a return of more than {config.OUTLIER_Z} trailing standard deviations while the other
indices stayed calm, or a return across a data gap of more than {config.GAP_DAYS} days.
Several are real events (e.g. elections, the 2015 franc shock) - they are flagged for review,
not deleted.
<table><tr><th>Date</th><th>Index</th><th>Reason</th><th class='num'>Return</th>
<th class='num'>z</th></tr>{flag_rows}</table></li>
<li><b>Calendar.</b> {c['weekend_rows']} weekend rows dropped (special sessions; the next return
covers the move). {c['zero_volume_share']:.0%} of rows have volume 0 - volume is not used.
Different holidays: returns are taken on each index's own trading days, correlations use
pairwise complete observations. Late starts (UK 100 from 2010, TOPIX from 2011) only shorten
their pairs. close = adjusted_close in every row.</li>
</ol></div>

<div class="card"><h2>Assumptions and limitations</h2><ul>
<li>Higher crisis correlation is partly mechanical: when volatility rises, measured correlation
rises even if the link between markets is unchanged (Forbes-Rigobon). The adjustment shown uses
the S&amp;P 500 as the single source market - a simplification.</li>
<li>Forbes-Rigobon assumes the crisis comes from one source market and no common global
shock. With common shocks (2008, 2020) it over-corrects, so the true value lies between the
raw and the adjusted number.</li>
<li>The VIX measures US expected volatility; a local crisis (e.g. Asia) may not show up.</li>
<li>Crisis days cluster in a few episodes (2008, 2020), so the crisis estimate rests on few
independent events.</li>
<li>Pearson correlation is sensitive to single large moves; hence the outlier treatment.</li>
<li>Index levels in local currency; no FX effect.</li>
</ul></div>"""
    # give back the HTML text
    return body


def build_report(res: dict) -> str:
    """Standalone page: output/index_analysis.html."""
    # the report content
    body = report_body(res)
    # wrap it into a full HTML page with styles and the chart library
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Index Analysis</title><script>{get_plotlyjs()}</script>
<style>
 body {{ margin:0; background:#0b1220; color:{TEXT}; font:14px/1.5 Inter,"Segoe UI",sans-serif; }}
 main {{ max-width:1400px; margin:0 auto; padding:28px 24px 40px; }}
 h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:0 0 10px; }}
 .sub, .note {{ color:{MUTED}; }} .card {{ background:#111a2b; border:1px solid {GRID};
   border-radius:12px; padding:18px 20px; margin-bottom:16px; min-width:0; }}
 .take {{ border-left:4px solid {ORANGE}; }} .take p {{ font-size:16px; margin:0; }}
 .grid3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
 table {{ width:100%; border-collapse:collapse; }} td, th {{ padding:6px; text-align:left;
   border-bottom:1px solid {GRID}; }} th {{ color:{MUTED}; font-size:12px; }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums; }} li {{ margin-bottom:10px; }}
 a {{ color:{BLUE}; }}
 @media (max-width:1100px) {{ .grid3 {{ grid-template-columns:1fr; }} }}
</style></head><body><main><p><a href="dashboard.html">&larr; FX risk dashboard</a></p>
{body}</main></body></html>"""


def save_outputs(res: dict) -> None:
    """Save the report and the CSV tables in output/."""
    # create the output folder if it is missing
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # save the HTML report
    config.INDEX_OUTPUT_PATH.write_text(build_report(res), encoding="utf-8")
    # save the list of outliers
    res["flags"].to_csv(config.OUTPUT_DIR / "index_outliers.csv", index=False)
    # save the sensitivity results (daily and weekly in one file)
    pd.concat([res["sens_daily"].assign(freq="daily"), res["sens_weekly"].assign(freq="weekly")]
              ).to_csv(config.OUTPUT_DIR / "index_sensitivity.csv", index=False)
    # save the 4 correlation matrices (daily/weekly x normal/crisis)
    for name in ("normal", "crisis"):
        res["daily"][name].round(3).to_csv(config.OUTPUT_DIR / f"index_corr_daily_{name}.csv")
        res["weekly"][name].round(3).to_csv(config.OUTPUT_DIR / f"index_corr_weekly_{name}.csv")


def main() -> None:
    """Command line: run the analysis, save, print the key numbers."""
    # run everything
    res = analyse()
    # save report and tables
    save_outputs(res)
    # short names
    d, w = res["daily"], res["weekly"]
    # print the result in the terminal
    print(f"Average pairwise correlation  normal -> crisis (FR-adjusted)\n"
          f"  daily : {d['avg_normal']:.2f} -> {d['avg_crisis']:.2f} ({d['avg_fr']:.2f})\n"
          f"  weekly: {w['avg_normal']:.2f} -> {w['avg_crisis']:.2f} ({w['avg_fr']:.2f})\n"
          f"Outliers flagged: {len(res['flags'])}. "
          f"NASDAQ anchors ok: {res['nasdaq']['anchors_ok']}\n"
          f"Report: {config.INDEX_OUTPUT_PATH}")


# run main() only when the file is started directly
if __name__ == "__main__":
    main()
