"""FX risk of a USD position seen from EUR: VaR, ES, parametric benchmark, backtest.

Direction of the quote:
    The ECB rate S(spot rate) is USD per 1 EUR.  EUR value = USD amount / S
    S goes UP   (stronger euro)  -> fewer EUR for our USD -> LOSS
    S goes DOWN (weaker euro)    -> more EUR for our USD  -> gain

Method (historical simulation, overlapping 5-day returns):
    r5_t   = ln(S_t / S_{t-5})               5-trading-day log return of the rate
    loss_t = 1 - exp(-r5_t)                  relative loss of the EUR value (share of position)
    VaR    = 99% quantile of loss_t over the last WINDOW_DAYS observations
    ES     = average of loss_t at or above the 97.5% quantile
    EUR    = share * current EUR value of the position

All losses are positive numbers (loss = +, gain = -).
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist 

import duckdb
import numpy as np
import pandas as pd

import config

# --- Building blocks ---------------------------------------------------------


def load_rates(db_path: Path = config.DUCKDB_PATH, series_id: str = config.SERIES_ID) -> pd.Series:
    """Read the stored ECB rates as a date-indexed Series (USD per EUR)"""
    """13.11.2026->1,1052"""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT rate_date, rate FROM fx_rates WHERE series_id = ? ORDER BY rate_date",
                     [series_id]).df()
    con.close()
    return pd.Series(df["rate"].values, index=pd.to_datetime(df["rate_date"]), name="rate")


def eur_value(rate: float | pd.Series, usd: float = config.POSITION_USD):
    """EUR value of our USD: USD / rate."""
    return usd / rate


def horizon_log_returns(rates: pd.Series, horizon: int = config.HORIZON_DAYS) -> pd.Series:
    """Rate change over 5 days, for every day."""
    # log of each rate -> minus the rate 5 days ago -> drop the first 5 empty rows
    return np.log(rates).diff(horizon).dropna()


def relative_loss(log_return):
    """
    How much of our money we lose if the rate moves like this (e.g. 0.02 = 2%).
    EUR value after the move: loss = 1 - exp(-r)
    r = rate change over 5 days. Loss = 1 - old_rate / new_rate
    """
    return 1 - np.exp(-log_return)


def historical_var(losses, confidence: float = config.VAR_CONFIDENCE) -> float:
    """VaR: the loss that is exceeded only on the worst 1% of days."""
    # sort the losses, take the value at 99% (about the 8th worst of 750)
    return float(np.quantile(losses, confidence))


def expected_shortfall(losses, confidence: float = config.ES_CONFIDENCE) -> float:
    """ES = average loss in the tail at or beyond the VaR at the same confidence."""
    losses = np.asarray(losses) # same losses, converted to numpy so we can filter them
    # return border of the worst 2.5% -> keep losses beyond it -> average them
    return float(losses[losses >= np.quantile(losses, confidence)].mean())


def parametric_var(daily_log_returns, confidence: float = config.VAR_CONFIDENCE,
                   horizon: int = config.HORIZON_DAYS) -> float:
    """
    Simple VaR check with a bell curve, to compare with the main VaR
    sigma_h = sigma_1day * sqrt(h);  worst rate move = z * sigma_h;  loss = 1 - exp(-move)
    """
    #usual rate move over 5 days
    #Bessel's correction
    sigma_h = float(np.std(daily_log_returns, ddof=1)) * math.sqrt(horizon)
    # how far out the worst 1% lies on a bell curve
    z = NormalDist().inv_cdf(confidence)
    # worst move, turned into a loss
    return float(relative_loss(z * sigma_h))


def kupiec_test(n: int, exceptions: int,
                p: float = 1 - config.VAR_CONFIDENCE) -> tuple[float, float]:
    """
    Checks if the VaR was broken about 1% of the time, as it should be
    """
    x = exceptions  # number of VaR breaches/Ausnahme
    phat = x / n  # % of days with a breach

    def xlogy(a, b):  # a * log(b), but gives 0 if a is 0 (avoids an error)
        return 0.0 if a == 0 else a * math.log(b)

    log_l0 = xlogy(n - x, 1 - p) + xlogy(x, p)# how likely our breaches are if the true rate is 1%
    log_l1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)# how likely they are with our actual rate
    lr = -2 * (log_l0 - log_l1) # difference between the two = test value
    p_value = math.erfc(math.sqrt(max(lr, 0) / 2))  # turn test value into p-value
    return lr, p_value


def basel_zone(exceptions_in_250: int) -> str:
    """Basel traffic light for a 99% VaR over 250 observations."""
    if exceptions_in_250 <= 4:
        return "Green"
    if exceptions_in_250 <= 9:
        return "Yellow"
    return "Red"


# --- Today's risk figures ----------------------------------------------------


def risk_snapshot(rates: pd.Series, window: int = config.WINDOW_DAYS) -> dict:
    """Today's main numbers: VaR, ES, check VaR, position value."""
    # losses of the last 750 days
    losses = relative_loss(horizon_log_returns(rates)).iloc[-window:]
    # daily rate changes of the last 750 days (for the check VaR)
    daily = np.log(rates).diff().dropna().iloc[-window:]
    # today's EUR value of our USD (USD / last rate)
    value = float(eur_value(rates.iloc[-1]))

    # main VaR, as %
    var_pct = historical_var(losses)
    # ES, as %
    es_pct = expected_shortfall(losses)
    # simple VaR (normal distribution), as %
    par_pct = parametric_var(daily)
    return {
        "as_of": rates.index[-1], # date of the last rate
        "rate": float(rates.iloc[-1]), # last rate
        "value_eur": value, # our USD in EUR today
        "value_eur_prev": float(eur_value(rates.iloc[-1 - config.HORIZON_DAYS])), # our USD in EUR 5 days ago
        "var_pct": var_pct, "var_eur": var_pct * value, # VaR in % and in EUR
        "es_pct": es_pct, "es_eur": es_pct * value, # ES in % and in EUR
        "param_var_pct": par_pct, "param_var_eur": par_pct * value, # simple VaR in % and in EUR
        "sqrt5_var_pct": historical_var(relative_loss(daily)) * math.sqrt(config.HORIZON_DAYS), # 1-day VaR x sqrt(5), only for comparison
        "n_scenarios": len(losses),# how many scenarios (750)
        "window": window,# window size in days
    }





# --- Rolling history and backtest --------------------------------------------


def rolling_risk(rates: pd.Series, window: int = config.WINDOW_DAYS) -> pd.DataFrame:
    """VaR/ES for every past day, using only data available on that day (no look-ahead).

    Columns (share of position):
        var, es, param_var  - risk estimate made at the end of day t
        fwd_loss            - loss actually realised from t to t+5 (known only later)
        exception           - fwd_loss > var
    """
    h = config.HORIZON_DAYS # 5 days
    loss = relative_loss(horizon_log_returns(rates))  # 5-day loss for every day
    daily = np.log(rates).diff() # daily rate change

    df = pd.DataFrame({"loss": loss}) # table with one column: loss
    roll = df["loss"].rolling(window) # moving window: always the last 750 days
    df["var"] = roll.quantile(config.VAR_CONFIDENCE)  # VaR for every day
    df["es"] = roll.apply(expected_shortfall, raw=True) # ES for every day
    sigma = daily.rolling(window).std().reindex(df.index) # usual daily move for every day
    df["param_var"] = relative_loss(NormalDist().inv_cdf(config.VAR_CONFIDENCE)
                                    * sigma * math.sqrt(h)) # simple VaR for every day
    df["fwd_loss"] = df["loss"].shift(-h) # real loss in the next 5 days
    df["value_eur"] = eur_value(rates.reindex(df.index)) # our USD in EUR on that day
    df = df.dropna(subset=["var"]) # drop first days without enough history
    df["exception"] = df["fwd_loss"] > df["var"] # breach: real loss bigger than VaR
    return df # table: one row per day




def backtest_summary(roll: pd.DataFrame) -> dict:
    """Backtest: how often VaR was broken, plus Kupiec and Basel light."""
    done = roll.dropna(subset=["fwd_loss"])  # drop last 5 days, no result yet
    h = config.HORIZON_DAYS # 5 days

    n, x = len(done), int(done["exception"].sum())# number of days, number of breaches
    lr, p = kupiec_test(n, x)# Kupiec test on all days

    sub = done.iloc[::-1].iloc[::h].iloc[::-1]  # only every 5th day, so windows don't overlap
    n_sub, x_sub = len(sub), int(sub["exception"].sum())# days and breaches in this subset
    lr_sub, p_sub = kupiec_test(n_sub, x_sub)  # Kupiec test on every 5th day

    last = done.iloc[-config.BACKTEST_DAYS:] # last 250 days
    x_last = int(last["exception"].sum())# breaches in the last 250 days
    return { # all backtest results for the dashboard
        "n": n, "exceptions": x, "rate": x / n, "kupiec_lr": lr, "kupiec_p": p, # all days
        "n_nonoverlap": n_sub, "exceptions_nonoverlap": x_sub, "kupiec_p_nonoverlap": p_sub, # every 5th day
        "n_last": len(last), "exceptions_last": x_last, # last 250 days
        "kupiec_p_last": kupiec_test(len(last), x_last)[1], # Kupiec on last 250 days
        "basel_zone": basel_zone(x_last), # Green / Yellow / Red
        "start": done.index[0], "end": done.index[-1],  # first and last tested day
    }


def save_results(roll: pd.DataFrame, db_path: Path = config.DUCKDB_PATH) -> None:
    """Store the risk history in DuckDB so every dashboard number can be traced."""
    out = roll.reset_index(names="rate_date") # turn the date into a normal column
    out["var_eur"] = out["var"] * out["value_eur"] # VaR in EUR
    out["es_eur"] = out["es"] * out["value_eur"]# ES in EUR
    con = duckdb.connect(str(db_path)) # open the database
    con.register("r", out) # make the table visible to SQL
    con.execute("CREATE OR REPLACE TABLE fx_risk_history AS SELECT * FROM r")  # save it, overwrite old one
    con.close()# close the database
