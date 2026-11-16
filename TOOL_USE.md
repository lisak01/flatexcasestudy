# Tool use

## Tools
| Tool | What it helped with |
|---|---|
| Claude (Anthropic), chat and coding agent | Drafting code module by module, chart styling and layout, explaining risk methods, reviewing README and slides |
| pytest, ruff, GitHub Actions | Checking every change automatically |

## How I worked
- I set the scope and the design decisions: local DuckDB as required, one adapter per source,
  hard vs. soft checks, historical VaR as main measure, VIX > 30 as crisis definition.
- I used the AI where it saves the most time: boilerplate code, chart styling and layout.
  The AI drafted one module at a time; I reviewed and adjusted each module before moving on
  and asked for an explanation of every line I could not explain myself.
- That gave me the time for the parts that need my judgement: method choice, data checks
  and interpreting the results. In daily work I would use it the same way.
- My background is data engineering, not banking regulation. I used the AI as a tutor to learn
  the risk concepts quickly (VaR, Expected Shortfall, Kupiec test, Basel traffic light) and checked
  the key rules against the original Basel documents.
- I did not accept numbers from the AI. Every figure is reproduced by `python run_all.py`.

## How I checked the output
| What | Check |
|---|---|
| EUR value | Recomputed by hand: 100,000 / 1.1224 = 89,095 |
| Quote direction | Test: a stronger euro must be a loss |
| VaR and ES | Tests on a small loss series with a known quantile |
| Parametric VaR | Test against the closed formula |
| No look-ahead | Test: VaR on day t uses only data up to t |
| Daily ETL | Test: a second daily run adds 0 rows |
| ECB holiday rows | Found because my hard check stopped the first real load; only these rows are dropped |
| NASDAQ split | Checked against three published closing values |
| Code | All tests and ruff run in CI on every push |

## Responsibility
The data workflow, the risk assessment, the analysis and the conclusions are mine.
I can trace every dashboard number back to the ECB source.