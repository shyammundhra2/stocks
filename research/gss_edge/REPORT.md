# GSS edge hunt: backtest report (branch `gss-edge-research`, nothing pushed, nothing live changed)

## Method
- Data: Yahoo `query1` daily adjusted closes, 2006-01 to 2026-11-25, 42 usable GSS universe ETFs plus IEF, SHY, BIL (and ^VIX for the live capitulation trigger). Short null gaps are forward-filled (max 5 days).
- Baseline: the repo's validated trend-book simulation (`backtest/backtest_gss_whipsaw_buf_sweep.py`): ER router, slope x R2 sizing, 15% cap, whole book to cash when the SPY-200DMA gate is off (exit 0.96x, re-entry 1.04x), monthly rebalance.
  This is a proxy for the live Kelly/covariance optimizer, not the optimizer itself.
- Rules: decide at close d, fill at close d+1 (next-day fills), 5 bps per unit of turnover, cash earns BIL (SHY before BIL's 2007-05 launch).
  Gate state is evaluated daily; trades happen at month-end or on a gate flip.
- Sharpe is computed on excess return over BIL. Periods: 2007-2015 and 2016-now.
- Pass rule: Sharpe > baseline in BOTH periods AND maxDD no more than 1pp worse in EACH period.
- Variants tried: **30**; passed the rule: **12**. All 30 are in `results_all.csv`. Grids were declared before running and nothing was tuned per period.

## Baseline (same engine)
| | Sharpe | maxDD |
|---|---|---|
| 2007-2015 | 0.108 | -22.5% |
| 2016-now | 0.776 | -29.7% |

## Four requested ideas
1. **Faster re-entry: FAIL.** 1.00x: Sharpe -0.01 / 0.68, DD -33.8% / -32.5%. 1.02x: 0.07 / 0.71. 50DMA-rising: Sharpe 0.25 / 0.78 (tie), but DD -25.8% in 2007-15 (3.3pp worse). Earlier re-entry just re-buys the whipsaw.
2. **Risk-off haven (IEF/SHY if it beats BIL over 126d, 50% of book): PASS** (see table). The gain is small.
3. **Multi-horizon momentum: FAIL.** 2-of-3 gives Sharpe 0.08 / 0.74; 3-of-3 gives 0.00 / 0.67 (cuts 2016+ DD to -21.6% but costs Sharpe in both periods).
4. **Capitulation reserve inside the book: PASS** (carve-out 20% into SPY on the live VIX-z/RSI14 trigger, exit on a new 252d high). See the control caveat below.

## Survivors (baseline: Sharpe 0.108 / DD -22.5% in 2007-15, 0.776 / -29.7% in 2016-now)
The last column is the paired 21-day block-bootstrap P(variant Sharpe > baseline Sharpe).

| variant | Sh 07-15 | DD 07-15 | Sh 16-now | DD 16-now | P(Sh>base) 07-15 / 16-now |
|---|---|---|---|---|---|
| 2a haven IEF 50% | 0.152 | -23.2% | 0.794 | -28.2% | 0.85 / 0.82 |
| 2b haven SHY 50% | 0.124 | -22.6% | 0.776 | -29.6% | 0.96 / 0.62 |
| 2c haven best of IEF/SHY 50% | 0.162 | -23.1% | 0.797 | -28.2% | 0.90 / 0.87 |
| 4a reserve carve-out 20% | 0.201 | -17.9% | 0.848 | -22.7% | 0.90 / 0.96 |
| 4b reserve from idle cash (<=20%) | 0.127 | -22.6% | 0.816 | -29.3% | 0.64 / 0.94 |
| 10 idle cash to IEF/SHY if it beats BIL | 0.168 | -23.4% | 0.793 | -28.2% | 0.92 / 0.80 |
| 11 haven best + idle reserve | 0.182 | -22.8% | 0.840 | -27.9% | 0.87 / 0.99 |
| 12 haven best + carve reserve | 0.266 | -18.1% | 0.877 | -21.1% | 0.97 / 0.98 |
| neighbours: haven 25% / 75% | 0.136 / 0.187 | -22.8% / -23.5% | 0.787 / 0.806 | -28.9% / -27.4% | 0.90/0.88, 0.89/0.87 |
| neighbours: reserve 10% / 30% | 0.154 / 0.249 | -20.0% / -17.1% | 0.811 / 0.878 | -25.6% / -20.8% | 0.92/0.97, 0.88/0.94 |

Neighbours pass, so no knife-edge, but haven 100% did NOT pass (DD -23.8% is 1.3pp worse in 07-15) and idle reserve at a 30% cap did not either.

## Honest caveats
- **Most of the Sharpe gains are statistically weak.** Bootstrap probabilities are mostly 0.85-0.97, not >0.99. Pass-by-a-hair: 2b (2016+ Sharpe 0.776 vs 0.776; effectively a tie).
  With 30 variants and 12 passes (many are neighbours or combos of the same two ideas), expect some of this to be selection luck.
- **Reserve (4a) control:** holding 20% in BIL with NO capitulation deploy (C1) leaves Sharpe at baseline (0.11 / 0.78) and cuts DD to -18.3% / -23.7%.
  So most of 4a's drawdown improvement is plain de-risking. Only the Sharpe lift (0.11 to 0.20, 0.78 to 0.85) is attributable to the capitulation deploy, and it rests on a handful of episodes (2008, 2011, 2015, 2018, 2020, 2022).
  The reserve also lowers CAGR in calm years, which matters if beating SPY ever becomes a goal (your stated objective is drawdown control).
- **Haven (2):** the effect is small. Two regimes dominate: the 2008 flight to Treasuries and IEF's 2022 loss, which the beat-BIL test sidesteps. It is not a large edge.
- Failed ideas I tested myself: vol-managed exposure on the book (target 10/15/20%), breadth-halve gate, book-drawdown breaker, SPY 20d vol halve, death-cross exit. None passed (most fail 2016+ Sharpe; death-cross passes 2016+ Sharpe but worsens DD and fails 2007-15 Sharpe).
  Vol-managed exposure passing in the earlier stock-sleeve work did not carry over to this book.
- The baseline is a proxy for the live optimizer; the live book (which spares DEFENSIVE_ASSETS in the gate, uses covariance Kelly sizing) may respond differently. Revalidate any survivor on the live engine before touching GSS.
- ^VIX is an extra Yahoo series beyond the requested universe, used only for the capitulation trigger.

## `_vol_headroom_deploy` bug: CONFIRMED (`macro/indicators/signals.py:149`)
`check_vol_headroom_bug.py` imports the live function and patches `get_portfolio_summary` in-process only:
- `weights = {}` returns **$0** (the `if not weights: return 0` guard).
- A book with one zero-weight entry returns $247,000; `{'GLD': 0.10}` returns $224,000.
- `get_trends` builds `summary["weights"]` from names with weight > 0 only, so an all-cash book (gate off, or nothing trending) gives `{}`, hence deploy = $0 even though that is exactly when vol headroom is largest.
  Cold start (summary not yet computed) hits the same path.
- Not fixed (lockdown). Suggested fix: distinguish "summary unavailable" (key missing / None, keep returning 0) from "available but empty book" (sig_p = 0, so w_max = vol_cap/sig_q).
  Cap w_max sensibly (a 15% cap on a ~20%-vol QQQ gives about 75% of the book).
  Note that RSI2 BUY also requires QQQ > 200DMA, so the empty-book case occurs when the SPY gate is still off while QQQ has already recovered above its 200DMA, or with nothing trending.

## Files
`fetch_data.py`, `prices.csv`, `engine.py`, `run_ideas.py` (batch 1), `run_batch2.py`, `make_report.py`, `check_vol_headroom_bug.py`, `results_all.csv`, `survivors_table.md`, `baseline_returns.csv`.

---
# Update 2026-11-26: overlays wired into the GSS code, re-backtested with the LIVE optimizer

## What changed in the code (branch `gss-edge-research` only; not pushed, not merged)
- `macro/indicators/overlays.py` (new, pure functions): `pick_haven` (IEF/SHY if the better 126d return beats BIL, else none) and `book_overlay` (scales optimizer weights by 1-20%, sizes an SPY reserve from the capitulation ladder, fills idle cash up to 50% of the book with the haven **only while the SPY gate is risk-off**; it never displaces defensives the optimizer already holds).
- `macro/indicators/signals.py`: capitulation logic extracted to a pure `capitulation_state(vix, spy)`; `get_vix_signal` calls it. Verified identical to the old inline version on 40 historical dates (`test_signals_refactor.py`).
- `macro/indicators/portfolio.py` `get_trends`: applies the overlays after the optimizer/throttle and exposes `summary["overlay"]`; overlay legs are included in `summary["weights"]` (so `_vol_headroom_deploy` sees them). Kill switch: `overlays.OVERLAYS_ENABLED`.
- `macro/indicators/data.py`: SHY and BIL added to the shared download list.
- The `_vol_headroom_deploy` empty-book bug is still unfixed (not asked).

## Live-optimizer replay (`live_replay.py`, `run_replay.py`, `replay_sim.py`, `run_live_backtest.py`)
The real `get_trends()` (router, sizing, covariance optimizer, gate, throttle) is called at 257 historical decision dates (month-ends + gate flips) on adjusted OHLCV from Yahoo. Only the data feeds are substituted (price store -> trailing window, VIX/VIX3M term structure, ML regime stub; `_effective_vol_cap` no longer uses ml_slow, so this changes no sizing). Same rules: next-day fill, 5 bps, cash earns BIL. The wired `get_trends` overlay output matched the offline overlay on all 14 sampled dates (`check_integration.py`).

| (Sharpe on excess over BIL, maxDD) | 2007-2015 | 2016-now | Rule |
|---|---|---|---|
| **Baseline (live optimizer, no overlays)** | 0.102 / -8.0% | 0.373 / -5.4% | |
| Haven only | 0.252 / -10.6% | 0.437 / -5.4% | **FAIL** (DD 2.6pp worse in 2007-15) |
| Reserve folded in only | 0.266 / -9.1% | 0.540 / -7.2% | **FAIL** (DD 1.1pp worse 07-15, 1.8pp worse 16-now) |
| Haven + reserve | 0.418 / -9.3% | 0.626 / -7.6% | **FAIL** (DD 1.3pp and 2.2pp worse) |

Sensitivity to the data window the optimizer sees (252 bars ~ the live 1y store vs 1000 bars): baseline 0.113/0.375, haven+reserve 0.424/0.597. Same conclusion.

**Verdict: neither change passes your rule on the live optimizer.** Sharpe rises in both periods for all three, which agrees with the proxy, but drawdown is worse by more than 1pp. The live book is much lower-vol than the proxy (baseline DD only -8%/-5%), so 1pp is a tight tolerance, and holding 50% IEF (2013 taper) or a 20% SPY reserve (2020, 2022) adds real drawdown relative to that.
Informational, **chosen after seeing the failure so NOT a clean pass** (3 extra variants on the same data): haven 25% -> Sh 0.191/0.408, DD -8.6%/-5.4% (DD 0.6pp worse in 2007-15, inside the 1pp tolerance); reserve 10% -> 0.218/0.480, DD -6.6%/-5.2%; both -> 0.311/0.520, DD -6.7%/-5.2%. These pass the rule but would need fresh data to trust.
Recommendation: do not run the overlays at full size live. If kept, set the sizes before UI testing (`RESERVE_FRAC`, `HAVEN_FRAC`) or flip `OVERLAYS_ENABLED=False`.
