# When to use mean reversion (MR) vs trend: regime-switching study
Branch `gss-edge-research`. Nothing pushed, merged or changed live. Code: `regime_study.py`, `run_regime.py`, `run_regime_robust.py`. Results: `regime_results.csv` (all 184 rules), `regime_passers_robustness.csv`.

## Bottom line
**No switching rule was demonstrated to beat the baselines out of sample.** 8 of 184 rules clear the stated bar (both periods), but that is only a little above the ~2.3 expected from random regimes. Rules selected on one period do not carry to the other, and doubling costs removes the edge, in at least one period, from all 8. The best lead is a single signal, *days since the 252-day low*, and it is not established. **Always-trend is the right default**; MR is dominated in both periods.

## Setup
- Data: Yahoo query1 adjusted OHLCV (2005 to 2026-11-26) for the 42 GSS universe ETFs, SPY, ^VIX, ^VIX3M, BIL. Same rules as before: next-day fills, 5 bps on all turnover, cash earns BIL, Sharpe on excess over BIL.
- **Trend sleeve:** monthly rebalance; hold names with price > 200DMA, > 50DMA and positive 20d slope, sized by slope x R2, 15% cap, whole sleeve to cash when the SPY-200DMA gate (0.96x / 1.04x) is off. (No ER filter: the market-level switch replaces it.)
- **MR sleeve:** daily; buy a name when RSI2 < 10 and it is above its 200DMA, 10% per slot (max 10), exit after 5 days or RSI2 > 65. Parameters were fixed up front and not tuned.
- **Baselines:** always-trend, always-MR, static 50/50 mix. A rule must beat the Sharpe of **all three** in **both** periods (2007-2015, 2016-now) and have maxDD no more than 1pp worse than the best-Sharpe baseline in each period.
- **Switching** runs through the same turnover/cost accounting as everything else (the composite target book is traded), so switch costs are included.

| Baseline | Sharpe / CAGR / maxDD 2007-2015 | 2016-now |
|---|---|---|
| Always-trend | 0.347 / 4.4% / -18.8% | 0.418 / 8.0% / -24.2% |
| Always-MR | 0.095 / 1.0% / -24.2% | 0.218 / 4.3% / -30.9% |
| Static 50/50 | 0.251 / 2.9% / -17.5% | 0.370 / 6.4% / -20.8% |

## Candidate signals (12, all causal, daily at the close)
VIX level; VIX/VIX3M term structure; implied minus realized vol (VIX - SPY 21d RV); SPY ADX(14); SPY efficiency ratio (20d); breadth (% of universe above its 200DMA); rate of drawdown (change in SPY drawdown-from-252d-high over 21d); average 63d pairwise correlation of 7 sector ETFs; and the **four new bottom signals**:
- days since the last local bottom and % gain since it, with the bottom defined two causal ways: (A) lowest close of the trailing 252 days; (C) a 10% zig-zag (a low is only "confirmed" after a 10% rally off it, so no look-ahead).

**No look-ahead, no fitted thresholds:** each signal is converted to its *expanding-window* percentile (5d-median smoothed) and used at fixed cutoffs, MR when the signal is in the top/bottom third or above/below the median (4 modes per signal, both directions, since I did not assume which direction favours MR). Combos are ANDs of a bottom signal with each other signal (median cutoffs, all four direction pairs), plus time x gain pairs. Total **184 rules** (48 singles + 136 combos). Nothing else was tried.

## Results
- **8 of 184 rules pass** the stated rule. 17 beat always-trend's Sharpe in both periods; 46 and 49 did in one period each (about a quarter, so one-period wins are the norm, not evidence).
- **Null calibration:** 400 random persistent regime sequences run through the same test pass at 1.3%, i.e. **~2.3 false passes expected among 184 rules**. 8 is above that but the rules are heavily overlapping.
- **Cross-period selection fails.** Rank correlation of Sharpe margin vs always-trend between the two periods across all rules is only +0.16. Top-10 rules chosen on 2007-2015 average +0.149 there but **-0.044** in 2016-now (3/10 positive); chosen on 2016-now, +0.167 vs **-0.047** in 2007-2015 (3/10 positive).
- **Cost robustness:** at 2x costs (10 bps) all 8 passers lose Sharpe vs always-trend in at least one period. Switch counts are 25 to 157 over 20 years.
- **Four sub-periods** (07-10, 11-15, 16-20, 21-now): 6 of the 8 beat always-trend in 3 of 4; none in all 4.

### The 8 passers (Sharpe / maxDD; baseline best = always-trend 0.347/-18.8% and 0.418/-24.2%)
| Rule (MR when ...) | 2007-2015 | 2016-now | switches | 2x-cost margin vs trend |
|---|---|---|---|---|
| days since 252d low in bottom third | 0.394 / -18.4% | 0.639 / -20.6% | 25 | -0.045 / +0.098 |
| days-252 low-half AND VIX high-half | 0.518 / -19.5% | 0.441 / -22.9% | 67 | +0.078 / -0.079 |
| days-252 low-half AND dd-rate high-half | 0.390 / -18.8% | 0.591 / -24.8% | 142 | -0.071 / +0.033 |
| gain-252 low-half AND IV-RV low-half | 0.419 / -19.2% | 0.455 / -25.0% | 157 | -0.036 / -0.059 |
| days-zz10 low-half AND VIX high-half | 0.374 / -19.5% | 0.438 / -25.0% | 60 | -0.073 / -0.097 |
| days-zz10 low-half AND VIX-TS high-half | 0.350 / -18.8% | 0.460 / -24.2% | 106 | -0.067 / -0.060 |
| days-zz10 high-half AND sector-corr low-half | 0.575 / -15.8% | 0.440 / -23.2% | 44 | +0.102 / -0.140 |
| gain-zz10 high-half AND ADX low-half | 0.394 / -13.8% | 0.554 / -21.0% | 136 | -0.141 / -0.004 |

## The new bottom signals specifically
- **Alone:** time since the 252d low is the only single signal of the 12 that passes (MR when it is in its bottom third, about <=150 days; MR is on 30% of days, 25 switches). Zig-zag time since bottom is a near miss (-0.006), and % gain since bottom alone fails under both definitions.
- **Combined:** 7 of the 8 passers include a bottom signal, but 136 of the 184 rules do (74%), so this is not enrichment (88% vs 74%). The signs of the combined rules disagree with each other (some need "recent bottom", some "old bottom"), which is what noise looks like.

## What failed
- Every other single signal alone: ADX, breadth, VIX term structure, drawdown rate, IV-RV, VIX level and sector correlation fail one or both periods; the efficiency ratio is the worst (worst-period margin -0.17).
- Always-MR loses to always-trend in both periods; the static 50/50 does not beat always-trend either (lower Sharpe, though shallower drawdown).
- Most rules that win one period lose the other; selection on one period is a coin flip on the other.

## Caveats
- The sleeves are research proxies (equal-capital ETF sleeves), not the live optimizer, and MR parameters were fixed at one reasonable setting. A differently built MR sleeve could change the answer.
- 20 years and ~2 regimes per period; the 8 passers are in a correlated family so the null comparison is rough.
- If you want to keep pursuing something: the days-since-252d-low rule is the only candidate worth a fresh out-of-sample check (new data from here, or another asset class); I would not trade it on this evidence.
