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

---
# Addendum 2026-11-26: 12-1 momentum (code: `regime_mom.py`, `regime_mom_followup.py`; data: `regime_mom_results.csv`, `regime_mom_passers.csv`, `trend_ranking_results.csv`)
12-1 momentum = close[t-21] / close[t-252] - 1 (12-month return skipping the latest month), causal. Same bar as above: Sharpe beats the baselines in BOTH periods (2007-2015, 2016-now) and maxDD no more than 1pp worse; then cross-period selection, 4 sub-periods, and **double trading costs (10 bps)**.

## Part 1: 12-1 momentum as an MR-vs-trend regime signal
Two versions: SPY's own 12-1, and the median 12-1 across the 42-ETF universe. Tested alone (4 cutoffs each, both directions) and ANDed with each of the 12 existing signals, including all four bottom signals (days since the 252d low is one of them), at median cutoffs and all direction pairs: **104 rules**.

| Alone (MR when ...) | 2007-2015 Sharpe / DD | 2016-now Sharpe / DD | Result |
|---|---|---|---|
| SPY 12-1 in top third | 0.240 / -18.8% | 0.056 / -32.3% | fail |
| **SPY 12-1 in bottom third** | **0.461 / -20.0%** | **0.491 / -24.3%** | Sharpe beats always-trend (0.347 / 0.418) in both, but DD is 1.2pp worse in 2007-15 (miss by 0.2pp); at 2x cost the margin is +0.024 / -0.033, so it does not hold |
| SPY 12-1 above / below median | 0.139 / 0.378 | 0.236 / 0.383 | fail |
| Universe-median 12-1 top third / bottom third | 0.225 / 0.355 | 0.296 / 0.559 | fail (bottom third: DD -27.8% in 2007-15) |
| Universe-median 12-1 above / below median | 0.241 / 0.274 | 0.080 / 0.531 | fail |

- **Direction:** the better single rules use MR when 12-1 momentum is *weak* (bottom third), i.e. dip-buying after weak 12-month trends, the same flavour as "recent 252d low" from the first study. Strong-momentum states favour trend and MR there loses badly in 2016-now.
- **Combined:** 1 of 104 passes (`universe-median 12-1 below median AND % gain since 10% zig-zag bottom above median`: 0.385 / -17.9% and 0.478 / -24.2%, 70 switches, 3/4 sub-periods). It fails at 2x cost (-0.089 / -0.014 margin vs always-trend). Random regimes pass at 1.0% (~1.0 false pass expected among 104), so 1 is noise.
- **With days since the 252d low (8 combos):** 2 of 8 beat always-trend's Sharpe in both periods (both "12-1 below median AND days-since-low below median"), none pass the full bar (DDs of -22 to -24%), and neither survives 2x cost in both periods. Adding 12-1 does not rescue the earlier days-since-low lead; it does not make it worse either.
- **Cross-period selection:** rank correlation of the Sharpe margin between periods is +0.21; top-10 chosen on 2007-15 give +0.106 there and -0.002 in 2016-now (3/10 positive); chosen on 2016-now, +0.126 and +0.019 in 2007-15 (8/10 positive). Mixed, no reliable carry-over.
- **Part 1 verdict: 12-1 momentum does not give a MR-vs-trend switching rule that holds in both periods, out of sample, and at double costs.** Always-trend stays the default.

## Part 2: 12-1 momentum as the trend-sleeve ranking (vs the current slope x R2)
Baseline T0 = the current signal: names above 200DMA and 50DMA with positive 20d slope, sized by slope x R2, 15% cap, SPY-200DMA gate, monthly rebalance (the trend sleeve from the study above). Variants swap the ranking, keeping gate, cap and costs. Bar is relative to T0.

| Variant | 2007-2015 Sh / CAGR / DD | 2016-now Sh / CAGR / DD | Bar | 2x cost | Sub-periods won |
|---|---|---|---|---|---|
| T0 current (slope x R2) | 0.347 / 4.4% / -18.8% | 0.418 / 8.0% / -24.2% | | | |
| T1 eligible = >200DMA and 12-1>0, sized by 12-1 | 0.548 / 7.4% / -17.4% | 0.547 / 10.5% / **-28.8%** | **fail** (DD 4.6pp worse 2016-now) | fail | 2/4 |
| T2 top-8 of T1 | 0.514 / 7.6% / -19.9% | 0.477 / 10.6% / -40.2% | fail | fail | 2/4 |
| T3 current filter, sized by 12-1 | 0.316 / 3.9% / -20.6% | 0.276 / 5.5% / -34.6% | fail | fail | 2/4 |
| T4 current score, only names with 12-1>0 | 0.303 / 3.7% / -18.4% | 0.396 / 7.6% / -26.5% | fail | fail | 2/4 |
| T5 rank-average (slope x R2, 12-1), current filter | 0.412 / 5.2% / -18.0% | 0.361 / 6.6% / -24.6% | fail | fail | 2/4 |
| T0b control: current score, top-8 only | 0.281 / 3.6% / -19.8% | 0.400 / 8.1% / -25.1% | fail | fail | 1/4 |

**None of the 6 pre-declared ranking variants passes.** T1 has the best Sharpe in both periods but concentrates into drawdowns (2016-now -28.8%).

**What the random-ranking null showed:** 120 random orderings of the *same eligible set as T1* (same sizing, cap, gate) have Sharpe 0.481 +/- 0.034 (2007-15) and 0.667 +/- 0.047 (2016-now). T1 sits at the 98th percentile of random in 2007-15 but only the **1st percentile in 2016-now**, and the current signal T0 is below **every** random draw in both periods. So the 12-1 *ranking* is not what helps. The gain comes from the eligibility set and from not concentrating by score.

**Exploratory follow-up (NOT pre-declared; added after seeing that null, so 3 extra variants on the same data and no claim of a clean pass):**

| Variant | 2007-2015 Sh / CAGR / DD | 2016-now Sh / CAGR / DD | Bar | 2x cost Sh | Sub-periods won |
|---|---|---|---|---|---|
| T6 equal-weight of (>200DMA and 12-1>0) | 0.505 / 6.3% / -18.7% | 0.686 / 10.9% / -16.6% | pass | 0.491 / 0.668 (pass) | 3/4 |
| T7 equal-weight of the current filter (>200DMA, >50DMA, slope>0) | 0.425 / 5.3% / -16.3% | 0.427 / 7.2% / -21.1% | pass | 0.385 / 0.385 (pass) | 2/4 |
| T8 equal-weight of (>200DMA, >50DMA, 12-1>0) | 0.387 / 4.7% / -20.8% | 0.496 / 8.2% / -21.8% | fail | fail | 2/4 |

- Equal weight beats slope x R2 sizing on this sleeve (T7 vs T0), and the 12-1>0 eligibility is a better filter than the 50DMA + slope filter (T6 vs T7, and T8 shows stacking both is worse). T6 passes every check you set: both periods, 2x cost, 3 of 4 sub-periods.
- **Treat this as a hypothesis, not a finding:** it was found after looking at the results, 10 ranking variants were examined in total, and nothing here is out of sample beyond the two periods. It is a proxy sleeve, and the **live book does not size by slope x R2**: it sizes with the covariance/inverse-vol optimizer, so this says nothing yet about the live optimizer. The way to test it properly is to replay `get_trends` with a 12-1 eligibility/selection variant (as in the live replay) and, ideally, on data after today.

## Bottom line for the addendum
1. 12-1 momentum as a regime switch: **no rule passes** the bar (1 of 104 against ~1 expected by chance, which fails at 2x cost). Best idea is "MR when SPY's 12-1 is weak", a near miss that does not survive costs.
2. 12-1 as the trend ranking: **the pre-declared variants all fail** (best Sharpe, T1, loses on drawdown). The random-ranking null says the ranking itself adds nothing. A 12-1>0 eligibility filter with equal weights passes all of the bar, but only as a post-hoc exploratory result that needs a live-optimizer replay and fresh data before anyone should lean on it.
