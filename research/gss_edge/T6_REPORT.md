# T6 (12-1>0 eligibility) on the LIVE optimizer, 2026-11-26
Replay of the real `get_trends()` at 257 decision dates (window 1000 bars), next-day fill, 5 bps, cash=BIL, Sharpe on excess over BIL.
Hook: `portfolio.MOM_ELIGIBILITY` ("live" default = no behaviour change; "m12"; "m12_noer"). Code: `run_replay_t6.py`, `eval_t6.py`.

| Variant | 2007-15 Sh / DD | 2016-now Sh / DD | Bar | 2x cost |
|---|---|---|---|---|
| baseline (live) | 0.102 / -8.0% | 0.373 / -5.4% | | |
| m12 (TREND state, >200DMA & 12-1>0) | 0.048 / -8.2% | 0.442 / -5.4% | fail (Sharpe down 2007-15) | fail |
| m12_noer (any ER state) | 0.067 / -8.0% | 0.521 / -8.3% | fail (Sharpe down 2007-15, DD 2.9pp worse 2016-now) | fail |
| m12_noer equal-weight, same gross | 0.089 / -7.4% | 0.513 / -8.9% | fail | fail |

Verdict: the proxy-sleeve pass does NOT transfer. All variants help 2016-now and lose 2007-15 (3/4 sub-periods beat baseline, never all). The live optimizer's inverse-vol sizing and gate already do what equal-weight did in the proxy. Do not adopt.
