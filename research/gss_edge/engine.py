"""GSS edge-research engine. Baseline = the repo's validated ETF/trend-book simulation
(backtest/backtest_gss_whipsaw_buf_sweep.py: ER router, slope*R2 sizing, 15% cap, SPY-200DMA gate
0.96x exit / 1.04x re-entry, whole book to cash when gated, monthly rebalance) with the user's rules:
  * NEXT-DAY FILLS: decision at close d, trade at close d+1, new weights earn from d+1->d+2.
  * 5 bps per unit of turnover. Cash (1 - invested) earns BIL (SHY proxy before BIL exists, ~4 months).
  * Gate is evaluated DAILY (state), trades happen on month-end or on a gate flip.
Config knobs (all default = baseline): see CFG."""
import numpy as np, pandas as pd, sys
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

P = pd.read_csv("/Users/riddhisiddhi/stocks/research/gss_edge/prices.csv", index_col=0, parse_dates=True).ffill(limit=5)   # Yahoo has scattered null days; bridge short gaps only
TR_START, COST = "2007-01-01", 5e-4
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, CAP = 20, 20, 0.40, 0.35, 15.0, 0.15
UNIV = [t for t in TREND_ASSETS if t in P.columns and P[t].notna().sum() > 260]
idx = P.index; n = len(idx); K = len(UNIV)
PX = P[UNIV].values
RET = np.vstack([np.zeros((1, K)), PX[1:] / PX[:-1] - 1.0]); RET = np.where(np.isfinite(RET), RET, 0.0)
cashr = P["BIL"].pct_change(fill_method=None).fillna(P["SHY"].pct_change(fill_method=None)).fillna(0).values
spy = P["SPY"].values; spy_s = P["SPY"]
m200s = spy_s.rolling(200).mean(); m50s = spy_s.rolling(50).mean()
m200, m50 = m200s.values, m50s.values
IEFr, SHYr = P["IEF"].pct_change(fill_method=None).fillna(0).values, P["SHY"].pct_change(fill_method=None).fillna(0).values


def roll_sr(p, win=WIN):
    m = len(p); sl = np.full(m, np.nan); r2 = np.full(m, np.nan); lp = np.log(p)
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn; pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s * 1000; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2

def rsi(p, k):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values

SL = np.full((n, K), np.nan); R2 = SL.copy(); RS2 = SL.copy(); ER = SL.copy(); M50 = SL.copy(); M200 = SL.copy()
RET63 = SL.copy(); RET126 = SL.copy(); RET252 = SL.copy()
for k, c in enumerate(UNIV):
    s = P[c]; v = s.dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
    a, b = roll_sr(v.values); SL[:, k], R2[:, k] = ri(a), ri(b); RS2[:, k] = ri(rsi(v.values, 2))
    ER[:, k] = ri(((v - v.shift(ER_L)).abs() / v.diff().abs().rolling(ER_L).sum().replace(0, np.nan)).values)
    M50[:, k] = s.rolling(50).mean().values; M200[:, k] = s.rolling(200).mean().values
    RET63[:, k], RET126[:, k], RET252[:, k] = [s.pct_change(h, fill_method=None).values for h in (63, 126, 252)]
fin = np.isfinite
a200 = PX > M200
trend_ok = (ER >= ER_HI) & a200 & (PX > M50) & (SL > 0)
rev_ok = (ER <= ER_LO) & a200 & fin(RS2) & (RS2 < RSI_BUY)
SCORE = np.where(trend_ok, np.maximum(SL * R2, 0), np.where(rev_ok, np.clip((RSI_BUY - RS2) / RSI_BUY, 0, 1) * R2, 0))
SCORE = np.where(fin(SCORE) & fin(PX), SCORE, 0.0)
npos = (RET63 > 0).astype(int) + (RET126 > 0) + (RET252 > 0)
# capitulation inputs (live rules, macro/indicators/signals.py)
vix = P["^VIX"].ffill(); vz = ((vix - vix.rolling(50).mean()) / vix.rolling(50).std()).values
rsi14 = rsi(spy, 14); dd_exit = (spy_s / spy_s.rolling(252, min_periods=60).max().shift(1) - 1).values
r126_bil = (1 + pd.Series(cashr)).rolling(126).apply(np.prod, raw=True).values - 1
r126_ief = P["IEF"].pct_change(126, fill_method=None).values; r126_shy = P["SHY"].pct_change(126, fill_method=None).values
ds = pd.Series(idx); month_end = set(g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")))
SI = max(int(np.searchsorted(idx.values, np.datetime64(TR_START))), 220)

CFG = dict(exit=0.96, reentry=1.04, reentry_m50=False, haven=None, haven_frac=0.5, mh=0,
           reserve=None, volt=None, ddbrk=None, breadth=None, spyvol=None, deathcross=False, idle_haven=False)


def target_core(j, cfg):
    s = SCORE[j].copy()
    if cfg["mh"]: s = np.where(npos[j] >= cfg["mh"], s, 0.0)
    tot = s.sum()
    if tot <= 0: return np.zeros(K)
    w = np.minimum(s / tot, CAP); w = np.where(s > 0, w, 0); return w / w.sum()


SPYVOL = pd.Series(spy).pct_change(fill_method=None).rolling(20).std().values * np.sqrt(252)
SPYI = UNIV.index("SPY")

def sim(cfg=None):
    cfg = {**CFG, **(cfg or {})}
    w = np.zeros(K); hv = np.zeros(3)         # held: core weights, [IEF, SHY, SPY-reserve]
    pend = None; off = False; deployed = False; hwz = 0.0; dd_scaled = False
    out = np.zeros(n); lastcore = np.zeros(K); hpick = None; eq = np.ones(n); rf = cfg.get("reserve_frac", 0.2)
    for t in range(SI, n):
        r = w @ RET[t] + hv[0] * IEFr[t] + hv[1] * SHYr[t] + hv[2] * RET[t, SPYI]       # 1. earn on held book
        r += (1 - w.sum() - hv.sum()) * cashr[t]
        if pend is not None:                                                              # 2. next-day fill + costs
            nw, nh = pend; r -= (np.abs(nw - w).sum() + np.abs(nh - hv).sum()) * COST
            w, hv, pend = nw, nh, None
        out[t] = r; eq[t] = eq[t - 1] * (1 + r)
        if not fin(m200[t]): continue                                                     # 3. decide at close t
        prev = off
        if off:
            up = spy[t] > m200[t] * cfg["reentry"]
            if cfg["reentry_m50"]: up = up or (fin(m50[t - 10]) and m50[t] > m50[t - 10])
            if up: off = False
        elif spy[t] < m200[t] * cfg["exit"] or (cfg.get("deathcross") and m50[t] < m200[t]): off = True
        if cfg["reserve"] in ("carve", "idle"):
            if deployed and fin(dd_exit[t]) and dd_exit[t] > 0: deployed = False; hwz = 0.0
            elif not deployed and fin(vz[t]) and vz[t] > 2 and rsi14[t] < 30: deployed = True; hwz = vz[t]
            elif deployed and fin(vz[t]): hwz = max(hwz, vz[t])
        if t in month_end or off != prev:
            core = np.zeros(K) if off else target_core(t, cfg)
            if cfg["volt"] and core.sum() > 0:
                sv = (RET[max(t - 20, 0):t + 1] @ core).std() * np.sqrt(252)
                core = core * min(1.0, cfg["volt"] / sv) if sv > 0 else core
            if cfg["breadth"] is not None and core.sum() > 0:
                if np.nanmean(a200[t][fin(M200[t])]) < cfg["breadth"]: core = core * 0.5
            lastcore = core; hpick = None
            if cfg["haven"] and (off or cfg.get("idle_haven")):
                ok = {k: v for k, v in {"IEF": r126_ief[t], "SHY": r126_shy[t]}.items() if fin(v) and fin(r126_bil[t]) and v > r126_bil[t]}
                hpick = (max(ok, key=ok.get) if cfg["haven"] == "best" else (cfg["haven"] if cfg["haven"] in ok else None)) if ok else None
        core = lastcore.copy()
        if cfg["ddbrk"] and core.sum() > 0:                                               # book drawdown breaker
            dd = eq[t] / eq[max(SI, t - 126):t + 1].max() - 1
            dd_scaled = True if dd < -cfg["ddbrk"] else (False if dd > -cfg["ddbrk"] / 2 else dd_scaled)
            if dd_scaled: core = core * 0.5
        if cfg["spyvol"] and fin(SPYVOL[t]) and SPYVOL[t] > cfg["spyvol"]: core = core * 0.5
        if cfg["reserve"] in ("carve", "carve0"): core = core * (1 - rf)
        nh = np.zeros(3)
        if hpick:
            frac = cfg["haven_frac"] if off else max(1 - core.sum(), 0) * cfg["haven_frac"]
            nh[0 if hpick == "IEF" else 1] = frac
        if cfg["reserve"] in ("carve", "idle") and deployed:
            frac = 0.01 + 0.99 * min(max((hwz - 2) / 2, 0), 1)
            base = rf if cfg["reserve"] == "carve" else min(max(1 - core.sum() - nh.sum(), 0), rf)
            nh[2] = round(base * frac, 2)
        pend = (core, nh)
    return pd.Series(out, index=idx)


def stats(r, lo, hi):
    x = r.loc[lo:hi]; ex = x - pd.Series(cashr, index=idx).loc[lo:hi]
    sh = ex.mean() / ex.std() * np.sqrt(252); eq = (1 + x).cumprod()
    return sh, float(eq.iloc[-1] ** (252 / len(x)) - 1), float((eq / eq.cummax() - 1).min())

PERIODS = [("2007-2015", "2007-01-01", "2015-12-31"), ("2016-now", "2016-01-01", "2100-01-01")]
def evaluate(r): return {p: stats(r, lo, hi) for p, lo, hi in PERIODS}
