"""MR-vs-trend regime study. Sleeves on the GSS ETF universe (engine.py arrays), causal regime signals,
switching rules vs always-trend / always-MR / static 50-50. Rules: next-day fill, 5bps on all turnover (incl. the
switch itself, since the composite book is run through the same turnover accounting), cash earns BIL."""
import sys, itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, engine as E
O = pd.read_pickle("/Users/riddhisiddhi/stocks/research/gss_edge/ohlcv.pkl")
idx, n, K, SI = E.idx, E.n, E.K, E.SI; CASH = E.cashr; COST = 5e-4; CAP = 0.15
rx = lambda s: s.reindex(idx).ffill().values

# ---------- sleeves: target-weight matrices decided at close t ----------
def gate_series():
    off = np.zeros(n, bool); s = False
    for t in range(n):
        if np.isfinite(E.m200[t]):
            if s and E.spy[t] > E.m200[t] * 1.04: s = False
            elif (not s) and E.spy[t] < E.m200[t] * 0.96: s = True
        off[t] = s
    return off
OFF = gate_series()

def trend_W():
    W = np.zeros((n, K)); cur = np.zeros(K); prev = False
    ok = E.a200 & (E.PX > E.M50) & (E.SL > 0); sc = np.where(ok, np.maximum(E.SL * E.R2, 0), 0.0); sc = np.where(np.isfinite(sc), sc, 0.0)
    for t in range(SI, n):
        if t in E.month_end or OFF[t] != prev:
            if OFF[t]: cur = np.zeros(K)
            else:
                s = sc[t]; tot = s.sum()
                if tot > 0: w = np.where(s > 0, np.minimum(s / tot, CAP), 0); cur = w / w.sum()
                else: cur = np.zeros(K)
        prev = OFF[t]; W[t] = cur
    return W

def mr_W(entry=10.0, exit_rsi=65.0, hold=5, slot=0.10):
    W = np.zeros((n, K)); pos = {}                      # asset -> days held
    for t in range(SI, n):
        for k in list(pos):
            pos[k] += 1
            if pos[k] >= hold or (np.isfinite(E.RS2[t, k]) and E.RS2[t, k] > exit_rsi): del pos[k]
        cand = [(E.RS2[t, k], k) for k in range(K) if k not in pos and E.a200[t, k] and np.isfinite(E.RS2[t, k]) and E.RS2[t, k] < entry]
        for _, k in sorted(cand):
            if (len(pos) + 1) * slot <= 1.0 + 1e-9: pos[k] = 0
        for k in pos: W[t, k] = slot
    return W

def run(W):
    """W[t] = target decided at close t -> filled close t+1 -> earns return of day t+2."""
    held = np.vstack([np.zeros((2, K)), W[:-2]]); prevh = np.vstack([np.zeros((1, K)), held[:-1]])
    r = (held * E.RET).sum(1) + (1 - held.sum(1)) * CASH
    # trade at close t-1 moves prevh->held ; charged on day t-1
    turn = np.abs(held - prevh).sum(1); cost = np.zeros(n); cost[:-1] = turn[1:] * COST
    return pd.Series(r - cost, index=idx)
WT, WM = trend_W(), mr_W()
RT, RM, R50 = run(WT), run(WM), run(0.5 * WT + 0.5 * WM)

# ---------- causal regime signals ----------
spy = O["Close"]["SPY"]; hi, lo = O["High"]["SPY"], O["Low"]["SPY"]
vix, v3 = O["Close"]["^VIX"], O["Close"]["^VIX3M"]
def adx(h, l, c, n_=14):
    up, dn = h.diff(), -l.diff(); pdm = np.where((up > dn) & (up > 0), up, 0.); mdm = np.where((dn > up) & (dn > 0), dn, 0.)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = 1 / n_; atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=c.index).ewm(alpha=a, adjust=False).mean() / atr; mdi = 100 * pd.Series(mdm, index=c.index).ewm(alpha=a, adjust=False).mean() / atr
    return (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=a, adjust=False).mean()
rv21 = spy.pct_change().rolling(21).std() * np.sqrt(252) * 100
dd = spy / spy.rolling(252, min_periods=60).max() - 1
secs = [c for c in ["XLK", "XLE", "XLI", "XLB", "XLV", "XLF", "XLP"]]
sr = O["Close"][secs].pct_change(); corr = pd.Series(np.nan, index=sr.index)
vals = sr.values
for i in range(63, len(sr)):
    c = np.corrcoef(vals[i - 62:i + 1].T); corr.iloc[i] = c[np.triu_indices(len(secs), 1)].mean()
breadth = pd.Series(np.nanmean(np.where(np.isfinite(E.M200), E.a200, np.nan), axis=1), index=idx)

def bottom_rolling(c, w=252):
    mn = c.rolling(w, min_periods=60).min(); days = c.rolling(w, min_periods=60).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)
    return days, c / mn - 1
def bottom_zigzag(c, thr=0.10):
    p = c.values; L = p[0]; Li = 0; H = p[0]; up = False; bot = p[0]; boti = 0; days = np.full(len(p), np.nan); gain = np.full(len(p), np.nan)
    for t in range(1, len(p)):
        if not up:
            if p[t] < L: L, Li = p[t], t
            if p[t] >= L * (1 + thr): up = True; bot, boti = L, Li; H = p[t]
        else:
            if p[t] > H: H = p[t]
            if p[t] <= H * (1 - thr): up = False; L, Li = p[t], t
        days[t] = t - boti; gain[t] = p[t] / bot - 1
    return pd.Series(days, index=c.index), pd.Series(gain, index=c.index)
tA, gA = bottom_rolling(spy); tC, gC = bottom_zigzag(spy)
RAW = {"vix": vix, "vix_ts": vix / v3, "iv_minus_rv": vix - rv21, "adx": adx(hi, lo, spy), "spy_er": (spy - spy.shift(20)).abs() / spy.diff().abs().rolling(20).sum(),
       "breadth200": breadth, "dd_rate21": dd - dd.shift(21), "sector_corr63": corr,
       "bottom_days_252": tA, "bottom_gain_252": gA, "bottom_days_zz10": tC, "bottom_gain_zz10": gC}
BOTTOM = ["bottom_days_252", "bottom_gain_252", "bottom_days_zz10", "bottom_gain_zz10"]; OTHER = [k for k in RAW if k not in BOTTOM]
def exp_pct(s):                       # expanding percentile rank, causal; 5d-median smoothed to cut chatter
    s = s.reindex(idx).ffill(); a = s.values; out = np.full(n, np.nan)
    for t in range(250, n):
        w = a[:t + 1]; w = w[np.isfinite(w)]
        if len(w) > 250 and np.isfinite(a[t]): out[t] = (w <= a[t]).mean()
    return pd.Series(out, index=idx).rolling(5, min_periods=1).median().values
PCT = {k: exp_pct(v) for k, v in RAW.items()}

# ---------- rules: boolean MR-state series ----------
def state(k, mode):
    p = PCT[k]; f = {"hi33": p > 2 / 3, "lo33": p < 1 / 3, "hi50": p > 0.5, "lo50": p < 0.5}[mode]; return np.where(np.isfinite(p), f, False)
RULES = {f"{k}|{m}": state(k, m) for k in RAW for m in ("hi33", "lo33", "hi50", "lo50")}
for b in BOTTOM:
    for o in OTHER:
        for mb, mo in itertools.product(("hi50", "lo50"), repeat=2): RULES[f"{b}|{mb} AND {o}|{mo}"] = state(b, mb) & state(o, mo)
for a_, b_ in [("bottom_days_252", "bottom_gain_252"), ("bottom_days_zz10", "bottom_gain_zz10")]:
    for mb, mo in itertools.product(("hi50", "lo50"), repeat=2): RULES[f"{a_}|{mb} AND {b_}|{mo}"] = state(a_, mb) & state(b_, mo)

PER = [("2007-2015", "2007-01-01", "2015-12-31"), ("2016-now", "2016-01-01", "2100-01-01")]
def stats(r, lo, hi):
    x = r.loc[lo:hi]; ex = x - pd.Series(CASH, index=idx).loc[lo:hi]; eq = (1 + x).cumprod()
    return ex.mean() / ex.std() * np.sqrt(252), float(eq.iloc[-1] ** (252 / len(x)) - 1), float((eq / eq.cummax() - 1).min())
def ev(r): return {p: stats(r, lo, hi) for p, lo, hi in PER}
BASES = {"always_trend": ev(RT), "always_MR": ev(RM), "static_50_50": ev(R50)}
BEST = {p: max(BASES.values(), key=lambda e: e[p][0])[p] for p, _, _ in PER}       # best-Sharpe baseline per period
def composite(mr): return run(np.where(mr[:, None], WM, WT))
def passes(e): return all(e[p][0] > max(b[p][0] for b in BASES.values()) and e[p][2] >= BEST[p][2] - 0.01 for p in e)
def n_switch(mr): return int((np.diff(mr.astype(int)[SI:]) != 0).sum())
