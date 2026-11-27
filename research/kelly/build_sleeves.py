"""Monthly excess-over-cash return series for the 5 sleeves (research, not live code).
Sleeves: stock_sieve (12-1 top10 <=3/sector, SPY200 gate, vol-managed cap1.0), etf_12_1 (top-8 12-1 on
trend-ETF universe, SPY200 gate), vix_capitulation (SPY, ladder $1k..$100k of 100k sleeve), qqq_rsi2 (prod rule
RSI2<=5, >200SMA, exit>=70), trend_etf (equal-weight 200SMA trend on the ETF universe). Cash cushion = 0 by construction.
All signals lagged one day / one rebalance; 10 bps on rebalance turnover for momentum sleeves, 5 bps one-way for daily sleeves."""
import sys, numpy as np, pandas as pd, yfinance as yf
from collections import Counter
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.helpers import compute_RSI
R = "/Users/riddhisiddhi/stocks/"
close = pd.read_parquet(R + ".cache/selbuffer/sp500_close.parquet")
sdf = pd.read_csv(R + "stockselection/sp500_sectors.csv"); sec = dict(zip(sdf["Symbol"], sdf[sdf.columns[-1]]))
ETF = pd.read_csv(R + "trendselection/trend_etf_ranking.csv")["Ticker"].tolist()
raw = yf.download(ETF + ["SPY", "QQQ", "^VIX", "^IRX"], start="2004-01-01", end="2026-11-26", progress=False, auto_adjust=True)["Close"].ffill()
irx = raw["^IRX"]; spy = raw["SPY"]; vix = raw["^VIX"]
cash_d = (irx / 100 / 252).fillna(0)

def compound(d):   # daily -> monthly
    return (1 + d).groupby(pd.Grouper(freq="ME")).prod() - 1
cash_m = compound(cash_d)

# ---------- stock sieve (monthly, as backtest_volmanaged_sleeve.py) ----------
def sieve():
    spy_c = close["SPY"].dropna(); stocks = [c for c in close.columns if c != "SPY"]; cs = close[stocks]; idx = cs.index
    rets = cs.pct_change(); mom = cs.shift(21) / cs.shift(252) - 1
    cm = (irx.reindex(idx).ffill() / 100 / 12).fillna(0)
    off = (spy_c < spy_c.rolling(200).mean()).reindex(idx).ffill().fillna(False)
    me = cs.resample("ME").last().index
    reb = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in me if idx.searchsorted(d, side="right") - 1 > 300})
    base, dts, cs_, prev = [], [], [], set()
    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]; m = mom.loc[d0]; fwd = cs.loc[d1] / cs.loc[d0] - 1
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd); m, fwd = m[ok], fwd[ok]
        c = float(cm.loc[d1])
        if len(m) < 60: continue
        if bool(off.loc[d0]): r, held = c, set()
        else:
            win = rets.loc[:d0].tail(252)
            order = [t for t in m.sort_values(ascending=False).index if win[t].abs().max() < .5 and 0 < m[t] < 4]
            pk, cnt = [], Counter()
            for t in order:
                s = sec.get(t, t)
                if cnt[s] < 3: pk.append(t); cnt[s] += 1
                if len(pk) == 10: break
            sel = [x for x in pk if x in fwd.index]; r, held = (fwd[sel].mean(), set(sel)) if sel else (c, set())
        to = len(held ^ prev) / max(len(held) + len(prev), 1); base.append(r - to * 10 / 1e4); prev = held; cs_.append(c); dts.append(d1)
    B = pd.Series(base, index=dts); C = pd.Series(cs_, index=dts)
    w = (0.15 / (B.shift(1).rolling(6).std() * np.sqrt(12))).clip(0, 1).fillna(1.0)
    vm = w * B + (1 - w) * C
    return vm, B, C

# ---------- ETF sleeves ----------
px = raw[[t for t in ETF if t in raw.columns]]; px = px.loc[:, px.notna().sum() > 800]
er = px.pct_change(); spy200 = spy.rolling(200).mean()
def etf_12_1(K=8):
    mom = px.shift(21) / px.shift(252) - 1; gate = (spy > spy200).astype(float)
    me = px.resample("ME").last().index; idx = px.index
    reb = [idx[idx.searchsorted(d, side="right") - 1] for d in me]
    w = pd.DataFrame(0.0, index=idx, columns=px.columns); cur = pd.Series(0.0, index=px.columns); ri = {d: 1 for d in reb}
    for t in idx:
        if t in ri:
            cur = pd.Series(0.0, index=px.columns)
            if gate.loc[t] > 0:
                m = mom.loc[t].dropna(); m = m[m > 0].sort_values(ascending=False).head(K)
                if len(m): cur[m.index] = 1.0 / K
        w.loc[t] = cur.values
    wl = w.shift(1); turn = wl.diff().abs().sum(axis=1).fillna(0)
    gross = (wl * er).sum(axis=1); inv = wl.sum(axis=1)
    return gross + (1 - inv) * cash_d - turn * 5 / 1e4
def trend_etf():
    on = (px > px.rolling(200, min_periods=200).mean()).astype(float).where(px.notna(), 0.0)
    n = px.notna().sum(axis=1).clip(lower=1); wl = on.div(n, axis=0).shift(1)
    turn = wl.diff().abs().sum(axis=1).fillna(0); inv = wl.sum(axis=1)
    return (wl * er).sum(axis=1) + (1 - inv) * cash_d - turn * 5 / 1e4

# ---------- capitulation (SPY) ----------
def capit():
    z = (vix - vix.rolling(50).mean()) / vix.rolling(50).std(); r14 = compute_RSI(spy, 14)
    dde = spy / spy.rolling(252, min_periods=60).max().shift(1) - 1
    df = pd.concat({"z": z, "r": r14, "d": dde}, axis=1).dropna()
    dep, hw, tgt = False, 0.0, []
    for zz, rr, dd in df.itertuples(index=False):
        if dep and dd > 0: dep, hw = False, 0.0
        else:
            if not dep and zz > 2 and rr < 30: dep, hw = True, zz
            elif dep: hw = max(hw, zz)
        tgt.append((1000 + min(max((hw - 2) / 2, 0), 1) * 99000) / 100000 if dep else 0.0)
    f = pd.Series(tgt, index=df.index).shift(1).reindex(spy.index).fillna(0)
    turn = f.diff().abs().fillna(0)
    return f * spy.pct_change() + (1 - f) * cash_d - turn * 5 / 1e4
def qqq_rsi2():
    c = raw["QQQ"]; rsi = compute_RSI(c, 2); sma = c.rolling(200, min_periods=200).mean(); pos, cur = [], 0
    for p, s, r in zip(c, sma, rsi):
        if np.isnan(s) or p < s: cur = 0
        elif r <= 5: cur = 1
        elif r >= 70: cur = 0
        pos.append(cur)
    f = pd.Series(pos, index=c.index).shift(1).fillna(0); turn = f.diff().abs().fillna(0)
    return f * c.pct_change() + (1 - f) * cash_d - turn * 5 / 1e4

vm, base, C = sieve()
out = pd.DataFrame({
    "stock_sieve": vm - C, "stock_sieve_novolmgd": base - C,
    "etf_12_1": compound(etf_12_1()) - cash_m, "vix_capitulation": compound(capit()) - cash_m,
    "qqq_rsi2": compound(qqq_rsi2()) - cash_m, "trend_etf": compound(trend_etf()) - cash_m,
})
out["cash_m"] = cash_m
out.index = out.index.to_period("M")
out = out.dropna(subset=["stock_sieve", "etf_12_1", "vix_capitulation", "qqq_rsi2", "trend_etf"])
out.to_csv(R + "research/kelly/sleeve_monthly_excess.csv"); print(out.index[0], out.index[-1], len(out))
print(out.drop(columns="cash_m").describe().T[["mean", "std"]].assign(ann_ex=lambda d: d["mean"] * 12, sharpe=lambda d: d["mean"] / d["std"] * 12 ** .5).round(3))
