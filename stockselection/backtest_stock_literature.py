"""
More edge from the literature: does any additional price-based single-stock
anomaly IMPROVE the shipped winner (12-1 momentum + SPY-200DMA gate, Sharpe 1.22)?
All variants gated (SPY-200DMA, T-bill cash), top quintile, monthly, net@10bps.

  base_mom       12-1 total-return momentum                    (the shipped rule)
  resid_mom      12-1 RESIDUAL momentum (CAPM resid vs SPY)     (Blitz-Huij-Martens
                 2011 - lower crash risk)
  mom_x_lowidv   top-40% mom AND bottom-40% idiosyncratic vol   (Ang 2006 - high
                 idio-vol underperforms; "quality momentum")
  mom_x_lowMAX   top-40% mom AND bottom-40% MAX daily ret (21d)  (Bali 2011 - avoid
                 lottery/short-vol-crash names)
  lowidv_only    bottom-quintile idiosyncratic vol alone        (the anomaly solo)

Rolling 252d CAPM beta to SPY -> residual return; residual momentum = cum resid
(t-252..t-21); idio vol = std(resid, 252d); MAX = max daily ret (21d). A variant
only counts if it beats base_mom in Sharpe AND drawdown in MULTIPLE sub-periods.
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
COST, QUINTILE = 10.0, 0.20


def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 3
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); cg = eq[-1] ** (12 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    close = pd.read_parquet(CACHE)
    spy = close["SPY"]; spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]; idx = cs.index
    ret = cs.pct_change()
    spy_ret = spy.pct_change()

    # rolling 252d CAPM beta to SPY, residual returns, residual momentum, idio vol
    print("computing residuals / idio-vol ...")
    var_spy = spy_ret.rolling(252).var()
    beta = ret.rolling(252).cov(spy_ret).div(var_spy, axis=0)
    resid = ret.sub(beta.mul(spy_ret, axis=0))
    resid_mom = resid.rolling(231).sum().shift(21)          # cum resid t-252..t-21
    idiov = resid.rolling(252).std()
    maxret = ret.rolling(21).max()

    mom = cs.shift(21) / cs.shift(252) - 1.0                 # total-return 12-1

    irx = yf.download("^IRX", start="2004-01-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200)

    me = cs.resample("ME").last().index
    reb = [idx[idx.searchsorted(d, side="right") - 1] for d in me
           if idx.searchsorted(d, side="right") - 1 > 300]
    reb = sorted(set(reb))

    variants = ["base_mom", "resid_mom", "mom_x_lowidv", "mom_x_lowMAX", "lowidv_only",
                "EW-universe"]
    ser = {v: [] for v in variants}; dts = []; prev = {v: set() for v in variants}

    def topq(sig, valid):
        s = sig[valid].dropna()
        s = s[np.isfinite(s)]
        if len(s) < 50:
            return None
        nq = max(int(len(s) * QUINTILE), 5)
        return list(s.sort_values().index[-nq:])

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        valid = fwd.notna() & np.isfinite(fwd)
        off = bool(gate_off.loc[d0]); c = float(cash_mo.loc[d1])

        def book(sel, name):
            if sel is None or len(sel) == 0:
                r, held = 0.0, set()
            elif off:
                r, held = c, set()                        # gate to cash
            else:
                sel = [x for x in sel if valid.get(x, False)]
                r, held = (fwd[sel].mean(), set(sel)) if sel else (c, set())
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1) if held or prev[name] else 0.0
            ser[name].append(r - to * COST / 1e4); prev[name] = held

        m = mom.loc[d0]
        book(topq(m, valid), "base_mom")
        book(topq(resid_mom.loc[d0], valid), "resid_mom")
        # double sorts: top-40% mom AND bottom-40% of the risk metric
        mv = m[valid & np.isfinite(m)]
        if len(mv) >= 50:
            top_m = set(mv.sort_values().index[-int(len(mv) * 0.4):])
            iv = idiov.loc[d0][list(top_m)].dropna()
            mx = maxret.loc[d0][list(top_m)].dropna()
            book(list(set(iv.sort_values().index[:max(int(len(iv) * 0.5), 5)])), "mom_x_lowidv")
            book(list(set(mx.sort_values().index[:max(int(len(mx) * 0.5), 5)])), "mom_x_lowMAX")
        else:
            book(None, "mom_x_lowidv"); book(None, "mom_x_lowMAX")
        # low idio-vol alone: bottom quintile idio vol (negate so topq picks lowest)
        book(topq(-idiov.loc[d0], valid), "lowidv_only")
        # EW benchmark (ungated)
        held = set(fwd[valid].index)
        to = len(held ^ prev["EW-universe"]) / max(len(held) + len(prev["EW-universe"]), 1)
        ser["EW-universe"].append(fwd[valid].mean() - to * COST / 1e4); prev["EW-universe"] = held
        dts.append(d1)

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    print(f"\nLiterature overlays (all gated SPY-200DMA), S&P 500, monthly, net@10bps\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==         {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for v in variants:
            sh, cg, dd = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            star = " *" if v == "base_mom" else "  "
            print(f"  {v:13s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
