"""Run a named batch of variants; log EVERY variant tried to results_all.csv (variant counter honest)."""
import sys, os, warnings; warnings.filterwarnings("ignore")
import pandas as pd, engine as E

BASE = E.evaluate(E.sim())
def passes(ev):
    ok = True
    for p in BASE:
        ok &= ev[p][0] > BASE[p][0] and ev[p][2] >= BASE[p][2] - 0.01
    return bool(ok)

def run(batch, variants):
    rows = []
    for name, cfg in variants.items():
        ev = E.evaluate(E.sim(cfg))
        row = dict(batch=batch, variant=name)
        for p in BASE:
            row[f"sh_{p}"], row[f"cagr_{p}"], row[f"dd_{p}"] = ev[p]
        row["PASS"] = passes(ev); rows.append(row)
        print(f"{name:34s}" + " | ".join(f"{p}: Sh {ev[p][0]:5.2f} (b {BASE[p][0]:.2f}) DD {ev[p][2]*100:6.1f}% (b {BASE[p][2]*100:.1f}%)" for p in BASE), "PASS" if row["PASS"] else "")
    df = pd.DataFrame(rows)
    f = "results_all.csv"; old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    pd.concat([old[old.get("batch", pd.Series(dtype=str)) != batch] if len(old) else old, df]).to_csv(f, index=False)
    return df

BATCH1 = {
 "1a reentry@1.00": dict(reentry=1.00), "1b reentry@1.02": dict(reentry=1.02),
 "1c reentry on 50DMA rising": dict(reentry_m50=True),
 "2a haven IEF 50%": dict(haven="IEF"), "2b haven SHY 50%": dict(haven="SHY"), "2c haven best(IEF/SHY)": dict(haven="best"),
 "3a >=2of3 momentum": dict(mh=2), "3b 3of3 momentum": dict(mh=3),
 "4a reserve carve-out 20%": dict(reserve="carve"), "4b reserve=idle cash (<=20%)": dict(reserve="idle"),
}
if __name__ == "__main__":
    print("BASELINE", {p: tuple(round(x, 3) for x in v) for p, v in BASE.items()})
    run("batch1", BATCH1)
