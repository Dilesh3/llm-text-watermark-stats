"""
sanity_checks.py - checks a results file before it is handed to the analysts.

Works on both results files:
  python sanity_checks.py prototype/results_prototype.csv
  python sanity_checks.py data/results_gpt2.csv

What it checks (and what we expect):
  1. every text has the intended length T
  2. unwatermarked texts (human, AI with delta = 0) are about 50% green,
     z has mean ~0 and SD ~1, and about 5% are flagged at alpha = 0.05
  3. watermarked texts are clearly greener, more so for delta = 2 than delta = 1
  4. the same with repeated (previous token, token) pairs counted once
Writes a short report next to the input file.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def summarise(d):
    d = d.copy()
    d["setting"] = np.where(d.source == "human", "human",
                            "AI, delta = " + d.delta.fillna(-1).astype(int).astype(str))
    rows = []
    for name, g in d.groupby("setting", sort=False):
        fp = (g.z > 1.645).mean()
        n = len(g)
        # one-proportion test: is the flag rate at alpha = 0.05 really 5%?
        # (only meaningful for unwatermarked text; for watermarked text the flag rate is the power)
        watermarked = name.startswith("AI") and not name.endswith("= 0")
        p_cal = np.nan if watermarked else stats.binomtest(int((g.z > 1.645).sum()), n, 0.05).pvalue
        rows.append({
            "setting": name, "texts": n, "mean T": round(g["T"].mean(), 1),
            "mean green %": round(100 * g.green_frac.mean(), 1),
            "mean z": round(g.z.mean(), 2), "SD of z": round(g.z.std(ddof=1), 2),
            "flagged at z > 1.645": f"{100 * fp:.1f}%",
            "p (flag rate = 5%)": "" if np.isnan(p_cal) else f"{p_cal:.2g}",
            "flagged at z > 4": f"{100 * (g.z > 4).mean():.1f}%",
            "mean z (dedup)": round(g.z_dedup.mean(), 2),
            "flagged at z > 1.645 (dedup)": f"{100 * (g.z_dedup > 1.645).mean():.1f}%",
        })
    return pd.DataFrame(rows)


def _table(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:                      # 'tabulate' not installed
        return df.to_string(index=False)


def main(path):
    d = pd.read_csv(path)
    s = summarise(d)
    lines = [f"# Sanity checks for {os.path.basename(path)}", "",
             f"{len(d)} texts. Lengths: min T = {d['T'].min()}, max T = {d['T'].max()}.", "",
             _table(s), "",
             "Expected: human and delta = 0 rows near 50% green, mean z near 0, SD near 1, "
             "about 5% flagged at z > 1.645 and none at z > 4; delta = 1 and 2 rows far greener. "
             "'dedup' columns count each repeated (previous token, token) pair once."]
    report = "\n".join(lines)
    out = os.path.splitext(path)[0] + "_sanity_report.md"
    with open(out, "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prototype/results_prototype.csv")
