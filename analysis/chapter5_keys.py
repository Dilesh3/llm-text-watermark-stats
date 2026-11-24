"""
chapter5_keys.py — Chapter 5: does real human text behave as H₀ says?

H₀ says a human text's green count is Binomial(T, γ), so at α = 0.05 about
5% of human texts should be flagged. This script tests that claim and
explains why it can fail for one fixed key.

It reports
  1. Team key: chi-square goodness of fit of human green counts against
     Binomial(T, γ), and a one-proportion test of the false-positive rate.
  2. Key effect: the false-positive rate under many different keys.
     The average over keys is close to α, but single keys can be far off,
     because common token pairs ("of the") appear in every text and have one
     fixed colour per key, so all human texts lean green or red together.
  3. Cause: how much of the text the most common pairs cover, and their
     green share under the team key.
  4. Fix (calibration): estimate the mean and SD of z on half of the human
     texts, standardise z on the other half; the false-positive rate is then
     close to α under every key.

Usage
-----
  python chapter5_keys.py ../wm/prototype/results_prototype.csv --keys 20
  python chapter5_keys.py results_gpt2.csv --keys 20 --out chapter5_keys.csv
"""

import argparse
import csv
from collections import Counter

import numpy as np
from scipy import stats

from results import TEAM_KEY, load
from watermark import Vocabulary, WatermarkConfig, WatermarkEngine

A05 = 1.645


def z_scores(eng, recs, dedup):
    return np.array([eng.detect_ids(r["ids"], r["context_id"], ignore_repeats=dedup).z
                     for r in recs])


def gof(eng, recs, gamma):
    """Chi-square GOF of raw green counts against Binomial(T, γ) (equal T only)."""
    T = len(recs[0]["ids"])
    G = np.array([eng.detect_ids(r["ids"], r["context_id"]).G for r in recs])
    n = len(G)
    # bins with expected count >= 5, merged from the tails inwards
    pmf = stats.binom.pmf(np.arange(T + 1), T, gamma)
    edges, cur = [], 0.0
    for k in range(T + 1):
        cur += pmf[k] * n
        if cur >= 5:
            edges.append(k)
            cur = 0.0
    edges[-1] = T
    lo, obs, exp = 0, [], []
    for hi in edges:
        obs.append(int(((G >= lo) & (G <= hi)).sum()))
        exp.append(n * pmf[lo:hi + 1].sum())
        lo = hi + 1
    exp = np.array(exp) * n / sum(exp)
    chi2, p = stats.chisquare(obs, exp, ddof=0)
    return T, G, chi2, len(obs) - 1, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="results CSV from the generation step")
    ap.add_argument("--keys", type=int, default=20, help="number of keys (team key + others)")
    ap.add_argument("--group", default="human", choices=["human", "ai0"],
                    help="human texts, or AI texts without watermark (δ=0)")
    ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--key", type=int, default=TEAM_KEY)
    ap.add_argument("--seed", type=int, default=7, help="seed for the calibration split")
    ap.add_argument("--out", default="chapter5_keys.csv")
    a = ap.parse_args()

    meta, recs, eng = load(a.data, a.vocab_size, a.key)
    gamma, V = meta["gamma"], meta["vocab_size"]
    if a.group == "human":
        nulls = [r for r in recs if r["source"] == "human"]
    else:
        nulls = [r for r in recs if r["source"] == "ai" and r["delta"] == 0]
    n = len(nulls)
    print(f"{n} {a.group} texts, {meta['model']}, γ={gamma}\n")

    # ---- 1. Team key: GOF and false-positive test ------------------------- #
    T, G, chi2, df, p_gof = gof(eng, nulls, gamma)
    z_team = z_scores(eng, nulls, dedup=False)
    k = int((z_team > A05).sum())
    p_fp = stats.binomtest(k, n, 0.05, alternative="two-sided").pvalue
    print("1. TEAM KEY")
    print(f"   Green counts vs Binomial({T}, {gamma}): mean G {G.mean():.1f} (expected {T * gamma:.0f}), "
          f"variance {G.var(ddof=1):.1f} (expected {T * gamma * (1 - gamma):.0f})")
    print(f"   Chi-square GOF: χ² = {chi2:.1f}, df = {df}, p = {p_gof:.3g}")
    print(f"   False positives at α=0.05: {k}/{n} = {k / n:.1%} "
          f"(expected 5%; one-proportion test p = {p_fp:.3g})\n")

    # ---- 2 & 4. Many keys, with and without calibration ------------------- #
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(n)
    cal, test = idx[: n // 2], idx[n // 2:]
    keys = [a.key] + [k_ for k_ in range(1, a.keys + 1) if k_ != a.key][: a.keys - 1]
    rows = []
    for key in keys:
        e = WatermarkEngine(WatermarkConfig(key=key, gamma=gamma), Vocabulary.of_size(V))
        zr, zd = z_scores(e, nulls, False), z_scores(e, nulls, True)
        mu, sd = zr[cal].mean(), zr[cal].std(ddof=1)
        zc = (zr[test] - mu) / sd
        rows.append({"key": "team" if key == a.key else key,
                     "mean_z": zr.mean(), "sd_z": zr.std(ddof=1),
                     "fp_raw": (zr > A05).mean(), "fp_dedup": (zd > A05).mean(),
                     "fp_test_half_raw": (zr[test] > A05).mean(),
                     "fp_test_half_calibrated": (zc > A05).mean(),
                     "fp_z4": (zr > 4).mean()})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}
                    for r in rows)

    def span(col):
        v = np.array([r[col] for r in rows])
        return f"mean {v.mean():5.1%}   min {v.min():5.1%}   max {v.max():5.1%}"

    print(f"2. KEY EFFECT: false-positive rate at α=0.05 over {len(keys)} keys")
    print(f"   raw (textbook test)        {span('fp_raw')}")
    print(f"   repeats counted once       {span('fp_dedup')}")
    mz = np.array([r["mean_z"] for r in rows])
    print(f"   mean z of human texts per key ranges {mz.min():+.2f} to {mz.max():+.2f} "
          f"(H₀ says 0)")
    print(f"   at z>4: {span('fp_z4')}\n")

    # ---- 3. Cause: common pairs ------------------------------------------- #
    pairs = Counter()
    for r in nulls:
        seq = [r["context_id"]] + r["ids"]
        pairs.update(zip(seq[:-1], seq[1:]))
    total = sum(pairs.values())
    top = pairs.most_common(50)
    covered = sum(c for _, c in top)
    green_tok = sum(c for (p, t), c in top if eng.greens.is_green(p, t))
    print("3. CAUSE")
    print(f"   The 50 most common token pairs make up {covered / total:.1%} of all scored tokens.")
    print(f"   Under the team key, {green_tok / covered:.1%} of those tokens are green (fair share 50%).")
    print("   Each pair has ONE colour per key, shared by every text, so texts don't vary")
    print("   independently: the whole corpus is pushed greener or redder together.\n")

    # ---- 4. Fix ----------------------------------------------------------- #
    print("4. FIX: calibrate on half the human texts, test on the other half")
    print(f"   uncalibrated   {span('fp_test_half_raw')}")
    print(f"   calibrated     {span('fp_test_half_calibrated')}")
    print(f"\nper-key results written to {a.out}")


if __name__ == "__main__":
    main()
