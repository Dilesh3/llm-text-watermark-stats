"""Tests for Chapters 3-6 on the GPT-2 analysis table.

Ch 3 the test, Ch 4 errors and power, Ch 5 extra checks of H0 (the main
Chapter 5 tests are in analysis/chapter5_keys.py), Ch 6 regression of z on sqrt(T).
Chapter 7 (p-chart) is analysis/pchart.py.

Usage:  python chapter_tests.py analysis_table_gpt2.csv raw results_gpt2.csv
        ('raw' = textbook count, every token; leave it out for the repeats-counted-once columns)
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

ARGS = [a for a in sys.argv[1:] if a != "raw"]
RAW = "raw" in sys.argv[1:]
TAB = ARGS[0]
RES = ARGS[1] if len(ARGS) > 1 else "results_gpt2.csv"   # the generation results file
t = pd.read_csv(TAB)
if RAW:
    t["T"], t["G"], t["z"] = t.T_raw, t.G_raw, t.z_raw
    t["green_share"] = t.G_raw / t.T_raw
    t["flag_a05"] = (t.z_raw > 1.645).astype(int); t["flag_z4"] = (t.z_raw > 4).astype(int)
    from scipy import stats as _s
    t["p_z"] = _s.norm.sf(t.z_raw); t["p_exact"] = _s.binom.sf(t.G_raw - 1, t.T_raw, 0.5)
    print("USING RAW (textbook) COUNTS")
r = pd.read_csv(RES)
A05, Z4 = 1.645, 4.0


def ci(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k}/{n} = {k/n:.1%} [95% CI {lo:.1%}, {hi:.1%}]"


def hdr(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


ai = t[t.source == "ai"]
hu = t[t.source == "human"]

# ---------------------------------------------------------------- Ch 3
hdr("CH3  one worked example + z vs exact")
ex = r[(r.text_id == "ca05")]
for _, row in ex.iterrows():
    T, G = int(row["T"]), int(row["G"])
    zc = (G - 0.5 - T / 2) / np.sqrt(T / 4)
    print(f"ca05 {row.source:5s} δ={row.delta}: T={T} G={G} share={G/T:.3f} z={row.z:.2f} "
          f"p_z={row.p_z:.3g} p_cc={stats.norm.sf(zc):.3g} p_exact={row.p_exact:.3g}")
# how often does z-test flag but exact test not, at alpha=0.05, for null texts, by length
for L in (25, 50, 100, 200):
    h = hu[hu.length == L]
    pz = h.p_z.astype(float); pe = h.p_exact.astype(float)
    print(f"human L={L}: flagged p_z<.05 {np.mean(pz<.05):.3f}  p_exact<.05 {np.mean(pe<.05):.3f}  "
          f"median p_z/p_exact {np.median(pz/pe):.2f}")
# exact theoretical size of z>1.645 rule at each T
for T in (25, 50, 100, 200):
    g = int(np.floor(T / 2 + 1.645 * np.sqrt(T / 4))) + 1
    print(f"T={T}: z>1.645 needs G>={g}; true Type I = {stats.binom.sf(g-1, T, .5):.4f}")

# ---------------------------------------------------------------- Ch 4
hdr("CH4  detection rate (empirical, with CI) vs theoretical power")
p_hat = {d: ai[(ai.delta == d) & (ai.length == 200)].green_share.mean() for d in (1, 2)}
print("mean green share at 200 tokens:", {d: round(v, 4) for d, v in p_hat.items()})
for d in (0, 1, 2):
    for L in (25, 50, 100, 200):
        s = ai[(ai.delta == d) & (ai.length == L)]
        n = len(s)
        line = f"δ={d} L={L:3d}  z>4: {ci(int(s.flag_z4.sum()), n)}   α=.05: {ci(int(s.flag_a05.sum()), n)}"
        if d > 0:
            Tm = s["T"].median()
            g4 = int(np.floor(Tm / 2 + 4 * np.sqrt(Tm / 4))) + 1
            line += f"   theory z>4 (T={Tm:.0f}): {stats.binom.sf(g4-1, Tm, p_hat[d]):.1%}"
        print(line)
for L in (25, 50, 100, 200):
    s = hu[hu.length == L]
    print(f"human L={L:3d}  z>4: {ci(int(s.flag_z4.sum()), len(s))}   α=.05: {ci(int(s.flag_a05.sum()), len(s))}")
# sample size formula
za, zb = 4.0, stats.norm.ppf(0.8)
for d, p in p_hat.items():
    n = ((za * 0.5 + zb * np.sqrt(p * (1 - p))) / (p - 0.5)) ** 2
    n2 = ((1.645 * 0.5 + zb * np.sqrt(p * (1 - p))) / (p - 0.5)) ** 2
    print(f"sample size for 80% power, δ={d} (p={p:.3f}): z>4 -> {np.ceil(n):.0f} tokens; α=.05 -> {np.ceil(n2):.0f} tokens")
# Bayes base rate
for prev in (0.01, 0.10, 0.30):
    for thr, fpr, pw in ((A05, 0.05, 1.0), (Z4, stats.norm.sf(4), 0.95)):
        ppv = pw * prev / (pw * prev + fpr * (1 - prev))
        print(f"prevalence {prev:.0%}, threshold z>{thr}: P(AI | flagged) = {ppv:.4f}")

# ---------------------------------------------------------------- Ch 5 (extra checks)
hdr("CH5  extra: dispersion, δ=0 control, dedup")
for name, g in (("human", r[r.source == "human"]), ("ai δ=0", r[(r.source == "ai") & (r.delta == 0)])):
    g200 = g[g["T"] == 200]
    n = len(g200); s2 = g200.G.var(ddof=1)
    chi = (n - 1) * s2 / 50.0
    p_disp = stats.chi2.sf(chi, n - 1)
    k = int((g.z > A05).sum()); kd = int((g.z_dedup > A05).sum())
    print(f"{name}: n(T=200)={n} mean G {g200.G.mean():.2f} var {s2:.1f} (50) dispersion χ²={chi:.0f} df={n-1} p={p_disp:.3g}")
    print(f"   FP raw {ci(k, len(g))} binom p={stats.binomtest(k, len(g), .05).pvalue:.3g};  "
          f"FP dedup {ci(kd, len(g))} binom p={stats.binomtest(kd, len(g), .05).pvalue:.3g}")
    print(f"   mean z {g.z.mean():.3f} (t-test vs 0 p={stats.ttest_1samp(g.z, 0).pvalue:.3g}), sd z {g.z.std():.3f}")
    # GOF for the T=200 subset
    G = g200.G.values
    pmf = stats.binom.pmf(np.arange(201), 200, .5)
    edges, cur = [], 0.0
    for kk in range(201):
        cur += pmf[kk] * n
        if cur >= 5:
            edges.append(kk); cur = 0
    edges[-1] = 200
    lo, obs, exp = 0, [], []
    for hi in edges:
        obs.append(((G >= lo) & (G <= hi)).sum()); exp.append(n * pmf[lo:hi + 1].sum()); lo = hi + 1
    exp = np.array(exp) * n / sum(exp)
    c2, pg = stats.chisquare(obs, exp)
    print(f"   GOF vs Binomial(200,.5): χ²={c2:.1f} df={len(obs)-1} p={pg:.3g}")
# decomposition: FP under normal approx with observed mean/sd
g = r[r.source == "human"]
m, sd = g.z.mean(), g.z.std()
print(f"human FP if z~N({m:.3f},{sd:.3f}): {stats.norm.sf((A05-m)/sd):.3f}; mean shift only {stats.norm.sf(A05-m):.3f}; "
      f"sd only {stats.norm.sf(A05/sd):.3f}")

# ---------------------------------------------------------------- Ch 7
hdr("CH6  regression of z on sqrt(T)")
ai0 = ai.copy(); ai0["sT"] = np.sqrt(ai0["T"])
hh = hu.copy(); hh["sT"] = np.sqrt(hh["T"])
rng = np.random.default_rng(7)
for name, s, p in (("δ=1", ai0[ai0.delta == 1], p_hat[1]), ("δ=2", ai0[ai0.delta == 2], p_hat[2]),
                   ("human", hh, 0.5)):
    m_all = smf.ols("z ~ sT", s).fit()
    m_cl = smf.ols("z ~ sT", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.text_id)[0]})
    # one random length per text -> independent observations
    one = s.groupby("text_id", group_keys=False).apply(lambda g: g.sample(1, random_state=int(rng.integers(1e9))))
    m_one = smf.ols("z ~ sT", one).fit()
    m0 = smf.ols("z ~ sT - 1", one).fit()
    resid_sd = s.assign(res=m_all.resid).groupby("length").res.std()
    print(f"{name}: theory slope 2(p-0.5)={2*(p-0.5):.3f}")
    print(f"   all rows (n={len(s)}): slope {m_all.params.sT:.3f} SE {m_all.bse.sT:.4f} (cluster SE {m_cl.bse.sT:.4f}) "
          f"intercept {m_all.params.Intercept:.3f} (p={m_cl.pvalues.Intercept:.3g}) R²={m_all.rsquared:.3f}")
    print(f"   one length/text (n={len(one)}): slope {m_one.params.sT:.3f} SE {m_one.bse.sT:.4f} "
          f"intercept {m_one.params.Intercept:.3f} p={m_one.pvalues.Intercept:.3g} R²={m_one.rsquared:.3f};  "
          f"no-intercept slope {m0.params.sT:.3f}")
    print(f"   residual SD by length: {resid_sd.round(2).to_dict()}  (theory sqrt(4p(1-p))={np.sqrt(4*p*(1-p)):.2f})")
# combined model with source dummies and interaction
comb = pd.concat([ai0.assign(grp="d" + ai0.delta.astype(int).astype(str)), hh.assign(grp="human")])
comb = comb[comb.grp.isin(["human", "d1", "d2"])]
mc = smf.ols("z ~ sT * C(grp, Treatment('human'))", comb).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(comb.text_id + comb.grp)[0]})
print(mc.summary().tables[1])
