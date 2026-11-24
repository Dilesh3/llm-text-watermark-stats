"""All figures for the decks, from the GPT-2 data. Plain matplotlib style.

Usage:  python make_figures.py results_gpt2.csv analysis_table_gpt2.csv chapter5_keys_gpt2.csv figures/
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from watermark import Vocabulary, WatermarkConfig, WatermarkEngine  # noqa  (same folder)

RES_CSV, TAB_CSV, KEYS_CSV, OUT = (sys.argv[1:5] + ["results_gpt2.csv", "analysis_table_gpt2.csv",
                                                    "chapter5_keys_gpt2.csv", "figures"][len(sys.argv) - 1:])[:4]
os.makedirs(OUT, exist_ok=True)
res = pd.read_csv(RES_CSV)
tab = pd.read_csv(TAB_CSV)
# textbook counts everywhere
tab["zr"] = tab.z_raw
tab["f4"] = (tab.z_raw > 4).astype(int)
tab["f05"] = (tab.z_raw > 1.645).astype(int)
tab["gs"] = tab.G_raw / tab.T_raw

HUM, D0, D1, D2 = "#1f77b4", "#7f7f7f", "#ff7f0e", "#d62728"
GREEN, RED = "#2ca02c", "#d62728"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.8,
    "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return k / n, k / n - lo, hi - k / n


# ------------------------------------------------------------------ toy example
def fig_toy():
    words = ["mat", "sofa", "floor", "bed"]
    s = np.array([2.0, 1.3, 1.0, 0.6])
    green = np.array([0, 1, 0, 1], bool)
    p0 = np.exp(s) / np.exp(s).sum()
    s1 = s + 2 * green
    p1 = np.exp(s1) / np.exp(s1).sum()
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, 100 * p0, 0.4, color="#bdbdbd", label="before boost")
    ax.bar(x + 0.2, 100 * p1, 0.4, color=[GREEN if g else RED for g in green], label="after adding δ = 2 to green words")
    for i in range(4):
        ax.text(i - 0.2, 100 * p0[i] + 1, f"{100*p0[i]:.1f}%", ha="center", fontsize=10)
        ax.text(i + 0.2, 100 * p1[i] + 1, f"{100*p1[i]:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n({'green' if g else 'red'})" for w, g in zip(words, green)])
    ax.set_ylabel("probability (%)")
    ax.set_ylim(0, 62)
    ax.set_title("'The cat sat on the ...'   P(green word): 35% -> 80%")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#bdbdbd", label="before boost"),
                       Patch(color=GREEN, label="after boost, green word"),
                       Patch(color=RED, label="after boost, red word")], loc="upper right")
    save(fig, "toy_boost")


# ------------------------------------------------------------------ null distribution
def fig_null():
    T = 200
    k = np.arange(60, 171)
    pmf = stats.binom.pmf(k, T, 0.5)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(k, pmf, width=1.0, color="#9ecae1", edgecolor="#6baed6", linewidth=0.3,
           label="Binomial(200, 0.5): green count of a human text")
    cut = 129  # smallest G with z > 4
    ax.bar(k[k >= cut], pmf[k >= cut], width=1.0, color=RED)
    ax.axvline(cut - 0.5, color="black", ls="--", lw=1)
    ax.text(cut + 1, 0.045, "z > 4 cut-off\n(G >= 129)", fontsize=10)
    ax.annotate("human text ca05\nG = 93, z = -0.99", xy=(93, stats.binom.pmf(93, T, .5)),
                xytext=(64, 0.05), arrowprops=dict(arrowstyle="->", color=HUM), color=HUM, fontsize=10)
    ax.annotate("GPT-2 text ca05 (δ = 1)\nG = 140, z = 5.66\np = 7.5e-9", xy=(140, 0.0005),
                xytext=(142, 0.03), arrowprops=dict(arrowstyle="->", color=D1), color=D1, fontsize=10)
    ax.plot([140], [0.0005], "v", color=D1, ms=10)
    ax.set_xlabel("number of green tokens G (out of T = 200)")
    ax.set_ylabel("probability under H0")
    ax.set_ylim(0, 0.062)
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "null_distribution")


# ------------------------------------------------------------------ alpha / beta
def fig_alpha_beta():
    T = 100
    k = np.arange(25, 96)
    h = stats.binom.pmf(k, T, 0.5)
    a = stats.binom.pmf(k, T, 0.701)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(k, h, 1.0, color=HUM, alpha=0.35, label="human text: Binomial(100, 0.50)")
    ax.bar(k, a, 1.0, color=D1, alpha=0.35, label="δ = 1 text: Binomial(100, 0.70)")
    cut = 71
    ax.bar(k[k < cut], a[k < cut], 1.0, color=D1, alpha=0.9, label=f"Type II error (missed AI): {100*stats.binom.cdf(cut-1,T,0.701):.0f}%")
    ax.bar(k[k >= cut], h[k >= cut], 1.0, color=HUM, alpha=1.0, label=f"Type I error (false alarm): {100*stats.binom.sf(cut-1,T,0.5):.4f}%")
    ax.axvline(cut - 0.5, color="black", ls="--", lw=1)
    ax.text(cut + 1, 0.06, "flag if z > 4\n(G >= 71)", fontsize=10)
    ax.set_xlabel("green tokens G out of T = 100")
    ax.set_ylabel("probability")
    ax.set_title("At 100 tokens the strict test misses about half of the δ = 1 texts")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "alpha_beta")


# ------------------------------------------------------------------ power curves
def power(T, p, zt):
    g = np.floor(T / 2 + zt * np.sqrt(T / 4)) + 1
    return stats.binom.sf(g - 1, T, p)


def fig_power():
    p1 = tab[(tab.source == "ai") & (tab.delta == 1) & (tab.length == 200)].gs.mean()
    p2 = tab[(tab.source == "ai") & (tab.delta == 2) & (tab.length == 200)].gs.mean()
    Ts = np.arange(5, 401, 5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, zt, name in ((axes[0], 4.0, "strict test (z > 4)"), (axes[1], 1.645, "usual test (alpha = 0.05)")):
        for d, p, c in ((1, p1, D1), (2, p2, D2)):
            Tf = np.linspace(5, 400, 400)
            pw = stats.norm.sf((0.5 + zt * np.sqrt(0.25 / Tf) - p) / np.sqrt(p * (1 - p) / Tf))
            ax.plot(Tf, pw, color=c, lw=1.5, label=f"δ = {d}: theory (p = {p:.2f})")
            pts = []
            for L in (25, 50, 100, 200):
                s = tab[(tab.source == "ai") & (tab.delta == d) & (tab.length == L)]
                col = "f4" if zt == 4 else "f05"
                pts.append((L,) + wilson(int(s[col].sum()), len(s)))
            L, m, lo, hi = zip(*pts)
            ax.errorbar(L, m, yerr=[lo, hi], fmt="o", color=c, ms=5, capsize=3, label=f"δ = {d}: our GPT-2 texts")
        ax.set_title(name)
        ax.set_xlabel("text length (tokens)")
        ax.set_xlim(0, 400)
    axes[0].set_ylabel("share of AI texts detected")
    axes[0].legend(loc="lower right", fontsize=10)
    save(fig, "power_curves")
    return p1, p2


# ------------------------------------------------------------------ Bayes
def fig_bayes():
    prev = np.logspace(-3, np.log10(0.5), 200)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for fpr, pw, lab, c in ((0.05, 1.0, "alpha = 0.05", "#9467bd"), (stats.norm.sf(4), 0.95, "z > 4", "#2ca02c")):
        ppv = pw * prev / (pw * prev + fpr * (1 - prev))
        ax.plot(100 * prev, 100 * ppv, color=c, lw=2, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 50])
    ax.set_xticklabels(["0.1%", "1%", "10%", "50%"])
    ax.axvline(1, color="grey", ls=":", lw=1)
    ax.annotate("17%", xy=(1, 16.8), xytext=(1.5, 25), arrowprops=dict(arrowstyle="->"))
    ax.annotate("99.7%", xy=(1, 99.7), xytext=(1.5, 88), arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("share of texts that really are AI (prevalence)")
    ax.set_ylabel("P(really AI | flagged)  (%)")
    ax.set_title("How much can we trust a flag? (Bayes' theorem)")
    ax.legend(loc="lower right")
    save(fig, "bayes")


# ------------------------------------------------------------------ overview of z by group
def fig_z_groups():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.arange(-4, 15, 0.4)
    groups = [("human", res[res.source == "human"], HUM),
              ("GPT-2, δ = 0 (no watermark)", res[(res.source == "ai") & (res.delta == 0)], D0),
              ("GPT-2, δ = 1", res[(res.source == "ai") & (res.delta == 1)], D1),
              ("GPT-2, δ = 2", res[(res.source == "ai") & (res.delta == 2)], D2)]
    for name, g, c in groups:
        ax.hist(g.z, bins=bins, histtype="stepfilled", alpha=0.35, color=c, label=f"{name} (mean z {g.z.mean():.2f})")
        ax.hist(g.z, bins=bins, histtype="step", color=c, lw=1.2)
    ax.axvline(4, color="black", ls="--", lw=1)
    ax.text(4.1, 105, "z = 4", fontsize=10)
    ax.set_xlabel("z-score of a 200-token text")
    ax.set_ylabel("number of texts (498 per group)")
    ax.set_ylim(0, 150)
    ax.legend(loc="upper left", fontsize=9.5, ncol=2)
    save(fig, "z_by_group")


# ------------------------------------------------------------------ GOF
def fig_gof():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, g, c) in zip(axes, (("Human texts (498)", res[res.source == "human"], HUM),
                                       ("GPT-2 without watermark (358 with T = 200)",
                                        res[(res.source == "ai") & (res.delta == 0)], D0))):
        g = g[g["T"] == 200]
        n = len(g)
        bins = np.arange(70.5, 131.5, 3)
        ax.hist(g.G, bins=bins, color=c, alpha=0.6, edgecolor="white", label="observed")
        k = np.arange(70, 132)
        ax.plot(k, n * 3 * stats.binom.pmf(k, 200, .5), color="black", lw=1.5, label="expected if H0 is true")
        ax.set_title(f"{name}\nvariance {g.G.var(ddof=1):.1f} (H0 says 50)", fontsize=11.5)
        ax.set_xlabel("green tokens G out of 200")
        ax.legend(loc="upper left", fontsize=9.5)
    axes[0].set_ylabel("number of texts")
    axes[0].text(118, 55, "chi-sq = 59.4, df = 30\np = 0.001", fontsize=10)
    axes[1].text(118, 55, "chi-sq = 26.5, df = 26\np = 0.44", fontsize=10)
    save(fig, "gof_human_vs_gpt2")


# ------------------------------------------------------------------ keys
def fig_keys():
    k = pd.read_csv(KEYS_CSV)
    k["label"] = ["team"] + [str(int(x)) for x in k.key[1:]]
    k = k.sort_values("fp_raw").reset_index(drop=True)
    x = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, 100 * k.fp_raw, "o", color=HUM, label="textbook count")
    ax.plot(x, 100 * k.fp_dedup, "s", color="#9467bd", ms=5, label="repeated pairs counted once")
    ax.plot(x, 100 * k.fp_test_half_calibrated, "^", color=GREEN, label="calibrated (held-out half)")
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.text(len(k) - 0.5, 5.3, "5% (what H0 promises)", ha="right", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(k.label, fontsize=9)
    ti = int(np.where(k.label == "team")[0][0])
    ax.get_xticklabels()[ti].set_fontweight("bold")
    ax.set_xlabel("secret key (sorted by false-positive rate)")
    ax.set_ylabel("human texts flagged at alpha = 0.05 (%)")
    ax.set_title("Same 498 human texts, 20 different keys")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "keys_false_positive")


# ------------------------------------------------------------------ common pairs
def fig_pairs():
    try:                                   # GPT-2 tokenizer: Hugging Face (Colab) or gpt3-tokenizer (same vocabulary)
        from transformers import GPT2TokenizerFast
        tk = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:
        import gpt3_tokenizer as tk
    eng = WatermarkEngine(WatermarkConfig(key=15485863, gamma=0.5), Vocabulary.of_size(50257))
    pairs = Counter()
    for r in res[res.source == "human"].itertuples():
        seq = [int(r.context_id)] + [int(t) for t in r.token_ids.split()]
        pairs.update(zip(seq[:-1], seq[1:]))
    top = pairs.most_common(15)[::-1]
    labels = [(tk.decode([p]) + tk.decode([t])).strip() for (p, t), _ in top]
    cols = [GREEN if eng.greens.is_green(p, t) else RED for (p, t), _ in top]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(range(15), [c for _, c in top], color=cols)
    ax.set_yticks(range(15))
    ax.set_yticklabels([f'"{l}"' for l in labels], fontsize=11)
    ax.set_xlabel("times the pair occurs in the 498 human texts")
    ax.set_title("Most common token pairs and their colour under our key", fontsize=12)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="green under our key"), Patch(color=RED, label="red under our key")],
              loc="lower right", fontsize=10)
    save(fig, "common_pairs")


# ------------------------------------------------------------------ regression
def fig_regression():
    rng = np.random.default_rng(3)
    base = tab.copy()
    base["grp"] = np.where(base.source == "human", "human", "d" + base.delta.fillna(0).astype(int).astype(str))
    base = base[base.grp.isin(["human", "d1", "d2"])]
    base["sT"] = np.sqrt(base.T_raw)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    xs = np.linspace(0, 15, 50)
    for g, c, lab in (("human", HUM, "human"), ("d1", D1, "δ = 1"), ("d2", D2, "δ = 2")):
        s = base[base.grp == g]
        ax.scatter(s.sT + rng.normal(0, 0.12, len(s)), s.zr, s=5, alpha=0.18, color=c)
        b1, b0 = np.polyfit(s.sT, s.zr, 1)
        ax.plot(xs, b0 + b1 * xs, color=c, lw=2, label=f"{lab}: z = {b0:.2f} + {b1:.3f} sqrt(T)")
    ax.axhline(4, color="black", ls="--", lw=1)
    ax.set_xlabel("sqrt(T)   (T = 25, 50, 100, 200 tokens)")
    ax.set_ylabel("z-score")
    ax.set_title("Evidence grows with the square root of length")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim(0, 15)
    ax2 = axes[1]
    for g, c, lab in (("d1", D1, "δ = 1"), ("d2", D2, "δ = 2")):
        s = base[base.grp == g]
        b1, b0 = np.polyfit(s.sT, s.zr, 1)
        r = s.zr - (b0 + b1 * s.sT)
        sd = r.groupby(s.length).std()
        ax2.plot(sd.index, sd.values, "o-", color=c, label=lab)
        p = s[s.length == 200].gs.mean()
        ax2.axhline(np.sqrt(4 * p * (1 - p)), color=c, ls=":", lw=1)
    ax2.set_ylim(0.5, 1.2)
    ax2.set_xticks([25, 50, 100, 200])
    ax2.set_xlabel("text length (tokens)")
    ax2.set_ylabel("SD of residuals")
    ax2.set_title("Residual spread stays about constant\n(dotted: theory sqrt(4p(1-p)))", fontsize=11.5)
    ax2.legend(fontsize=9.5)
    save(fig, "regression")


# ------------------------------------------------------------------ p-chart
def build_doc(eng, humans, ais, k, window=50):
    def take(recs, n, start):
        out, i = [], start
        while len(out) < n:
            out += [int(t) for t in recs.iloc[i % len(recs)].token_ids.split()]
            i += 1
        return out[:n], i
    p1, nxt = take(humans, 400, 2 * k)
    p2, _ = take(ais, 200, k)
    p3, _ = take(humans, 400, nxt)
    ids = p1 + p2 + p3
    cols = eng.colours(ids, None)
    rows = []
    for s in range(0, len(cols) - window + 1, window):
        rows.append(((s + 1 + s + 1 + window) / 2, sum(cols[s:s + window]) / window, 400 <= s + 1 < 600))
    return rows


def fig_pchart():
    eng = WatermarkEngine(WatermarkConfig(key=15485863, gamma=0.5), Vocabulary.of_size(50257))
    humans = res[res.source == "human"].reset_index(drop=True)
    for d, w, name in ((2, 50, "pchart_delta2"), (1, 50, "pchart_delta1_w50"), (1, 100, "pchart_delta1_w100")):
        ais = res[(res.source == "ai") & (res.delta == d)].reset_index(drop=True)
        rows = build_doc(eng, humans, ais, 0, w)
        sd = np.sqrt(0.25 / w)
        ucl, lcl = 0.5 + 3 * sd, 0.5 - 3 * sd
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axvspan(400, 600, color="#fdd0a2", alpha=0.5, lw=0)
        ax.text(500, 1.0, "AI-written part", ha="center", fontsize=10)
        x = [r[0] for r in rows]; y = [r[1] for r in rows]
        ax.plot(x, y, "-", color="grey", lw=1)
        ax.plot([a for a, b, c in rows if not c], [b for a, b, c in rows if not c], "o", color=HUM, label="human window")
        ax.plot([a for a, b, c in rows if c], [b for a, b, c in rows if c], "o", color=D2 if d == 2 else D1, label="AI window")
        ax.axhline(0.5, color="black", lw=1)
        ax.axhline(ucl, color="black", ls="--", lw=1)
        ax.axhline(lcl, color="black", ls=":", lw=1)
        ax.text(1005, ucl, f"UCL = {ucl:.3f}", va="center", fontsize=10)
        ax.text(1005, 0.5, "CL = 0.5", va="center", fontsize=10)
        ax.text(1005, lcl, f"LCL = {lcl:.3f}", va="center", fontsize=10)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.15, 1.05)
        ax.set_xlabel(f"position in document (tokens), windows of {w}")
        ax.set_ylabel("green share in window")
        ax.set_title(f"p-chart for a 1000-token document (400 human + 200 GPT-2 δ = {d} + 400 human)")
        ax.legend(loc="lower left", fontsize=10)
        save(fig, name)


# ------------------------------------------------------------------ exact vs z
def fig_exact_vs_z():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for T, c in ((10, "#8c564b"), (40, "#9467bd"), (200, "#17becf")):
        G = np.arange(int(T / 2), T + 1)
        z = (G - T / 2) / np.sqrt(T / 4)
        pz = stats.norm.sf(z)
        pe = stats.binom.sf(G - 1, T, .5)
        m = pe > 1e-6
        ax.plot(pe[m], pz[m], "o-", ms=3, color=c, label=f"T = {T}")
    ax.plot([1e-6, 1], [1e-6, 1], color="black", lw=1, ls="--", label="perfect agreement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("exact binomial p-value")
    ax.set_ylabel("z-test p-value")
    ax.set_title("Below the line: z-test claims more evidence than there is")
    ax.legend(fontsize=10)
    save(fig, "exact_vs_z")


# ------------------------------------------------------------------ toy LDA (appendix)
def fig_toy_lda():
    H = [(70, 40), (37.5, 30), (65, 45), (50, 41), (40, 46), (57.5, 42), (47.5, 34), (60, 43), (45, 50), (57.5, 44)]
    A = [(60, 29), (60, 33), (75, 27), (57.5, 27), (67.5, 28), (60, 44), (70, 32), (80, 48), (70, 38), (72.5, 35)]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(*zip(*H), color=HUM, s=40, zorder=3, label="human (simulated)")
    ax.scatter(*zip(*A), color=D1, s=40, zorder=3, marker="s", label="AI (simulated)")
    for lab, (x, y) in (("H1", H[0]), ("A6", A[5])):
        ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=10)
    xs = np.linspace(35, 82, 50)
    ax.plot(xs, (0.2158 * xs - 3.317) / 0.2554, color="black", lw=1.2, ls="--", label="LDA boundary")
    ax.set_xlabel("green %")
    ax.set_ylabel("perplexity")
    ax.set_title("Toy data: two features separate 18 of 20 texts")
    ax.legend(fontsize=9.5, loc="upper left")
    save(fig, "toy_lda")


if __name__ == "__main__":
    fig_toy(); fig_null(); fig_alpha_beta(); print(fig_power()); fig_bayes(); fig_z_groups(); fig_gof()
    fig_keys(); fig_pairs(); fig_regression(); fig_pchart()
    fig_exact_vs_z(); fig_toy_lda()
