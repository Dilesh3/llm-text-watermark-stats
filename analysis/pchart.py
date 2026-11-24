"""
pchart.py — Chapter 7: find the AI-written section of a mixed document
with a p-control chart (Module 5).

Builds documents laid out as   human | AI | human   (default 400 | 200 | 400
tokens) from the dataset, splits each into non-overlapping windows, and
plots the green share per window against the 3σ control limits.

Usage
-----
  python pchart.py ../wm/prototype/results_prototype.csv --delta 2 --window 50 --png pchart.png --csv pchart.csv
  python pchart.py ../wm/prototype/results_prototype.csv --delta 2 --window 50 --docs 50     # rates over 50 docs

The plot shows document 1; --docs N also reports, over N documents, how often
AI windows cross the upper limit (hit rate) and how often human windows do
(false alarms).
"""

import argparse
import csv

import numpy as np

from results import TEAM_KEY, load
from watermark import p_chart_limits

HUMAN_C, AI_C = "#2a78d6", "#eb6834"      # one fixed colour each, used in every plot
INK, MUTED = "#2b2b2a", "#8a8984"


def take(recs, n_tokens, start):
    """Concatenate ids from recs (starting at index `start`) until n_tokens."""
    out, i = [], start
    while len(out) < n_tokens:
        out += recs[i % len(recs)]["ids"]
        i += 1
    return out[:n_tokens], i


def build_doc(humans, ais, h1, a, h2, k):
    part1, nxt = take(humans, h1, 2 * k)
    part2, _ = take(ais, a, k)
    part3, _ = take(humans, h2, nxt)
    ids = part1 + part2 + part3
    labels = ["human"] * h1 + ["ai"] * a + ["human"] * h2
    return ids, labels


def score_doc(eng, ids, labels, window):
    rows = eng.window_counts(ids, window)
    for r in rows:
        share_ai = labels[r["start"]: r["end"]].count("ai") / window
        r["truth"] = "ai" if share_ai >= 0.5 else "human"
        r["share_ai"] = share_ai
    return rows


def plot(rows, lcl, centre, ucl, window, ai_span, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    x = [(r["start"] + r["end"]) / 2 for r in rows]
    y = [r["p"] for r in rows]
    ax.axvspan(*ai_span, color=AI_C, alpha=0.08, lw=0)
    ax.text(sum(ai_span) / 2, 0.99, "AI-written section", ha="center", va="top",
            color=MUTED, fontsize=9, transform=ax.get_xaxis_transform())
    ax.axhline(centre, color=MUTED, lw=1)
    ax.axhline(ucl, color=INK, lw=1, ls="--")
    ax.axhline(lcl, color=MUTED, lw=1, ls=":")
    tr = ax.get_yaxis_transform()            # x in axes units, y in data units
    for yv, lab, c in ((ucl, f"UCL {ucl:.2f}", INK), (centre, f"γ = {centre:.2f}", MUTED),
                       (lcl, f"LCL {lcl:.2f}", MUTED)):
        ax.text(1.01, yv, lab, transform=tr, va="center", fontsize=9, color=c, clip_on=False)
    ax.set_xlim(0, rows[-1]["end"])
    ax.plot(x, y, color=MUTED, lw=1, zorder=1)
    for truth, c, lab in (("human", HUMAN_C, "Human window"), ("ai", AI_C, "AI window")):
        xs = [xi for xi, r in zip(x, rows) if r["truth"] == truth]
        ys = [r["p"] for r in rows if r["truth"] == truth]
        ax.scatter(xs, ys, s=40, color=c, edgecolor="white", linewidth=1.5, zorder=3, label=lab)
    flagged = [(xi, r["p"]) for xi, r in zip(x, rows) if r["p"] > ucl]
    if flagged:
        ax.scatter(*zip(*flagged), s=140, facecolor="none", edgecolor=INK, linewidth=1.2,
                   zorder=4, label="Above UCL (flagged)")
    ax.set_xlabel(f"Token position (windows of {window} tokens)", color=INK)
    ax.set_ylabel("Green share in window", color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--delta", type=float, default=2.0, help="δ of the AI section")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--layout", type=int, nargs=3, default=[400, 200, 400],
                    metavar=("HUMAN1", "AI", "HUMAN2"))
    ap.add_argument("--docs", type=int, default=1, help="number of documents to evaluate")
    ap.add_argument("--png", default="pchart.png")
    ap.add_argument("--csv", default="pchart.csv")
    ap.add_argument("--vocab-size", type=int, help="auto-detected if omitted")
    ap.add_argument("--key", type=int, default=TEAM_KEY)
    a = ap.parse_args()

    meta, recs, eng = load(a.data, a.vocab_size, a.key)
    humans = [r for r in recs if r["source"] == "human"]
    ais = [r for r in recs if r["source"] == "ai" and r["delta"] == a.delta]
    if not ais:
        raise SystemExit(f"no AI texts with δ={a.delta:g} in {a.data}")
    centre, lcl, ucl = p_chart_limits(a.window, meta["gamma"])
    h1, n_ai, h2 = a.layout

    all_rows = []
    for k in range(a.docs):
        ids, labels = build_doc(humans, ais, h1, n_ai, h2, k)
        for r in score_doc(eng, ids, labels, a.window):
            r["doc"] = k
            all_rows.append(r)

    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "start", "end", "T", "G", "p",
                                          "truth", "share_ai"])
        w.writeheader()
        w.writerows(all_rows)

    ai_w = [r for r in all_rows if r["truth"] == "ai"]
    hu_w = [r for r in all_rows if r["truth"] == "human"]
    hit = np.mean([r["p"] > ucl for r in ai_w]) if ai_w else float("nan")
    fa = np.mean([r["p"] > ucl for r in hu_w]) if hu_w else float("nan")
    print(f"window={a.window}  limits: LCL={lcl:.3f}  centre={centre:.2f}  UCL={ucl:.3f}")
    print(f"{a.docs} document(s): AI windows above UCL {hit:.0%} ({len(ai_w)} windows), "
          f"human windows above UCL {fa:.1%} ({len(hu_w)} windows)")

    doc0 = [r for r in all_rows if r["doc"] == 0]
    n_flag = sum(r["p"] > ucl for r in doc0 if r["truth"] == "ai")
    n_ai_w = sum(r["truth"] == "ai" for r in doc0)
    title = (f"{n_flag} of {n_ai_w} AI windows cross the control limit; "
             f"{sum(r['p'] > ucl for r in doc0 if r['truth'] == 'human')} human windows do")
    plot(doc0, lcl, centre, ucl, a.window, (h1, h1 + n_ai), a.png, title)
    print(f"wrote {a.png} and {a.csv}")


if __name__ == "__main__":
    main()
