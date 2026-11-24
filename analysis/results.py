"""
results.py — Turns the generation results file into the analysis table every chapter
uses, and (with --pilot) recommends lengths from the measured green share.

Input: the generation results CSV (results/results_gpt2.csv, or the prototype
file wm/prototype/results_prototype.csv). It needs the columns
text_id, genre, group, source (human/ai), delta, T, G, context_id, token_ids.

Usage
-----
  python results.py ../wm/prototype/results_prototype.csv --pilot
  python results.py ../wm/prototype/results_prototype.csv --out analysis_table.csv
  python results.py results_gpt2.csv --out analysis_table.csv

Each text is scored at every length in --lengths (its first L tokens).
A text that re-tokenizes to a few tokens fewer than L (GPT-2 output often
comes back as 195-199 tokens) is still scored at L, using all its tokens.

Output columns
--------------
  text_id, genre, group, source (human/ai), delta, length,
  T, G, green_share, z, p_z, p_exact, flag_a05 (z>1.645), flag_z4 (z>4)
      -> each (previous token, token) pair counted once ("dedup")
  T_raw, G_raw, z_raw, flag_a05_raw
      -> every token counted (as in the textbook test)
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict

import numpy as np

from watermark import (Vocabulary, WatermarkConfig, WatermarkEngine, binom_sf,
                       error_rates, p_chart_limits)

A05 = 1.645
Z4 = 4.0
TEAM_KEY = 15485863                    # same as wm/watermark_core.py
VOCAB_CANDIDATES = {50257: "GPT-2", 17352: "trigram prototype"}


# --------------------------------------------------------------------------- #
# Loading the generation results file
# --------------------------------------------------------------------------- #

def _records(path):
    csv.field_size_limit(10**8)
    recs = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            delta = float(r["delta"]) if r["delta"] not in ("", "nan", "NaN") else 0.0
            recs.append({
                "text_id": r["text_id"], "genre": r["genre"], "group": r["group"],
                "source": r["source"], "delta": delta,
                "context_id": int(float(r["context_id"])),
                "ids": [int(t) for t in r["token_ids"].split()],
                "G_r2": int(float(r["G"])), "T_r2": int(float(r["T"])),
            })
    return recs


def _matches(eng, recs, n=8):
    for r in recs[:n]:
        d = eng.detect_ids(r["ids"], r["context_id"])
        if (d.G, d.T) != (r["G_r2"], r["T_r2"]):
            return False
    return True


def load(path, vocab_size=None, key=TEAM_KEY, gamma=0.5):
    """Load the generation CSV and build a detector that provably matches its scoring."""
    recs = _records(path)
    sizes = [vocab_size] if vocab_size else list(VOCAB_CANDIDATES)
    max_id = max(max(r["ids"]) for r in recs)
    for V in sizes:
        if V <= max_id:
            continue
        eng = WatermarkEngine(WatermarkConfig(key=key, gamma=gamma), Vocabulary.of_size(V))
        if _matches(eng, recs):
            meta = {"key": key, "gamma": gamma, "vocab_size": V,
                    "model": VOCAB_CANDIDATES.get(V, f"vocab {V}"),
                    "length": max(len(r["ids"]) for r in recs)}
            return meta, recs, eng
    raise SystemExit("Recomputed green counts do not match the file's G column. "
                     "Check --key and --vocab-size (they must equal the generator's).")


# --------------------------------------------------------------------------- #
# Analysis table
# --------------------------------------------------------------------------- #

def score_row(eng, rec, ids, length):
    d = eng.detect_ids(ids, rec["context_id"], ignore_repeats=True)
    r = eng.detect_ids(ids, rec["context_id"], ignore_repeats=False)
    return {"text_id": rec["text_id"], "genre": rec["genre"], "group": rec["group"],
            "source": rec["source"], "delta": rec["delta"] if rec["source"] == "ai" else "",
            "length": length,
            "T": d.T, "G": d.G, "green_share": round(d.green_fraction, 4),
            "z": round(d.z, 4), "p_z": f"{d.p_value_z:.4g}", "p_exact": f"{d.p_value_exact:.4g}",
            "flag_a05": int(d.z > A05), "flag_z4": int(d.z > Z4),
            "T_raw": r.T, "G_raw": r.G, "z_raw": round(r.z, 4), "flag_a05_raw": int(r.z > A05)}


def build_table(meta, recs, eng, lengths):
    rows = []
    for i, rec in enumerate(recs):
        for L in lengths:
            if L > len(rec["ids"]) + 5:      # keep texts re-tokenized to 195-199 tokens at L = 200
                continue
            rows.append(score_row(eng, rec, rec["ids"][:L], L))
        if (i + 1) % 250 == 0:
            print(f"  scored {i + 1}/{len(recs)} texts", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------- #
# Pilot report
# --------------------------------------------------------------------------- #

def pilot_report(meta, recs, eng):
    groups = defaultdict(list)
    for rec in recs:
        res = eng.detect_ids(rec["ids"], rec["context_id"], ignore_repeats=True)
        groups[(rec["source"], rec["delta"])].append(res.green_fraction)

    print(f"PILOT REPORT  model={meta['model']}  vocab={meta['vocab_size']}  "
          f"γ={meta['gamma']}  length={meta['length']} tokens\n")
    print(f"{'group':<16}{'n':>5}{'mean green':>12}{'sd':>8}")
    shares = {}
    for (src, d), v in sorted(groups.items(), key=lambda kv: (kv[0][0] != "human", kv[0][1])):
        name = "human" if src == "human" else f"AI δ={d:g}"
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{name:<16}{len(v):>5}{statistics.mean(v):>11.1%}{sd:>8.3f}")
        if src == "ai" and d > 0:
            shares[d] = statistics.mean(v)

    cand = [10, 25, 50, 75, 100, 150, 200, 300, 400]
    print("\nPower (exact binomial) by length:")
    print(f"{'':<22}" + "".join(f"{L:>6}" for L in cand))
    for d, p in sorted(shares.items()):
        for zt, lab in ((Z4, "z>4"), (A05, "α=0.05")):
            pw = [error_rates(L, meta["gamma"], zt, p)["power"] for L in cand]
            print(f"δ={d:g} ({p:.0%}) {lab:<8}" + "".join(f"{x:>6.0%}" for x in pw))

    print("\nRecommendations (compare where power is between 20% and 80%):")
    for d, p in sorted(shares.items()):
        mid = [L for L in cand if 0.2 <= error_rates(L, meta["gamma"], Z4, p)["power"] <= 0.8]
        print(f"  δ={d:g}: lengths with 20–80% power at z>4: {mid or 'none in range'}")
    if shares:
        d_top = max(shares, key=shares.get)
        for W in (25, 50, 100):
            _, _, ucl = p_chart_limits(W, meta["gamma"])
            hit = binom_sf(int(np.floor(ucl * W)) + 1, W, shares[d_top])
            print(f"  Chapter 7 p-chart, window {W}: an AI window (δ={d_top:g}) crosses "
                  f"the 3σ limit {hit:.0%} of the time")
        print("  -> use the smallest window with ≥80% for a sharp, clear spike.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="results CSV from the generation step")
    ap.add_argument("--out", default="analysis_table.csv")
    ap.add_argument("--lengths", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--vocab-size", type=int, help="auto-detected if omitted")
    ap.add_argument("--key", type=int, default=TEAM_KEY)
    ap.add_argument("--pilot", action="store_true", help="print the pilot report only")
    a = ap.parse_args()

    meta, recs, eng = load(a.data, a.vocab_size, a.key)
    print(f"loaded {len(recs)} texts ({meta['model']}, vocab {meta['vocab_size']}); "
          f"green counts match the generation file", file=sys.stderr)
    if a.pilot:
        pilot_report(meta, recs, eng)
        return
    rows = build_table(meta, recs, eng, a.lengths)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
