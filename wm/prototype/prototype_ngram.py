"""
prototype_ngram.py - the whole pipeline end to end, WITHOUT GPT-2.

A small word-level language model (an interpolated trigram model) is trained
on the first ~1000 words of each Brown file. It plays the role of "the AI":
it continues each prompt with and without the watermark. Its text is much
less fluent than GPT-2's, but the statistics are identical, so the analysis
team can build and test all their code on this dataset while the GPT-2
dataset is generated in Colab.

Output: prototype/results_prototype.csv  (one row per text)
Run:    python prototype/prototype_ngram.py
"""

import collections
import csv
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from watermark_core import GAMMA, SECRET_KEY, detect, green_mask  # noqa: E402

N_TOKENS = 200
DELTAS = [0, 1, 2]
LAMBDAS = (0.2, 0.3, 0.5)            # weights of unigram, bigram, trigram
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class TrigramLM:
    def __init__(self, texts, min_count=2):
        counts = collections.Counter(t for txt in texts for t in tokenize(txt))
        self.itos = ["<unk>"] + sorted(w for w, c in counts.items() if c >= min_count)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.V = len(self.itos)
        uni = np.zeros(self.V)
        bi = collections.defaultdict(collections.Counter)
        tri = collections.defaultdict(collections.Counter)
        for txt in texts:
            ids = self.encode(txt)
            for i, w in enumerate(ids):
                uni[w] += 1
                if i >= 1:
                    bi[ids[i - 1]][w] += 1
                if i >= 2:
                    tri[(ids[i - 2], ids[i - 1])][w] += 1
        uni[0] = 0                                  # never generate <unk>
        self.p_uni = uni / uni.sum()
        self.bi = {k: self._arr(c) for k, c in bi.items()}
        self.tri = {k: self._arr(c) for k, c in tri.items()}

    @staticmethod
    def _arr(counter):
        ids = np.fromiter(counter.keys(), dtype=np.int64)
        cnt = np.fromiter(counter.values(), dtype=float)
        keep = ids != 0
        ids, cnt = ids[keep], cnt[keep]
        return ids, cnt / cnt.sum() if cnt.sum() else cnt

    def encode(self, text):
        return [self.stoi.get(t, 0) for t in tokenize(text)]

    def decode(self, ids):
        out = ""
        for i in ids:
            w = self.itos[i]
            out += w if (re.fullmatch(r"[^\w\s]", w) and w not in "(\"$") else " " + w
        return out.strip()

    def logits(self, u, v):
        """Scores for the next word given the two previous words (u, v)."""
        lam_u, lam_b, lam_t = LAMBDAS
        p = lam_u * self.p_uni
        total = lam_u
        if v in self.bi and len(self.bi[v][0]):
            ids, pr = self.bi[v]; p[ids] += lam_b * pr; total += lam_b
        if (u, v) in self.tri and len(self.tri[(u, v)][0]):
            ids, pr = self.tri[(u, v)]; p[ids] += lam_t * pr; total += lam_t
        p = p / total
        with np.errstate(divide="ignore"):
            return np.log(p)                        # -inf for <unk>

    def generate(self, prompt_ids, n, delta, rng, gamma=GAMMA, key=SECRET_KEY):
        ids = list(prompt_ids)
        for _ in range(n):
            s = self.logits(ids[-2], ids[-1])
            if delta:
                s = s + delta * green_mask(ids[-1], self.V, gamma, key)
            s = s - s[np.isfinite(s)].max()
            p = np.exp(s); p /= p.sum()
            ids.append(int(rng.choice(self.V, p=p)))
        return ids[len(prompt_ids):]


def main():
    human = pd.read_csv(os.path.join(ROOT, "data", "human_texts.csv"))
    train = pd.read_csv(os.path.join(HERE, "train_texts.csv"))
    lm = TrigramLM(list(train.train_text))
    print(f"vocabulary: {lm.V} words")
    rng = np.random.default_rng(2026)
    rows = []
    for k, r in enumerate(human.itertuples()):
        p_ids = lm.encode(r.prompt)
        base = dict(text_id=r.text_id, genre=r.genre, group=r.group)
        h_ids = lm.encode(r.human_text)[:N_TOKENS]
        rows.append(_row(base, "human", np.nan, h_ids, p_ids[-1], lm))
        for delta in DELTAS:
            g = lm.generate(p_ids, N_TOKENS, delta, rng)
            rows.append(_row(base, "ai", delta, g, p_ids[-1], lm))
        if k % 100 == 0:
            print(f"  {k}/{len(human)} prompts done")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "results_prototype.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"wrote {len(out)} rows")


def _row(base, source, delta, ids, ctx, lm):
    r = detect(ids, lm.V, context_token_id=ctx)
    rd = detect(ids, lm.V, context_token_id=ctx, ignore_repeated_bigrams=True)
    return dict(**base, source=source, delta=delta, T=r["T"], G=r["G"],
                green_frac=round(r["green_frac"], 4), z=round(r["z"], 4),
                p_z=r["p_z"], p_exact=r["p_exact"],
                T_dedup=rd["T"], G_dedup=rd["G"], z_dedup=round(rd["z"], 4),
                context_id=ctx, token_ids=" ".join(map(str, ids)), text=lm.decode(ids))


if __name__ == "__main__":
    main()
