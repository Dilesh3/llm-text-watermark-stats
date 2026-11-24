"""
prototype_figures.py - Chapter 2 figures from the prototype trigram model.
(The Colab notebook makes the same figures with GPT-2 for the final slides.)

Run: python prototype/prototype_figures.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from figures import coloured_text_figure, probability_bars  # noqa: E402
from prototype_ngram import TrigramLM  # noqa: E402
from watermark_core import detect, green_flags, green_mask  # noqa: E402

TEXT_ID = "ca05"   # a news text


def token_strings(lm, ids):
    return [lm.itos[i] if re.fullmatch(r"[^\w\s]", lm.itos[i]) else " " + lm.itos[i] for i in ids]


def main():
    res = pd.read_csv(os.path.join(HERE, "results_prototype.csv"))
    train = pd.read_csv(os.path.join(HERE, "train_texts.csv"))
    lm = TrigramLM(list(train.train_text))
    panels = []
    for source, delta, title in [("human", None, "Human (Brown corpus, 1961)"),
                                 ("ai", 2.0, "AI with watermark (δ = 2)")]:
        sel = res[(res.text_id == TEXT_ID) & (res.source == source)]
        if delta is not None:
            sel = sel[sel.delta == delta]
        r = sel.iloc[0]
        ids = list(map(int, r.token_ids.split()))
        flags = green_flags(ids, lm.V, context_token_id=int(r.context_id))
        panels.append(dict(title=title, tokens=token_strings(lm, ids), flags=flags,
                           stats=detect(ids, lm.V, context_token_id=int(r.context_id))))
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    coloured_text_figure(panels, os.path.join(HERE, "figures", "fig_coloured_text_prototype.png"))

    # probability bars at one real step
    context = "the jury said it did not find any evidence of"
    ids = lm.encode(context)
    s = lm.logits(ids[-2], ids[-1])
    mask = np.array(green_mask(ids[-1], lm.V))
    def sm(x):
        x = x - x[np.isfinite(x)].max(); e = np.exp(x); return e / e.sum()
    pb, pa = sm(s), sm(s + 2 * mask)
    order = np.argsort(-pb)[:10]
    rows = [dict(token=lm.itos[i], green=bool(mask[i]), p_before=float(pb[i]), p_after=float(pa[i]))
            for i in order]
    probability_bars(rows, float(pb[mask].sum()), float(pa[mask].sum()), context, 2,
                     os.path.join(HERE, "figures", "fig_probability_bars_prototype.png"))
    print("figures written")


if __name__ == "__main__":
    main()
