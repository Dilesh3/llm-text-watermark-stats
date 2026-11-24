"""
make_human_texts.py - builds data/human_texts.csv from the Brown corpus.

Why the Brown corpus: 500 samples of American English published in 1961,
in 15 genres (news, essays, fiction, ...). It is guaranteed human-written,
since it predates AI text generators by decades.

For every Brown file we take (in order):
  1. the first ~1000 words  -> only used to train the prototype model
  2. a PROMPT of complete sentences, at least 30 words
  3. the HUMAN CONTINUATION, the next sentences, at least 260 words

The AI later continues the same prompt, so human and AI texts share the
same topic (a fair, paired comparison).

Run once:  python make_human_texts.py   (needs nltk + the Brown corpus)
"""

import csv
import os

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.insert(0, os.path.join(os.path.dirname(HERE), "nltk_data"))
from nltk.corpus import brown  # noqa: E402

GROUPS = {
    "news": "press", "editorial": "press", "reviews": "press",
    "belles_lettres": "nonfiction", "learned": "nonfiction", "lore": "nonfiction",
    "government": "nonfiction", "religion": "nonfiction", "hobbies": "nonfiction",
    "fiction": "fiction", "mystery": "fiction", "science_fiction": "fiction",
    "adventure": "fiction", "romance": "fiction", "humor": "fiction",
}

detok = TreebankWordDetokenizer()


def clean(words):
    words = ['"' if w in ("``", "''") else w for w in words]
    return detok.detokenize(words).replace(" ;", ";").replace(" :", ":")


def take(sents, start, min_words):
    out, n, i = [], 0, start
    while i < len(sents) and n < min_words:
        out.append(sents[i]); n += len(sents[i]); i += 1
    return out, i, n


def main():
    rows, train_rows = [], []
    for f in brown.fileids():
        sents = brown.sents(f)
        train, i, _ = take(sents, 0, 1000)
        prompt, i, n_p = take(sents, i, 30)
        human, i, n_h = take(sents, i, 260)
        if n_h < 260 or n_p > 120:
            continue
        genre = brown.categories(f)[0]
        rows.append(dict(
            text_id=f, genre=genre, group=GROUPS[genre],
            prompt=" ".join(clean(s) for s in prompt),
            human_text=" ".join(clean(s) for s in human),
        ))
        train_rows.append(dict(text_id=f, train_text=" ".join(clean(s) for s in train)))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "human_texts.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    os.makedirs(os.path.join(HERE, "prototype"), exist_ok=True)
    with open(os.path.join(HERE, "prototype", "train_texts.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["text_id", "train_text"])
        w.writeheader(); w.writerows(train_rows)
    print(f"wrote {len(rows)} human texts")


if __name__ == "__main__":
    main()
