"""
demo.py — Reproduces every number in the PDF, then runs an end-to-end
simulation (generate watermarked + unwatermarked text, detect both).

    python demo.py
"""

import random

from watermark import (Vocabulary, WatermarkConfig, WatermarkEngine,
                       GreenListGenerator, softmax, test_counts, binom_sf,
                       error_rates)


def line(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
# PART B — toy example, exactly as in the PDF
# --------------------------------------------------------------------------- #
line("PART B: planting the watermark (toy scheme, K = 200)")

# Candidate words in fixed order (positions 1-4). IDs here are arbitrary labels;
# the toy green list depends only on vocabulary ORDER and the previous word's ID.
cands = Vocabulary(["mat", "sofa", "floor", "bed"], ids=[1, 2, 3, 4])
WORD_ID = {"the": 262, "a": 1600}

cfg = WatermarkConfig(key=200, gamma=0.5, delta=2.0, scheme="toy")
eng = WatermarkEngine(cfg, cands)

scores = {cands.word_to_id[w]: s for w, s in
          [("mat", 2.0), ("sofa", 1.3), ("floor", 1.0), ("bed", 0.6)]}

print("\nStep 2 - softmax without watermark:")
for t, p in softmax(scores).items():
    print(f"  {cands.id_to_word[t]:6s} {p:6.1%}")

for prev in ("the", "a"):
    info = eng.greens.explain(WORD_ID[prev])
    nums = {cands.id_to_word[t]: n for t, n in info["numbers"].items()}
    green = sorted(cands.id_to_word[t] for t in info["green"])
    print(f"\nStep 3 - previous word '{prev}' (ID {WORD_ID[prev]}): "
          f"seed={info['seed']}  numbers={nums}  green={green}")

wrong = GreenListGenerator(WatermarkConfig(key=201, scheme="toy"), cands.ids)
info = wrong.explain(262)
print(f"\nWrong key K=201 after 'the': seed={info['seed']}  "
      f"numbers={[info['numbers'][i] for i in cands.ids]}  "
      f"green={sorted(cands.id_to_word[t] for t in info['green'])}")

counts = {w: 0 for w in cands.words}
for k in range(1, 101):
    g = GreenListGenerator(WatermarkConfig(key=k, scheme="toy"), cands.ids)
    for t in g.green_list(262):
        counts[cands.id_to_word[t]] += 1
print(f"Green under how many of keys 1..100: {counts}")

print("\nStep 4 - softmax WITH watermark after 'the':")
before = softmax(scores)
after = eng.next_token_probs(WORD_ID["the"], scores)
for t in scores:
    print(f"  {cands.id_to_word[t]:6s} {before[t]:6.1%} -> {after[t]:6.1%}")
green = eng.greens.green_list(WORD_ID["the"])
print(f"  P(green word): {sum(before[t] for t in green):.0%} -> "
      f"{sum(after[t] for t in green):.0%}")


# --------------------------------------------------------------------------- #
# PARTS C & D — the hypothesis test
# --------------------------------------------------------------------------- #
line("PARTS C & D: detection z-test (alpha = 0.05 -> z > 1.645)")
for name, G, T in [("Ex1 10-word AI text", 7, 10),
                   ("Ex2 40-word AI text", 28, 40),
                   ("Ex3 200-word AI text", 140, 200),
                   ("Ex4 200-word human essay", 104, 200)]:
    print(f"\n{name}:\n  " + test_counts(G, T, 0.5, z_threshold=1.645).summary())


# --------------------------------------------------------------------------- #
# PART E — thresholds, errors, power
# --------------------------------------------------------------------------- #
line("PART E: Type I error and power (AI text 70% green, exact binomial)")
print(f"{'length':>7} {'threshold':>10} {'min G':>6} {'Type I':>10} {'power':>7}")
for T in (40, 200):
    for zt in (1.645, 4.0):
        r = error_rates(T, 0.5, zt, ai_green_rate=0.7)
        print(f"{T:>7} {'z>' + str(zt):>10} {r['min_green']:>6} "
              f"{r['type1']:>10.3%} {r['power']:>7.0%}")

print(f"\nExact vs z-test, 28/40: exact p = {binom_sf(28, 40, 0.5):.4f} "
      f"(z-test 0.0057)")
print(f"Exact vs z-test, 7/10:  exact p = {binom_sf(7, 10, 0.5):.2f} "
      f"(z-test 0.10)")


# --------------------------------------------------------------------------- #
# END-TO-END — real "hash" scheme, a fake language model, 1,000-word vocab
# --------------------------------------------------------------------------- #
line("END-TO-END: generate with/without watermark, then detect (hash scheme)")

vocab = Vocabulary([f"w{i}" for i in range(1000)])
SECRET = 0x5EC12E7_C0FFEE
engine = WatermarkEngine(WatermarkConfig(key=SECRET, gamma=0.5, delta=2.0,
                                         scheme="hash", z_threshold=4.0), vocab)


def fake_lm(context):
    """Stand-in LLM: deterministic pseudo-random scores conditioned on last token."""
    r = random.Random(context[-1] * 7919)
    return {t: r.gauss(0, 1.5) for t in vocab.ids}


rng = random.Random(42)
prompt = [vocab.word_to_id["w7"]]
for T in (10, 40, 200):
    wm = engine.generate(fake_lm, prompt, T, rng, watermark=True)
    plain = engine.generate(fake_lm, prompt, T, rng, watermark=False)
    print(f"\n{T}-token text")
    print("  watermarked   : " + engine.detect_ids(wm, prev_id=prompt[-1]).summary())
    print("  no watermark  : " + engine.detect_ids(plain, prev_id=prompt[-1]).summary())

print("\nSample watermarked text:", vocab.decode(
    engine.generate(fake_lm, prompt, 12, random.Random(1))))

# Detection with the WRONG key looks like human text
wm = engine.generate(fake_lm, prompt, 200, random.Random(7))
wrong_key = WatermarkEngine(WatermarkConfig(key=SECRET + 1, scheme="hash"), vocab)
print("\n200 watermarked tokens checked with the WRONG key:\n  "
      + wrong_key.detect_ids(wm, prev_id=prompt[-1]).summary())
