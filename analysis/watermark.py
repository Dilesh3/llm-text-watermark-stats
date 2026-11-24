"""
watermark.py — Green-list text watermarking engine.

Implements the scheme from "AI Text Watermarking: A Complete Worked Example":

    previous token + secret key -> seed -> pseudo-random numbers
    -> lowest gamma-fraction of the vocabulary is GREEN
    -> (generation) add delta to green tokens' scores, softmax, sample
    -> (detection) count green tokens G out of T, one-proportion z-test
                   (and exact binomial test for short texts)

Two green-list schemes:

  * "toy"  — the exact mod-101 arithmetic from the PDF (hand-checkable).
             seed = (K * prev_id) mod 101
             next = (17 * prev + 5) mod 101   (one number per vocab word, in order)

  * "hash" — what a real system does: SHA-256 of (key, prev_id) seeds a
             numpy PRNG that shuffles the vocabulary. Fast enough for GPT-2's
             50,257-token vocabulary.

Requires numpy. The HuggingFace hook at the bottom also needs torch + transformers.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class WatermarkConfig:
    key: int                     # secret key K (keep private!)
    gamma: float = 0.5           # fraction of vocabulary that is green
    delta: float = 2.0           # score boost added to green tokens
    scheme: str = "hash"         # "toy" (PDF arithmetic) or "hash" (real)
    z_threshold: float = 4.0     # flag as watermarked if z > this

    def __post_init__(self):
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if self.scheme not in ("toy", "hash"):
            raise ValueError("scheme must be 'toy' or 'hash'")


# --------------------------------------------------------------------------- #
# Vocabulary: word <-> ID
# --------------------------------------------------------------------------- #

class Vocabulary:
    """Fixed word list. Order matters: it is the 'vocabulary order' of Step 3b."""

    def __init__(self, words: Sequence[str], ids: Optional[Sequence[int]] = None):
        self.words = list(words)
        self.ids = list(ids) if ids is not None else list(range(len(words)))
        if len(self.ids) != len(self.words):
            raise ValueError("words and ids must have the same length")
        self.word_to_id = dict(zip(self.words, self.ids))
        self.id_to_word = dict(zip(self.ids, self.words))

    @classmethod
    def of_size(cls, n: int) -> "Vocabulary":
        """Anonymous vocabulary 0..n-1 (e.g. a real model's token ids)."""
        return cls([str(i) for i in range(n)])

    def __len__(self):
        return len(self.words)

    def encode(self, text_or_words) -> List[int]:
        words = text_or_words.split() if isinstance(text_or_words, str) else text_or_words
        return [self.word_to_id[w] for w in words]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.id_to_word[i] for i in ids)


# --------------------------------------------------------------------------- #
# Green-list construction (Part B, Step 3)
# --------------------------------------------------------------------------- #

TOY_MOD = 101


def toy_seed(key: int, prev_id: int) -> int:
    """Step 3a: seed = (K * ID of previous word) mod 101."""
    return (key * prev_id) % TOY_MOD


def toy_numbers(seed: int, n: int) -> List[int]:
    """Step 3b: next = (17 * previous + 5) mod 101, one per vocab word."""
    out, x = [], seed
    for _ in range(n):
        x = (17 * x + 5) % TOY_MOD
        out.append(x)
    return out


def hash_seed(key: int, prev_id: int) -> int:
    """Real-system seed: SHA-256 of key and previous token id -> 64-bit int."""
    digest = hashlib.sha256(f"{key}:{prev_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class GreenListGenerator:
    """Rebuilds the green list for any position from (previous token, key)."""

    def __init__(self, config: WatermarkConfig, vocab_ids: Sequence[int],
                 cache_size: int = 4096):
        self.cfg = config
        self.vocab_ids = list(vocab_ids)
        self.n = len(self.vocab_ids)
        self.n_green = int(config.gamma * self.n)
        # Fast path when ids are exactly 0..n-1 (real tokenizers)
        self._contiguous = self.vocab_ids == list(range(self.n))
        self._pos = None if self._contiguous else {t: i for i, t in enumerate(self.vocab_ids)}
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

    def _position(self, token_id: int) -> int:
        return token_id if self._contiguous else self._pos[token_id]

    def green_mask(self, prev_id: int) -> np.ndarray:
        """Boolean array over vocabulary positions: True = green."""
        if prev_id in self._cache:
            self._cache.move_to_end(prev_id)
            return self._cache[prev_id]

        mask = np.zeros(self.n, dtype=bool)
        if self.cfg.scheme == "toy":
            nums = toy_numbers(toy_seed(self.cfg.key, prev_id), self.n)
            # Step 3c: smallest numbers are green (stable sort: ties by vocab order)
            order = sorted(range(self.n), key=lambda i: nums[i])
            mask[order[: self.n_green]] = True
        else:
            rng = np.random.default_rng(hash_seed(self.cfg.key, prev_id))
            mask[rng.permutation(self.n)[: self.n_green]] = True

        self._cache[prev_id] = mask
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return mask

    def green_list(self, prev_id: int) -> frozenset:
        mask = self.green_mask(prev_id)
        return frozenset(self.vocab_ids[i] for i in np.flatnonzero(mask))

    def is_green(self, prev_id: int, token_id: int) -> bool:
        return bool(self.green_mask(prev_id)[self._position(token_id)])

    def explain(self, prev_id: int) -> dict:
        """Seed, per-token random numbers and green list (toy scheme only)."""
        if self.cfg.scheme != "toy":
            raise ValueError("explain() is only meaningful for the toy scheme")
        seed = toy_seed(self.cfg.key, prev_id)
        nums = toy_numbers(seed, self.n)
        return {"seed": seed, "numbers": dict(zip(self.vocab_ids, nums)),
                "green": self.green_list(prev_id)}


# --------------------------------------------------------------------------- #
# Statistics (Parts C, D, E)
# --------------------------------------------------------------------------- #

def softmax(scores: Dict[int, float]) -> Dict[int, float]:
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: e / total for k, e in exps.items()}


def z_score(G: int, T: int, gamma: float) -> float:
    """z = (G - T*gamma) / sqrt(T*gamma*(1-gamma))."""
    return (G - T * gamma) / math.sqrt(T * gamma * (1 - gamma))


def normal_sf(z: float) -> float:
    """One-sided p-value P(Z >= z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def binom_sf(G: int, T: int, p: float) -> float:
    """Exact one-sided p-value P(X >= G) for X ~ Binomial(T, p)."""
    if G <= 0:
        return 1.0
    return min(1.0, sum(math.comb(T, k) * p**k * (1 - p) ** (T - k)
                        for k in range(G, T + 1)))


def min_green_to_flag(T: int, gamma: float, z_thresh: float) -> int:
    """Smallest green count G with z(G) > z_thresh."""
    G = math.floor(T * gamma + z_thresh * math.sqrt(T * gamma * (1 - gamma))) + 1
    while G > 0 and z_score(G - 1, T, gamma) > z_thresh:
        G -= 1
    return G


def error_rates(T: int, gamma: float, z_thresh: float, ai_green_rate: float) -> dict:
    """Exact Type I error and power (Part E) for a given length and threshold."""
    g = min_green_to_flag(T, gamma, z_thresh)
    if g > T:
        return {"min_green": g, "type1": 0.0, "power": 0.0}
    return {"min_green": g,
            "type1": binom_sf(g, T, gamma),
            "power": binom_sf(g, T, ai_green_rate)}


def p_chart_limits(n: int, gamma: float, k: float = 3.0) -> tuple:
    """Centre line and k-sigma control limits for a p-chart with window size n."""
    sd = math.sqrt(gamma * (1 - gamma) / n)
    return gamma, max(0.0, gamma - k * sd), min(1.0, gamma + k * sd)


@dataclass
class DetectionResult:
    T: int                      # tokens scored
    G: int                      # green tokens
    green_fraction: float
    expected: float
    std: float
    z: float
    p_value_z: float            # normal approximation
    p_value_exact: float        # exact binomial
    z_threshold: float
    is_watermarked: bool
    colours: List[Optional[bool]] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        verdict = "WATERMARKED" if self.is_watermarked else "not flagged"
        return (f"T={self.T}  G={self.G} ({self.green_fraction:.0%} green)  "
                f"expected={self.expected:.1f}  sd={self.std:.2f}  z={self.z:.2f}  "
                f"p(z)={self.p_value_z:.2g}  p(exact)={self.p_value_exact:.2g}  "
                f"-> {verdict} (threshold z>{self.z_threshold})")


def test_counts(G: int, T: int, gamma: float, z_threshold: float) -> DetectionResult:
    """Run the hypothesis test given green count G out of T."""
    z = z_score(G, T, gamma)
    return DetectionResult(
        T=T, G=G, green_fraction=G / T if T else 0.0,
        expected=T * gamma, std=math.sqrt(T * gamma * (1 - gamma)),
        z=z, p_value_z=normal_sf(z), p_value_exact=binom_sf(G, T, gamma),
        z_threshold=z_threshold, is_watermarked=z > z_threshold)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

# A "model" is any function: (list of previous token ids) -> {token_id: score}
ScoreFn = Callable[[List[int]], Dict[int, float]]


class WatermarkEngine:
    def __init__(self, config: WatermarkConfig, vocab: Vocabulary):
        self.cfg = config
        self.vocab = vocab
        self.greens = GreenListGenerator(config, vocab.ids)

    # ---- Planting (Part B) ------------------------------------------------ #

    def boost(self, prev_id: int, scores: Dict[int, float]) -> Dict[int, float]:
        """Step 4: add delta to every green token's score."""
        g = self.greens
        return {t: s + (self.cfg.delta if g.is_green(prev_id, t) else 0.0)
                for t, s in scores.items()}

    def next_token_probs(self, prev_id: int, scores: Dict[int, float],
                         watermark: bool = True) -> Dict[int, float]:
        return softmax(self.boost(prev_id, scores) if watermark else scores)

    def sample_next(self, prev_id: int, scores: Dict[int, float],
                    rng: random.Random, watermark: bool = True) -> int:
        """Step 5: sample from the (boosted) distribution."""
        probs = self.next_token_probs(prev_id, scores, watermark)
        r, acc = rng.random(), 0.0
        for tok, p in probs.items():
            acc += p
            if r < acc:
                return tok
        return tok  # floating-point fallback

    def generate(self, model: ScoreFn, prompt_ids: List[int], n_tokens: int,
                 rng: Optional[random.Random] = None, watermark: bool = True) -> List[int]:
        """Step 6: repeat for every token. Returns only the new tokens."""
        rng = rng or random.Random()
        ids = list(prompt_ids)
        for _ in range(n_tokens):
            scores = model(ids)
            ids.append(self.sample_next(ids[-1], scores, rng, watermark))
        return ids[len(prompt_ids):]

    # ---- Detecting (Part C / F) ------------------------------------------- #

    def colours(self, token_ids: Sequence[int], prev_id: Optional[int] = None,
                ignore_repeats: bool = False) -> List[Optional[bool]]:
        """
        Colour of each scored token: True (green), False (red) or None (skipped
        because the (previous, token) pair already appeared — only when
        ignore_repeats=True). If prev_id is None the first token is not scored.
        """
        ids = list(token_ids) if prev_id is None else [prev_id, *token_ids]
        seen, out = set(), []
        for i in range(1, len(ids)):
            pair = (ids[i - 1], ids[i])
            if ignore_repeats and pair in seen:
                out.append(None)
                continue
            seen.add(pair)
            out.append(self.greens.is_green(*pair))
        return out

    def detect_ids(self, token_ids: Sequence[int], prev_id: Optional[int] = None,
                   ignore_repeats: bool = False) -> DetectionResult:
        """
        Score a token sequence. If `prev_id` is given (e.g. last prompt token),
        the first token is scored too; otherwise scoring starts at token 2.
        With ignore_repeats=True each (previous, token) pair counts only once,
        so repeated phrases cannot inflate the green count.
        """
        cols = self.colours(token_ids, prev_id, ignore_repeats)
        scored = [c for c in cols if c is not None]
        T, G = len(scored), sum(scored)
        if T == 0:
            raise ValueError("need at least one scored token")
        res = test_counts(G, T, self.cfg.gamma, self.cfg.z_threshold)
        res.colours = cols
        return res

    def detect_text(self, text: str, prev_word: Optional[str] = None,
                    ignore_repeats: bool = False) -> DetectionResult:
        prev = self.vocab.word_to_id[prev_word] if prev_word else None
        return self.detect_ids(self.vocab.encode(text), prev, ignore_repeats)

    def window_counts(self, token_ids: Sequence[int], window: int,
                      prev_id: Optional[int] = None) -> List[dict]:
        """
        Non-overlapping windows for a p-chart (Module 5).
        Returns [{start, end, T, G, p}] with token positions in `token_ids`.
        """
        cols = self.colours(token_ids, prev_id)
        offset = 0 if prev_id is not None else 1      # cols[i] is token i+offset
        rows = []
        for s in range(0, len(cols) - window + 1, window):
            chunk = cols[s: s + window]
            G = sum(chunk)
            rows.append({"start": s + offset, "end": s + offset + window,
                         "T": window, "G": G, "p": G / window})
        return rows


# --------------------------------------------------------------------------- #
# Optional: HuggingFace transformers integration
# --------------------------------------------------------------------------- #

def make_hf_logits_processor(config: WatermarkConfig, vocab_size: int):
    """
    Build a transformers LogitsProcessor that applies this watermark to a real LLM.

        proc = make_hf_logits_processor(cfg, len(tokenizer))
        out = model.generate(**inputs, logits_processor=LogitsProcessorList([proc]),
                             do_sample=True)

    Detect afterwards with WatermarkEngine(cfg, Vocabulary.of_size(vocab_size))
    .detect_ids(new_token_ids, prev_id=last_prompt_token).
    Uses exactly the same green lists as the pure-Python engine.
    """
    import torch
    from transformers import LogitsProcessor

    gen = GreenListGenerator(config, list(range(vocab_size)))

    class GreenListLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for b in range(input_ids.shape[0]):
                mask = torch.from_numpy(gen.green_mask(int(input_ids[b, -1])))
                mask = mask.to(device=scores.device, dtype=scores.dtype)
                scores[b, :vocab_size] += config.delta * mask
            return scores

    return GreenListLogitsProcessor()
