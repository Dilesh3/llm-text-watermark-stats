"""
watermark_core.py - the shared heart of the project.

Both the WATERMARK (generation) and the DETECTOR must use exactly these
functions, otherwise the detector rebuilds different green lists.

Method: the "soft red list" watermark of Kirchenbauer et al. (ICML 2023),
using the previous token as context.

  previous token + secret key --> seed --> shuffle vocabulary
  --> first gamma * |V| tokens of the shuffle are GREEN, the rest RED

Only numpy and scipy are needed here (no PyTorch), so the detector can run
anywhere.
"""

from functools import lru_cache
import hashlib

import numpy as np
from scipy import stats

# The team's secret key. Anyone holding it can detect the watermark.
SECRET_KEY = 15485863
GAMMA = 0.5


def _seed(prev_token_id: int, key: int) -> int:
    """Combine the secret key and the previous token into one 64-bit seed.

    SHA-256 is a cryptographic hash: changing the key or the token by 1
    gives a completely unrelated seed (this is the real-world version of the
    'mod 101' formula in our toy example).
    """
    digest = hashlib.sha256(f"{key}:{int(prev_token_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@lru_cache(maxsize=2048)
def _green_mask_cached(prev_token_id: int, vocab_size: int, gamma: float, key: int) -> np.ndarray:
    rng = np.random.default_rng(_seed(prev_token_id, key))
    order = rng.permutation(vocab_size)          # the random shuffle
    n_green = int(gamma * vocab_size)            # e.g. 0.5 * 50257 = 25128
    mask = np.zeros(vocab_size, dtype=bool)
    mask[order[:n_green]] = True                 # first part of the shuffle = green
    mask.setflags(write=False)                   # cached arrays must never be modified
    return mask


def green_mask(prev_token_id: int, vocab_size: int, gamma: float = GAMMA,
               key: int = SECRET_KEY) -> np.ndarray:
    """Boolean array of length vocab_size: True where the token is GREEN
    at a position whose previous token is prev_token_id."""
    return _green_mask_cached(int(prev_token_id), int(vocab_size), float(gamma), int(key))


def is_green(prev_token_id: int, token_id: int, vocab_size: int,
             gamma: float = GAMMA, key: int = SECRET_KEY) -> bool:
    return bool(green_mask(prev_token_id, vocab_size, gamma, key)[int(token_id)])


def green_flags(token_ids, vocab_size: int, context_token_id=None,
                gamma: float = GAMMA, key: int = SECRET_KEY,
                ignore_repeated_bigrams: bool = False):
    """Colour every token of a text.

    token_ids        : the text's token ids
    context_token_id : the token just before the text (e.g. last prompt token).
                       If None, the first token is skipped (it has no previous token).
    ignore_repeated_bigrams : if True, a (previous token, token) pair that already
                       appeared is not scored again. Repeated pairs always get the
                       same colour, so they are not independent coin flips.

    Returns a list with one entry per token: True (green), False (red) or
    None (not scored).
    """
    ids = [int(t) for t in token_ids]
    flags = []
    seen = set()
    prev = None if context_token_id is None else int(context_token_id)
    for tok in ids:
        if prev is None:
            flags.append(None)
        elif ignore_repeated_bigrams and (prev, tok) in seen:
            flags.append(None)
        else:
            seen.add((prev, tok))
            flags.append(is_green(prev, tok, vocab_size, gamma, key))
        prev = tok
    return flags


def detect(token_ids, vocab_size: int, context_token_id=None,
           gamma: float = GAMMA, key: int = SECRET_KEY,
           ignore_repeated_bigrams: bool = False) -> dict:
    """The hypothesis test.

    H0: the text was written without knowledge of the key, so each scored
        token is green with probability gamma and G ~ Binomial(T, gamma).
    H1: the text is watermarked (green more often than gamma). One-sided.
    """
    flags = green_flags(token_ids, vocab_size, context_token_id, gamma, key,
                        ignore_repeated_bigrams)
    scored = [f for f in flags if f is not None]
    T = len(scored)
    G = int(sum(scored))
    if T == 0:
        return dict(T=0, G=0, green_frac=np.nan, z=np.nan, p_z=np.nan, p_exact=np.nan)
    z = (G - gamma * T) / np.sqrt(T * gamma * (1 - gamma))
    return dict(
        T=T,
        G=G,
        green_frac=G / T,
        z=float(z),
        p_z=float(stats.norm.sf(z)),                    # normal approximation
        p_exact=float(stats.binom.sf(G - 1, T, gamma)), # exact P(G' >= G) under H0
    )
