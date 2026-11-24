"""
gpt2_watermark.py - watermarked text generation with GPT-2 (Hugging Face).

Needs: torch, transformers, numpy, scipy  (all preinstalled on Google Colab).
Uses watermark_core.py for the green lists, so generation and detection
always agree.
"""

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from watermark_core import GAMMA, SECRET_KEY, detect, green_mask


class WatermarkLogitsProcessor(LogitsProcessor):
    """Adds delta to the score (logit) of every GREEN token before sampling.

    Hugging Face calls this once per generation step with
      input_ids : (batch, tokens so far)
      scores    : (batch, vocab_size) raw scores for the next token
    """

    def __init__(self, vocab_size: int, gamma: float = GAMMA, delta: float = 2.0,
                 key: int = SECRET_KEY):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.key = key

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.delta == 0:
            return scores
        if scores.shape[-1] != self.vocab_size:
            raise ValueError(f"scores have {scores.shape[-1]} columns, expected {self.vocab_size}")
        for b in range(input_ids.shape[0]):
            prev = int(input_ids[b, -1])
            mask = torch.from_numpy(np.array(green_mask(prev, self.vocab_size, self.gamma, self.key)))
            scores[b, mask.to(scores.device)] += self.delta
        return scores


def generate_batch(model, tokenizer, prompts, delta, n_tokens=200, gamma=GAMMA,
                   key=SECRET_KEY, seed=0, device=None):
    """Continue each prompt by exactly n_tokens tokens.

    Plain multinomial sampling (temperature 1, no top-k / top-p), as in the
    watermark paper. delta = 0 gives ordinary, unwatermarked AI text.
    Returns a list of dicts with the generated token ids and text.
    """
    device = device or next(model.parameters()).device
    vocab_size = model.config.vocab_size
    tokenizer.padding_side = "left"              # GPT-2 must be left-padded for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,             # forbid early end-of-text: fixed length
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList(
                [WatermarkLogitsProcessor(vocab_size, gamma, delta, key)]),
        )
    new_tokens = out[:, enc["input_ids"].shape[1]:].cpu().tolist()
    return [dict(gen_ids=ids, gen_text=tokenizer.decode(ids)) for ids in new_tokens]


def detect_text(text, prompt, tokenizer, vocab_size, n_tokens=200, gamma=GAMMA,
                key=SECRET_KEY, ignore_repeated_bigrams=False):
    """What a real detector does: it only sees TEXT.

    Re-tokenizes the text, keeps its first n_tokens tokens and uses the last
    prompt token as the context for the first one.
    """
    prompt_ids = tokenizer(prompt)["input_ids"]
    ids = tokenizer(text)["input_ids"][:n_tokens]
    res = detect(ids, vocab_size, context_token_id=prompt_ids[-1], gamma=gamma,
                 key=key, ignore_repeated_bigrams=ignore_repeated_bigrams)
    res["token_ids"] = ids
    res["context_id"] = prompt_ids[-1]
    return res


def next_token_distribution(model, tokenizer, context, delta=2.0, gamma=GAMMA,
                            key=SECRET_KEY, top=10):
    """Probabilities of the top next tokens before and after the watermark boost
    (for the Chapter 2 bar chart)."""
    device = next(model.parameters()).device
    ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float().cpu().numpy()
    mask = np.array(green_mask(int(ids[0, -1]), model.config.vocab_size, gamma, key))
    boosted = logits + delta * mask
    p_before = _softmax(logits)
    p_after = _softmax(boosted)
    order = np.argsort(-p_before)[:top]
    rows = [dict(token=tokenizer.decode([int(i)]), green=bool(mask[i]),
                 p_before=float(p_before[i]), p_after=float(p_after[i])) for i in order]
    return rows, float(p_before[mask].sum()), float(p_after[mask].sum())


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
