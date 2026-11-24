"""Smoke test of gpt2_watermark.py without downloading GPT-2.

Uses a tiny randomly initialised GPT-2 and a word-level tokenizer built locally.
A random untrained model gives almost uniform next-token probabilities, so with
gamma = 0.5 theory says P(green) = e^delta / (1 + e^delta):
  delta = 0 -> 0.500, delta = 1 -> 0.731, delta = 2 -> 0.881
"""
import os, sys, math
WM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the wm/ folder
sys.path.insert(0, WM)
import numpy as np, pandas as pd, torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from gpt2_watermark import generate_batch, detect_text, next_token_distribution
from watermark_core import detect

d = pd.read_csv(os.path.join(WM, "data", "human_texts.csv"))
tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tk.decoder = decoders.WordPiece(prefix="##")  # joins with spaces
tk.train_from_iterator(list(d.prompt) + list(d.human_text),
                       trainers.WordLevelTrainer(vocab_size=8000, special_tokens=["[UNK]", "[EOS]"]))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="[EOS]")
V = tok.vocab_size
torch.manual_seed(0)
model = GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=512, n_embd=64, n_layer=2, n_head=2,
                                   eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)).eval()

prompts = list(d.prompt[:24])
for delta, theory in [(0, 0.5), (1, math.e / (1 + math.e)), (2, math.e**2 / (1 + math.e**2))]:
    outs = generate_batch(model, tok, prompts, delta=delta, n_tokens=150, seed=1)
    assert all(len(o["gen_ids"]) == 150 for o in outs), "fixed length violated"
    # detection from the exact generated ids (context = last real prompt token)
    G = T = 0
    for p, o in zip(prompts, outs):
        r = detect(o["gen_ids"], V, context_token_id=tok(p)["input_ids"][-1])
        G += r["G"]; T += r["T"]
    # detection from text only (re-tokenized), as a real detector would
    G2 = T2 = 0
    for p, o in zip(prompts, outs):
        r = detect_text(o["gen_text"], p, tok, V, n_tokens=150)
        G2 += r["G"]; T2 += r["T"]
    print(f"delta={delta}: green from ids {G/T:.3f}, from text {G2/T2:.3f}, theory {theory:.3f}  (T={T})")

# human text should be ~50% green
G = T = 0
for p, h in zip(d.prompt[:100], d.human_text[:100]):
    r = detect_text(h, p, tok, V, n_tokens=150); G += r["G"]; T += r["T"]
print(f"human text green rate {G/T:.3f} (T={T})")

rows, gb, ga = next_token_distribution(model, tok, prompts[0], delta=2)
print("bar-chart data ok:", len(rows), round(gb, 3), round(ga, 3))
