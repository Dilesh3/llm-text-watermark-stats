"""Builds Watermark_GPT2_Colab.ipynb from the project's .py files,
so the notebook always contains exactly the tested code."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def src(name):
    with open(os.path.join(HERE, name)) as fh:
        return fh.read()


cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=text.strip("\n")))


md("""
# Watermarked text with GPT-2: generation, detection and Chapter 2 figures

This notebook produces the project's main dataset.

1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
2. When asked, upload **`human_texts.csv`** (from the project folder).
3. At the end, a zip file with the results and figures downloads automatically.

It runs in about 10–20 minutes on a free Colab GPU for all 498 prompts.
Settings are in the *Settings* cell; keep them unchanged unless the team agrees, so all numbers match.
""")

md("## 1. Project code\nThese cells write the shared code files. They are identical to the `.py` files in the project folder.")
for f in ["watermark_core.py", "gpt2_watermark.py", "figures.py", "sanity_checks.py"]:
    code(f"%%writefile {f}\n" + src(f))

md("## 2. Upload the human texts")
code('''
# [COLAB-ONLY]
import os
if not os.path.exists("human_texts.csv"):
    from google.colab import files
    files.upload()          # choose human_texts.csv
''')

md("## 3. Settings")
code('''
N_PROMPTS = 498        # all Brown texts; use e.g. 60 for a quick test run
N_TOKENS = 200         # length of every text, in GPT-2 tokens
DELTAS = [0, 1, 2]     # 0 = ordinary AI text (control), 1 = weak, 2 = standard watermark
BATCH = 16
from watermark_core import GAMMA, SECRET_KEY
print("gamma =", GAMMA, "| key =", SECRET_KEY)
''')

md("## 4. Load GPT-2")
code('''
# [MODEL-LOAD]
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
V = model.config.vocab_size
print("vocabulary size:", V)
''')

md("## 5. Generate the AI texts\nEach prompt is continued three times: δ = 0, 1 and 2. Fixed seeds make the run reproducible.")
code('''
import pandas as pd
from tqdm.auto import tqdm
from gpt2_watermark import generate_batch

human = pd.read_csv("human_texts.csv").head(N_PROMPTS)
generated = []
for delta in DELTAS:
    for start in tqdm(range(0, len(human), BATCH), desc=f"delta = {delta}"):
        chunk = human.iloc[start:start + BATCH]
        outs = generate_batch(model, tokenizer, list(chunk.prompt), delta=delta,
                              n_tokens=N_TOKENS, seed=10_000 * delta + start)
        for (_, r), o in zip(chunk.iterrows(), outs):
            generated.append(dict(text_id=r.text_id, delta=delta, text=o["gen_text"]))
print(len(generated), "AI texts generated")
''')

md("## 6. Detect: score every text (human and AI) from its text only")
code('''
import csv
import os
from gpt2_watermark import detect_text

def score(text_id, genre, group, source, delta, prompt, text):
    r = detect_text(text, prompt, tokenizer, V, n_tokens=N_TOKENS)
    rd = detect_text(text, prompt, tokenizer, V, n_tokens=N_TOKENS, ignore_repeated_bigrams=True)
    return dict(text_id=text_id, genre=genre, group=group, source=source, delta=delta,
                T=r["T"], G=r["G"], green_frac=round(r["green_frac"], 4), z=round(r["z"], 4),
                p_z=r["p_z"], p_exact=r["p_exact"],
                T_dedup=rd["T"], G_dedup=rd["G"], z_dedup=round(rd["z"], 4),
                context_id=r["context_id"], token_ids=" ".join(map(str, r["token_ids"])), text=text)

info = human.set_index("text_id")
rows = []
for r in tqdm(human.itertuples(), total=len(human), desc="human"):
    rows.append(score(r.text_id, r.genre, r.group, "human", float("nan"), r.prompt, " " + r.human_text))
for g in tqdm(generated, desc="AI"):
    h = info.loc[g["text_id"]]
    rows.append(score(g["text_id"], h.genre, h.group, "ai", g["delta"], h.prompt, g["text"]))
results = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/results_gpt2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
print(results.groupby(["source", "delta"], dropna=False)[["green_frac", "z"]].mean().round(3))
''')

md("## 7. Sanity checks\nExpected: human and δ = 0 texts about 50% green with mean z near 0; δ = 1 and δ = 2 clearly greener.")
code('''
import sanity_checks
sanity_checks.main("outputs/results_gpt2.csv")
''')

md("## 8. Chapter 2 figures")
code('''
from IPython.display import Image, display
from figures import coloured_text_figure, probability_bars
from gpt2_watermark import next_token_distribution
from watermark_core import green_flags, detect

EXAMPLE_ID = "ca05" if "ca05" in set(results.text_id) else results.text_id.iloc[0]

def panel(source, delta, title):
    sel = results[(results.text_id == EXAMPLE_ID) & (results.source == source)]
    if delta is not None:
        sel = sel[sel.delta == delta]
    r = sel.iloc[0]
    ids = [int(t) for t in r.token_ids.split()]
    return dict(title=title,
                tokens=[tokenizer.decode([i]) for i in ids],
                flags=green_flags(ids, V, context_token_id=int(r.context_id)),
                stats=detect(ids, V, context_token_id=int(r.context_id)))

coloured_text_figure([panel("human", None, "Human (Brown corpus, 1961)"),
                      panel("ai", 2, "GPT-2 with watermark (\\u03b4 = 2)")],
                     "outputs/fig_coloured_text_gpt2.png")
coloured_text_figure([panel("ai", 0, "GPT-2, no watermark (\\u03b4 = 0)"),
                      panel("ai", 2, "GPT-2 with watermark (\\u03b4 = 2)")],
                     "outputs/fig_coloured_text_gpt2_ai_vs_ai.png")

CONTEXT = "The jury said it did not find any evidence of"
rows_, g_before, g_after = next_token_distribution(model, tokenizer, CONTEXT, delta=2)
probability_bars(rows_, g_before, g_after, CONTEXT, 2, "outputs/fig_probability_bars_gpt2.png")
for f in ["fig_coloured_text_gpt2.png", "fig_coloured_text_gpt2_ai_vs_ai.png", "fig_probability_bars_gpt2.png"]:
    display(Image(f"outputs/{f}"))
''')

md("## 9. Download everything")
code('''
# [COLAB-ONLY]
import shutil
from google.colab import files
shutil.make_archive("watermark_gpt2_outputs", "zip", "outputs")
files.download("watermark_gpt2_outputs.zip")
''')

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[]),
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(HERE, "Watermark_GPT2_Colab.ipynb"), "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"notebook written with {len(cells)} cells")
