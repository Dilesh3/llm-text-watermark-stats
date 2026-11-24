# Generation: human texts and watermarked GPT-2 texts

Everything needed to build the dataset that the analysis in `../analysis/` works on.

## What's in this folder

| File | What it is |
|---|---|
| `Watermark_GPT2_Colab.ipynb` | **Start here.** Generates the GPT-2 texts, scores every text, runs the sanity checks and draws the three `fig_*_gpt2.png` figures |
| `data/human_texts.csv` | 498 human texts from the Brown corpus (1961): `text_id, genre, group, prompt, human_text` |
| `watermark_core.py` | The shared green-list function and the detector (z-test and exact binomial test). **Generator and detector both use this file** |
| `gpt2_watermark.py` | GPT-2 watermark (Hugging Face logits processor), batch generation, text-only detection |
| `figures.py` | Coloured-text figure and probability bar chart |
| `sanity_checks.py` | Checks a results file and writes a short report |
| `chapter2_slides.md` | Early slide text explaining the watermark |
| `make_human_texts.py` | Builds `data/human_texts.csv` from the Brown corpus (NLTK) |
| `build_notebook.py` | Builds the Colab notebook from the Python files |
| `prototype/` | The same pipeline with a small trigram model instead of GPT-2 (see below) |
| `tests/` | Checks of the GPT-2 code with a tiny, randomly initialised GPT-2 |

## How the human texts are cut

Each Brown passage (about 2,000 words) is cut into whole sentences, in order:

1. The first ~1,000 words are set aside. They were only used to train the prototype model.
2. The next sentences, at least 30 tokens, become the **prompt** given to GPT-2.
3. The sentences after that, at least 260 tokens, are the **human continuation**: what the author really wrote next.

Passages whose continuation is shorter than 260 tokens, or whose prompt is longer than 120, are dropped, leaving 498 texts.

## How to produce the GPT-2 dataset (about 15 minutes)

1. Open `Watermark_GPT2_Colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. When asked, upload `data/human_texts.csv`.
4. A zip downloads at the end: `results_gpt2.csv`, the sanity report and three figures.

For a quick trial, set `N_PROMPTS = 60` in the Settings cell first. Our full run is in `../results/`.

Settings used: GPT-2 small, γ = 0.5, δ ∈ {0, 1, 2}, plain sampling (temperature 1, top_k = 0, top_p = 1), at least 200 new tokens, batch size 16, seed = 10000·δ + batch start.

## The results table (one row per text)

`text_id, genre, group, source (human/ai), delta, T, G, green_frac, z, p_z, p_exact, T_dedup, G_dedup, z_dedup, context_id, token_ids, text`

- `T` = tokens scored, `G` = green tokens, `z` = z-score, `p_z` = normal-approximation p-value, `p_exact` = exact binomial p-value
- `*_dedup` = the same test counting each repeated (previous token, token) pair only once
- Every text is scored **from its text only** (re-tokenized), exactly as a real detector would
- `delta` is empty for human texts; 0 = AI without watermark (control), 1 = weak, 2 = strong

## The prototype

`prototype/results_prototype.csv` has the same columns, made with a small word-level trigram language model trained on the set-aside first ~1,000 words of each Brown passage. Its AI text is not fluent, but the statistics behave the same way, so the analysis code was written and tested on it while the GPT-2 run was being prepared. None of the reported numbers come from it.

Prototype sanity check (498 texts per row, 200 tokens each):

| | green % | mean z | flagged at z > 1.645 | flagged at z > 4 |
|---|---|---|---|---|
| Human | 51.0 | 0.27 | 11.4% | 0% |
| AI, δ = 0 | 51.0 | 0.27 | 6.4% | 0% |
| AI, δ = 1 | 71.3 | 6.03 | 100% | 99.0% |
| AI, δ = 2 | 85.6 | 10.08 | 100% | 100% |

The prototype was where we first saw human text being flagged more often than 5% at α = 0.05, which led to the H0 checks on the GPT-2 data.

## Tests

`tests/test_gpt2_code.py` and `tests/run_notebook_with_standin.py` run the GPT-2 code and the whole notebook with a tiny randomly initialised GPT-2 and a local tokenizer (needs `torch`, `transformers`, `tokenizers`). With a near-uniform model, theory predicts green rates of 50%, 73.1% and 88.1% for δ = 0, 1, 2; the code gives 49.2%, 73.1% and 88.2%.

## Credits

- Method: Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.
- Human text: the Brown corpus (Francis & Kučera), distributed with NLTK.
- Model: GPT-2 small (OpenAI), via Hugging Face `transformers`.
