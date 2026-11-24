# Chapter 2 – The watermark: slide content

Numbers marked **[GPT-2]** come from the Colab run (`outputs/results_gpt2_sanity_report.md` and the figures). Replace them after running the notebook. The numbers in brackets are from the prototype, as a guide.

---

## Short deck (15-minute talk)

### Slide 4 – "The AI hides a secret pattern while it writes"

**Visual:** `fig_coloured_text_gpt2.png` (human vs watermarked, tokens shaded green/red)

On slide:
- Before each word, a secret key splits the vocabulary into green and red halves
- The AI is nudged towards green words
- Human: ~50% green · Watermarked AI: ~**[GPT-2]**% green (prototype: 86%)

Speaker notes: "Both texts read normally. But if you hold the key and colour each word, the watermarked text is overwhelmingly green, while the human text is a coin flip, about half and half."

### Slide 5 – "A small push towards green words"

**Visual:** `fig_probability_bars_gpt2.png`

On slide:
- GPT-2 scores all 50,257 possible next tokens
- We add δ = 2 to the scores of the green half, then sample as usual
- Chance of picking a green token at this step: **[GPT-2]**% → **[GPT-2]**%

Speaker notes: "The model still chooses words that make sense. The boost only tips close calls towards green. Across hundreds of words that small tilt becomes a strong statistical signal."

---

## Long deck (submission) – Chapter 2, 7 slides

### 2.1 How a language model writes
- At each step GPT-2 computes a score (logit) for every token in its vocabulary of 50,257 tokens
- Softmax turns the scores into probabilities: p = e^score / Σ e^score
- The next token is **sampled** from these probabilities (not always the most likely one)
- Visual: the grey bars of `fig_probability_bars_gpt2.png`

### 2.2 Making the green list – toy walkthrough
- The 4-word example from our notes: previous word "the" (ID 262), key K = 200
- seed = (K × ID) mod 101 = 82 → random numbers mat 86, sofa 53, floor 98, bed 55
- The half with the smallest numbers is green: {sofa, bed}
- Same inputs → same list (the detector can rebuild it); without the key the list looks random

### 2.3 Making the green list – the real version
| | Toy example | Our implementation |
|---|---|---|
| Vocabulary | 4 words | 50,257 GPT-2 tokens |
| Green tokens (γ = 0.5) | 2 | 25,128 |
| Seed | (K × ID) mod 101 | SHA-256 hash of key and previous token |
| Shuffle | mod-101 number chain | NumPy random permutation from the seed |

- Code: `green_mask()` in `watermark_core.py`, used by both the generator and the detector

### 2.4 The boost
- Add δ to the score of every green token, then softmax and sample
- Visual: `fig_probability_bars_gpt2.png` (grey = before, blue = after)
- At this step the chance of a green token rises from **[GPT-2]**% to **[GPT-2]**%
- Larger δ = stronger watermark but more distortion of the text (we use δ = 1 and δ = 2)

### 2.5 The result: watermarked vs normal text
- Visuals: `fig_coloured_text_gpt2.png` (human vs AI) and `fig_coloured_text_gpt2_ai_vs_ai.png` (same model, δ = 0 vs δ = 2)
- Green share: human **[GPT-2]**%, AI δ = 0 **[GPT-2]**%, AI δ = 1 **[GPT-2]**%, AI δ = 2 **[GPT-2]**%
  (prototype: 51%, 51%, 71%, 86%)
- Key point: the δ = 0 panel proves the pattern comes from the watermark, not from "AI style"

### 2.6 Our data
- **Human text:** 498 samples from the Brown corpus (American English published in 1961, 15 genres), guaranteed human-written since it predates AI text generators
- **Prompts:** the opening sentences of each Brown sample; GPT-2 continues the same prompt, so human and AI texts share topics
- **AI text:** GPT-2 small, plain sampling (temperature 1), exactly 200 tokens per text, δ ∈ {0, 1, 2}, fixed random seeds
- **Total:** 498 human + 1,494 AI texts, all scored by the same detector
- γ = 0.5, secret key fixed for the whole project

### 2.7 Code: the watermark in 10 lines
```python
class WatermarkLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            prev = int(input_ids[b, -1])                       # previous token
            mask = green_mask(prev, self.vocab_size,           # key + previous token
                              self.gamma, self.key)            #   -> green list
            scores[b, torch.from_numpy(mask)] += self.delta    # boost green tokens
        return scores                                          # then sample as usual
```
- Reference: Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023 (our implementation follows their "soft red list" method with the previous token as context)
