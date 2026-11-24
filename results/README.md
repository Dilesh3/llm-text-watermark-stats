# Results on GPT-2

Outputs of the analysis on the 1,992 GPT-2 and human texts (team key, γ = 0.5). See the main README for the commands that produce them.

| File | What it holds |
|---|---|
| `results_gpt2.csv` | One row per text (498 human + 3 × 498 GPT-2): green count, z, p-values, token ids and the text itself. Made by the Colab notebook |
| `results_gpt2_sanity_report.md` | Green share, mean z and flag rates per group |
| `analysis_table_gpt2.csv` | The shared analysis table: 7,968 rows (1,992 texts × lengths 25, 50, 100, 200) |
| `pilot_gpt2.txt` | Green share per δ, expected power by length, p-chart window sizes |
| `chapter_tests_gpt2_raw.txt` | The test, errors and power, extra H0 checks and the regression, counting every token (the numbers in the slides) |
| `chapter_tests_gpt2.txt` | The same, counting each repeated (previous token, token) pair once per text |
| `chapter5_keys_gpt2_report.txt` | Goodness of fit, false-alarm test, 20-key study and calibration |
| `chapter5_keys_gpt2.csv` | False-alarm rate per key: every token, repeats counted once, calibrated |
| `pchart_gpt2_d2.csv`, `pchart_gpt2_d2.png` | p-chart windows for 50 documents, δ = 2, window 50 |
| `pchart_gpt2_d1.png`, `pchart_gpt2_d1_w100.png` | The same for δ = 1 with windows of 50 and 100 tokens |

## Columns of `analysis_table_gpt2.csv`

`text_id, genre, group, source, delta, length, T, G, green_share, z, p_z, p_exact, flag_a05, flag_z4, T_raw, G_raw, z_raw, flag_a05_raw`

- `source` is human or ai; `delta` is empty for human texts and 0 for GPT-2 without the watermark
- `length` is the number of leading tokens scored. GPT-2 output often re-tokenizes to 195–199 tokens; those texts are still scored at length 200
- `T, G, z, ...` count each repeated (previous token, token) pair once; the `*_raw` columns count every token, which is the textbook test used in the slides
- `flag_a05` means z > 1.645 and `flag_z4` means z > 4
