# Sanity checks for results_gpt2.csv

1992 texts. Lengths: min T = 195, max T = 200.

| setting       |   texts |   mean T |   mean green % |   mean z |   SD of z | flagged at z > 1.645   | p (flag rate = 5%)   | flagged at z > 4   |   mean z (dedup) | flagged at z > 1.645 (dedup)   |
|:--------------|--------:|---------:|---------------:|---------:|----------:|:-----------------------|:---------------------|:-------------------|-----------------:|:-------------------------------|
| human         |     498 |    200   |           50.7 |     0.19 |      1.07 | 9.4%                   | 4.8e-05              | 0.0%               |             0.11 | 7.0%                           |
| AI, delta = 0 |     498 |    199.6 |           50.3 |     0.07 |      1    | 5.8%                   | 0.41                 | 0.0%               |             0.01 | 5.0%                           |
| AI, delta = 1 |     498 |    199.7 |           70.1 |     5.69 |      0.98 | 100.0%                 |                      | 95.0%              |             5.48 | 100.0%                         |
| AI, delta = 2 |     498 |    199.6 |           84.1 |     9.63 |      0.82 | 100.0%                 |                      | 100.0%             |             9.36 | 100.0%                         |

Expected: human and delta = 0 rows near 50% green, mean z near 0, SD near 1, about 5% flagged at z > 1.645 and none at z > 4; delta = 1 and 2 rows far greener. 'dedup' columns count each repeated (previous token, token) pair once.
