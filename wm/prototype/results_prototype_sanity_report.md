# Sanity checks for results_prototype.csv

1992 texts. Lengths: min T = 200, max T = 200.

| setting       |   texts |   mean T |   mean green % |   mean z |   SD of z | flagged at z > 1.645   |   p (flag rate = 5%) | flagged at z > 4   |   mean z (dedup) | flagged at z > 1.645 (dedup)   |
|:--------------|--------:|---------:|---------------:|---------:|----------:|:-----------------------|---------------------:|:-------------------|-----------------:|:-------------------------------|
| human         |     498 |      200 |           51   |     0.27 |      1.09 | 11.4%                  |              1.1e-08 | 0.0%               |             0.13 | 6.8%                           |
| AI, delta = 0 |     498 |      200 |           51   |     0.27 |      0.92 | 6.4%                   |              0.15    | 0.0%               |             0.21 | 4.6%                           |
| AI, delta = 1 |     498 |      200 |           71.3 |     6.03 |      0.93 | 100.0%                 |                      | 99.0%              |             5.82 | 100.0%                         |
| AI, delta = 2 |     498 |      200 |           85.6 |    10.08 |      0.72 | 100.0%                 |                      | 100.0%             |             9.82 | 100.0%                         |

Expected: human and delta = 0 rows near 50% green, mean z near 0, SD near 1, about 5% flagged at z > 1.645 and none at z > 4; delta = 1 and 2 rows far greener. 'dedup' columns count each repeated (previous token, token) pair once.
