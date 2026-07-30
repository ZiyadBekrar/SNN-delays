# Significance of the learned-versus-fixed difference

Difference in mean accuracy (learned - fixed), in accuracy points, and the
two-sided Welch t-test p-value for unequal variances.

| Dataset | Delay type | d acc. [pts] | t     | p      | n |
|---------|------------|--------------|-------|--------|---|
| SSC     | Axonal     | 0.31         | 2.479 | 0.0414 | 5 |
| SSC     | Synaptic   | 0.15         | 1.158 | 0.2805 | 5 |
| PSMNIST | Axonal     | 0.55         | 3.695 | 0.0073 | 5 |
| PSMNIST | Synaptic   | 0.73         | 6.605 | 0.0005 | 5 |
| AL      | Axonal     | 3.02         | 3.065 | 0.0156 | 5 |
| AL      | Synaptic   | 1.27         | 1.885 | 0.1009 | 5 |
| HAR     | Axonal     | 0.79         | 3.047 | 0.0446 | 3 |
| HAR     | Synaptic   | 1.55         | 7.05  | 0.0021 | 3 |
