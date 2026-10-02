"""Power for the primary contrasts, by simulation with Fisher's exact test.

Pilot (protocol 1) press rates on all trials were 9% to 28%. For each baseline rate and
sample size, report the smallest increase detected with 80% power at the Holm worst-case
alpha of 0.025 (two contrasts).
"""
import numpy as np
from scipy.stats import fisher_exact

rng = np.random.default_rng(0)


def power(p0, p1, n, alpha=0.025, sims=2000):
    hits = 0
    for _ in range(sims):
        a, b = rng.binomial(n, p1), rng.binomial(n, p0)
        hits += fisher_exact([[a, n - a], [b, n - b]])[1] < alpha
    return hits / sims


for n in (100, 200, 300):
    for p0 in (0.10, 0.15, 0.25):
        d = 0.02
        while power(p0, p0 + d, n) < 0.8:
            d += 0.01
        print(f"n={n} per arm, baseline {p0:.0%}: detects +{d:.0%} (to {p0 + d:.0%}) with 80% power")
