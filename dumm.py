import pandas as pd
import numpy as np

def adaptive_pps(df, size_col, start=20, batch=10, max_n=70, target_rel_moe=0.05):
    # df must have claim amounts in size_col
    N = len(df)
    total = df[size_col].sum()
    n = start
    sampled_idx = []

    while n <= max_n:
        # PPS sample
        probs = df[size_col] / df[size_col].sum()
        sel = np.random.choice(df.index, size=n, replace=False, p=probs)
        sample = df.loc[sel]

        # Horvitz-Thompson weights
        pi = probs[sel] * n
        weights = 1 / pi

        # Estimate mean and variance
        est_mean = np.sum(weights * sample[size_col]) / N
        var_est = np.var(weights * sample[size_col]) / n  # rough approx
        se = np.sqrt(var_est)
        moe = 1.96 * se
        rel_moe = moe / est_mean

        print(f"n={n}, mean={est_mean:.1f}, MoE={moe:.1f}, RelMoE={rel_moe:.2%}")

        if rel_moe <= target_rel_moe:
            print("Stopping criterion met")
            return sample, n, est_mean, moe

        n += batch

    print("Hit max sample size without meeting target")
    return sample, n, est_mean, moe
