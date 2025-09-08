import numpy as np
import pandas as pd

def transformed_pps_sample(df, amount_col='amount', transform='sqrt',
                           exclude_bottom_pct=None, exclude_cutoff=None,
                           n_draw=50, random_state=0):
    """
    Returns sampled rows with pi_approx and weights using transformed-PPS.
    - transform: 'sqrt' or 'log' or None (raw)
    - exclude_bottom_pct: fraction (0-1) to drop lowest amounts (e.g., 0.10 to drop bottom 10%)
    - exclude_cutoff: alternative absolute cutoff (amount < cutoff dropped). If both provided, cutoff used.
    - n_draw: number of probabilistic draws (after exclusion)
    """
    rng = np.random.default_rng(random_state)
    df = df.copy().reset_index(drop=False).rename(columns={'index':'_orig_index'})
    # optional exclusion
    if exclude_cutoff is not None:
        df = df[df[amount_col] >= exclude_cutoff].reset_index(drop=True)
    elif exclude_bottom_pct is not None:
        cutoff = df[amount_col].quantile(exclude_bottom_pct)
        df = df[df[amount_col] >= cutoff].reset_index(drop=True)
    # compute transformed size
    if transform == 'sqrt':
        size = np.sqrt(df[amount_col].values)
    elif transform == 'log':
        size = np.log1p(df[amount_col].values)
    elif transform is None:
        size = df[amount_col].values.astype(float)
    else:
        raise ValueError("transform must be 'sqrt', 'log', or None")
    # single-draw probability
    total_size = size.sum()
    p_single = size / total_size
    # draw without replacement using p_single
    n_draw = min(n_draw, len(df))
    selected_idx = rng.choice(df.index.values, size=n_draw, replace=False, p=p_single)
    sample = df.loc[selected_idx].copy().reset_index(drop=True)
    p_i = p_single[selected_idx]
    # approximate inclusion prob for n_draw draws:
    pi_approx = np.clip(1.0 - (1.0 - p_i) ** n_draw, 1e-12, 1.0)
    sample['pi_approx'] = pi_approx
    sample['weight'] = 1.0 / sample['pi_approx']
    return sample, df  # sample + sampling frame used

# Example usage:
# sampled, frame_used = transformed_pps_sample(my_df, transform='sqrt', exclude_bottom_pct=0.10, n_draw=50)
