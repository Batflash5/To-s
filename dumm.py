import numpy as np
import pandas as pd
import random

def emergent_pps_sample(df, amount_col='amount', max_n=29, coverage_target=0.80, seed=None):
    """
    Emergent PPS sampling under a sample-size constraint (< max_n).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least [amount_col]. Will create 'orig_index' if missing.
    amount_col : str
        Column name with amounts (sizes).
    max_n : int
        Maximum allowed sample size (constraint).
    coverage_target : float
        Fraction of total amount S that must be covered before stopping.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with:
      'sampled_df'       DataFrame of sampled units
      'certainties_df'   DataFrame of certainty units (pi=1)
      'inclusion_probs'  Series of inclusion probs (indexed by orig_index)
      'ht_weights'       Series of Horvitz–Thompson weights = 1/pi
      'final_n'          Actual sample size
      'coverage'         Coverage fraction achieved
      'threshold'        Certainty threshold (S / max_n)
      'sampled_from_rem' List of indices sampled from remainder
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df = df.copy()
    if 'orig_index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'orig_index'})

    S = df[amount_col].sum()
    if S <= 0:
        raise ValueError("Total amount S must be positive.")

    # Certainty threshold
    threshold = S / max_n
    cert_mask = df[amount_col] >= threshold - 1e-12
    certainties_df = df[cert_mask].copy().set_index('orig_index')

    inclusion_probs = pd.Series(0.0, index=df['orig_index'], dtype=float)
    for idx in certainties_df.index:
        inclusion_probs.at[idx] = 1.0

    included_amount = certainties_df[amount_col].sum()
    coverage_cert = included_amount / S

    # Stop if certainties already meet coverage target or max_n
    if coverage_cert >= coverage_target or len(certainties_df) >= max_n:
        ht_weights = pd.Series(np.where(inclusion_probs > 0, 1.0/inclusion_probs, np.nan),
                               index=inclusion_probs.index)
        return {
            'sampled_df': certainties_df,
            'certainties_df': certainties_df,
            'inclusion_probs': inclusion_probs,
            'ht_weights': ht_weights,
            'final_n': len(certainties_df),
            'coverage': coverage_cert,
            'threshold': threshold,
            'sampled_from_rem': []
        }

    # Work with remainder
    remaining = df[~cert_mask].copy().reset_index(drop=True)
    S_rem = remaining[amount_col].sum()
    slots = max_n - len(certainties_df)
    step = S_rem / slots
    u = random.random() * step
    thresholds = np.array([u + k * step for k in range(slots)])
    cums = remaining[amount_col].cumsum().values
    idxs = np.searchsorted(cums, thresholds, side='right')
    idxs = np.clip(idxs, 0, len(remaining)-1)

    chosen = remaining.iloc[sorted(set(idxs))].copy().sort_values(amount_col, ascending=False)

    sampled_from_rem = []
    for _, row in chosen.iterrows():
        sampled_from_rem.append(row['orig_index'])
        included_amount += row[amount_col]
        if included_amount / S >= coverage_target:
            break

    sampled_idx = list(certainties_df.index) + sampled_from_rem
    n_rem_actual = len(sampled_from_rem)

    # Approximate pi for remainder
    if n_rem_actual > 0 and S_rem > 0:
        pi_rem_vals = (n_rem_actual * remaining[amount_col] / S_rem).clip(upper=1.0)
        for rk, val in zip(remaining['orig_index'], pi_rem_vals.values):
            inclusion_probs.at[rk] = float(val)

    ht_weights = pd.Series(np.where(inclusion_probs > 0, 1.0/inclusion_probs, np.nan),
                           index=inclusion_probs.index)
    sampled_df = df[df['orig_index'].isin(sampled_idx)].set_index('orig_index')
    coverage_final = included_amount / S

    return {
        'sampled_df': sampled_df,
        'certainties_df': certainties_df,
        'inclusion_probs': inclusion_probs,
        'ht_weights': ht_weights,
        'final_n': len(sampled_idx),
        'coverage': coverage_final,
        'threshold': threshold,
        'sampled_from_rem': sampled_from_rem
    }

# ================================
# Example usage (function calling)
# ================================

# Suppose you already have a DataFrame `df` with an 'amount' column:
# df = pd.read_csv("your_population.csv")

res = emergent_pps_sample(df, amount_col='amount', max_n=29, coverage_target=0.80, seed=2025)

# ================================
# Output interpretation
# ================================
print("Certainty threshold (amount >=):", res['threshold'])
print("Final sample size:", res['final_n'])
print("Coverage achieved:", round(res['coverage']*100, 2), "%")

print("\nSampled rows (by amount):")
print(res['sampled_df'].sort_values('amount', ascending=False)[['id','amount']])

# Build full output with probs + weights
inclusion_probs = res['inclusion_probs'].reindex(df['orig_index'])
ht_weights = res['ht_weights'].reindex(df['orig_index'])
df_out = df.copy()
df_out['inclusion_prob'] = inclusion_probs.values
df_out['ht_weight'] = ht_weights.values
df_out['is_sampled'] = df['orig_index'].isin(res['sampled_df'].index)

# Save to CSV
df_out.to_csv("emergent_pps_output.csv", index=False)
print("\nSaved full output with inclusion probs & weights to emergent_pps_output.csv")
