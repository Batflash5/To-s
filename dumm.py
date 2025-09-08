import numpy as np
import pandas as pd

def transformed_size(df, amount_col='amount', transform='sqrt'):
    """Return array of transformed sizes s_i."""
    if transform == 'sqrt':
        return np.sqrt(df[amount_col].values.astype(float))
    elif transform == 'log':
        return np.log1p(df[amount_col].values.astype(float))
    elif transform is None or transform == 'raw':
        return df[amount_col].values.astype(float)
    else:
        raise ValueError("transform must be 'sqrt', 'log', 'raw' or None")

# ---------- Linear mapping: pi_i = min(1, c * s_i / s_max) ----------
def linear_pi(df, amount_col='amount', transform='sqrt', c=1.0):
    """
    Compute pi_i using linear scaling (no fixed n).
    - df: DataFrame
    - c: scaling constant (>=0). Larger c => larger pi's.
    Returns DataFrame copy with columns: pi_linear, weight_linear.
    """
    df2 = df.copy().reset_index(drop=False).rename(columns={'index':'_orig_index'})
    s = transformed_size(df2, amount_col, transform)
    smax = s.max() if s.size>0 else 1.0
    # avoid division by zero
    if smax <= 0:
        pi = np.zeros_like(s)
    else:
        pi = np.minimum(1.0, c * s / smax)
    df2['pi_linear'] = pi
    # avoid infinite weights
    df2['weight_linear'] = np.where(pi>0, 1.0/pi, np.nan)
    return df2

def choose_c_by_capture_linear(df, amount_col='amount', transform='sqrt', target_fraction=0.5, c_lo=1e-6, c_hi=1000, tol=1e-6, max_iter=60):
    """
    Find c so that expected captured amount >= target_fraction * total_amount,
    where expected captured amount = sum(amount_i * pi_i(c)) and pi_i uses linear mapping.
    Uses bisection on c in [c_lo, c_hi].
    """
    total = df[amount_col].sum()
    if total == 0:
        return c_lo
    def captured(c):
        tmp = linear_pi(df, amount_col, transform, c)
        return (df[amount_col].values * tmp['pi_linear'].values).sum()
    lo, hi = c_lo, c_hi
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if captured(mid) >= target_fraction * total:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5*(lo+hi)

# ---------- Exponential (Poisson-style) mapping: pi_i = 1 - exp(-c * s_i / S) ----------
def exp_pi(df, amount_col='amount', transform='sqrt', c=1.0):
    """
    Compute pi_i using exponential mapping.
    - S = sum(s_i)
    - pi_i = 1 - exp(-c * s_i / S)
    Returns DataFrame copy with columns: pi_exp, weight_exp.
    """
    df2 = df.copy().reset_index(drop=False).rename(columns={'index':'_orig_index'})
    s = transformed_size(df2, amount_col, transform)
    S = s.sum() if s.size>0 else 1.0
    if S <= 0:
        pi = np.zeros_like(s)
    else:
        lam = c * s / S
        # numerical safety
        lam = np.clip(lam, 0.0, 700.0)  # avoid overflow in exp
        pi = 1.0 - np.exp(-lam)
    df2['pi_exp'] = pi
    df2['weight_exp'] = np.where(pi>0, 1.0/pi, np.nan)
    return df2

def choose_c_by_capture_exp(df, amount_col='amount', transform='sqrt', target_fraction=0.5, c_lo=1e-8, c_hi=100.0, tol=1e-6, max_iter=60):
    """
    Find c by bisection such that expected captured amount (sum amount_i * pi_i(c))
    >= target_fraction * total_amount. Uses exp_pi mapping.
    """
    total = df[amount_col].sum()
    if total == 0:
        return c_lo
    def captured(c):
        tmp = exp_pi(df, amount_col, transform, c)
        return (df[amount_col].values * tmp['pi_exp'].values).sum()
    lo, hi = c_lo, c_hi
    # Expand hi until captured(hi) >= target (to ensure bracket), but limit expansions
    for _ in range(40):
        if captured(hi) >= target_fraction * total:
            break
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if captured(mid) >= target_fraction * total:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5*(lo+hi)

# ---------- Sampling draw (independent draws using pi) ----------
def draw_independent(df_with_pi, pi_col='pi_exp', random_state=None):
    """
    Given a dataframe that contains a column with per-unit inclusion probabilities (pi_col),
    perform independent Bernoulli draws and return sampled rows with their pi and weight.
    """
    rng = np.random.default_rng(random_state)
    df2 = df_with_pi.copy().reset_index(drop=True)
    pi = df2[pi_col].values.astype(float)
    u = rng.random(size=len(pi))
    selected_mask = (u < pi)
    sampled = df2.loc[selected_mask].copy().reset_index(drop=True)
    sampled['selected'] = True
    sampled['weight'] = np.where(sampled[pi_col]>0, 1.0/sampled[pi_col], np.nan)
    return sampled, df2

# ---------- Example usage ----------
# 1) Compute exponentials, choose c for 50% expected dollar capture, then draw:
# c = choose_c_by_capture_exp(df, amount_col='amount', transform='sqrt', target_fraction=0.50)
# frame = exp_pi(df, amount_col='amount', transform='sqrt', c=c)
# sampled, frame_with_pi = draw_independent(frame, pi_col='pi_exp', random_state=1)
#
# 2) Or linear mapping: choose c via choose_c_by_capture_linear(...) then same draw with 'pi_linear'.
