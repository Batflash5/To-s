import numpy as np
import pandas as pd

def adaptive_pps(df: pd.DataFrame,
                 size_col: str = "amount",
                 start: int = 20,
                 batch: int = 10,
                 max_n: int = 70,
                 target_rel_moe: float = 0.05,
                 n_bootstrap: int = 100,
                 random_state: int = 42,
                 force_two_stages: bool = False):
    """
    Adaptive PPS sampling with bootstrap SE estimation and certainty handling.
    Returns sampled dataframe (certainties + final sample) and stage stats.
    """
    rng = np.random.default_rng(random_state)
    N = len(df)
    total_size = df[size_col].sum()
    df = df.copy().reset_index(drop=False).rename(columns={"index": "_orig_index"})
    df["_share"] = df[size_col] / total_size
    
    # Identify certainties conservatively: share >= 1/max_n
    strict_threshold = 1.0 / max_n
    certainties = df[df["_share"] >= strict_threshold].copy()
    rest = df[df["_share"] < strict_threshold].copy()
    
    stats = {}
    if len(rest) == 0:
        sampled = certainties.copy()
        sampled["pi_approx"] = 1.0
        sampled["weight"] = 1.0
        stats[0] = {"n": len(sampled), "mu_hat": float(sampled[size_col].mean()), "se": 0.0, "moe": 0.0, "rel_moe": 0.0}
        return sampled, stats
    
    rest_total = rest[size_col].sum()
    rest["p_single"] = rest[size_col] / rest_total
    rest_idx = rest.index.values
    
    n = start
    stage = 0
    last_sample = None
    min_stages = 2 if force_two_stages else 1
    
    while n <= max_n:
        stage += 1
        n_draw = min(n, len(rest))
        selected = rng.choice(rest_idx, size=n_draw, replace=False, p=rest["p_single"].values)
        sample = rest.loc[selected].copy().reset_index(drop=True)
        
        p_i = sample["p_single"].values
        pi_approx = 1.0 - (1.0 - p_i) ** n_draw
        pi_approx = np.clip(pi_approx, 1e-9, 1.0)
        
        y = sample[size_col].values
        mu_hat = (y / pi_approx).sum() / (1.0 / pi_approx).sum()
        
        # Bootstrap for SE (approximate)
        m = len(sample)
        boot_estimates = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            bidx = rng.integers(0, m, size=m)
            yb = y[bidx]
            pib = pi_approx[bidx]
            est_b = (yb / pib).sum() / (1.0 / pib).sum()
            boot_estimates[b] = est_b
        se = float(np.std(boot_estimates, ddof=1))
        moe = 1.96 * se
        rel_moe = moe / abs(mu_hat) if mu_hat != 0 else np.inf
        
        stats[n] = {"n": n, "mu_hat": float(mu_hat), "se": se, "moe": float(moe), "rel_moe": float(rel_moe)}
        last_sample = (sample, pi_approx)
        
        if (rel_moe <= target_rel_moe) and (stage >= min_stages):
            break
        n += batch
    
    sample, pi_approx = last_sample
    sample = sample.copy()
    sample["_orig_index"] = sample["_orig_index"].astype(int)
    sample["pi_approx"] = pi_approx
    sample["weight"] = 1.0 / sample["pi_approx"]
    
    certainties = certainties.copy()
    certainties["_orig_index"] = certainties["_orig_index"].astype(int)
    certainties["pi_approx"] = 1.0
    certainties["weight"] = 1.0
    
    sampled_df = pd.concat([certainties, sample], ignore_index=True).reset_index(drop=True)
    sampled_df = sampled_df.sort_values("_orig_index").reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=["_share", "p_single"], errors="ignore")
    
    return sampled_df, stats
