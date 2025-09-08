import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def adaptive_pps(df: pd.DataFrame,
                 size_col: str = "amount",
                 start: int = 20,
                 batch: int = 10,
                 max_n: int = 70,
                 target_rel_moe: float = 0.05,
                 n_bootstrap: int = 200,
                 random_state: int = 42,
                 force_two_stages: bool = False,
                 count_certs_in_n: bool = True,
                 cert_threshold: float = None
                ) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Adaptive PPS sampling with dynamic certainty detection and combined inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Population dataframe containing `size_col`.
    size_col : str
        Column name used as size measure.
    start : int
        Starting number of probabilistic draws (after considering certainties as per count_certs_in_n).
    batch : int
        Increment to add each stage.
    max_n : int
        Maximum total sample budget (if count_certs_in_n=True) or maximum probabilistic draws (if False).
    target_rel_moe : float
        Stopping criterion: relative MOE (moe / mean) target for population mean.
    n_bootstrap : int
        Number of bootstrap replications for SE estimation.
    random_state : int
        RNG seed.
    force_two_stages : bool
        If True, require at least two stages before allowing early stop.
    count_certs_in_n : bool
        If True, certainties reduce the remaining budget at each stage (i.e., n is total sample size).
        If False, n refers only to probabilistic draws and certainties are extras.
    cert_threshold : float or None
        If provided, use this fixed share cutoff (e.g., 0.03) to pre-declare certainties.
        If None, use dynamic detection per-stage.
    
    Returns
    -------
    sampled_df : pd.DataFrame
        DataFrame of selected rows (certainties + probabilistic sample) with columns:
        original index, size_col, pi_approx, weight.
    stats : dict
        Stage-wise statistics keyed by stage sample-size (number of probabilistic draws or total n depending).
        Each entry: { 'n_total': int, 'n_prob': int, 'mean_hat':..., 'total_hat':..., 'se':..., 'moe':..., 'rel_moe':... }
    """
    rng = np.random.default_rng(random_state)
    N = len(df)
    total_size_full = df[size_col].sum()
    df = df.copy().reset_index(drop=False).rename(columns={"index": "_orig_index"})
    # per-unit share of full total
    df["_share_full"] = df[size_col] / total_size_full

    # Optionally pre-declare certainties by a fixed share cutoff (business rule), else empty to start
    certainties = pd.DataFrame(columns=df.columns)
    if cert_threshold is not None:
        certainties = df[df["_share_full"] >= cert_threshold].copy()
        rest = df[df["_share_full"] < cert_threshold].copy().reset_index(drop=True)
    else:
        rest = df.copy().reset_index(drop=True)

    # statistics store
    stats = {}

    # helper to compute combined HT estimator (total & mean) given:
    # certainties_df (with actual amounts) and sampled_df (with pi_approx array)
    def combined_ht_estimator(cert_df, samp_df, samp_pi):
        # cert_df: certainties included (pi=1) - DataFrame or empty
        # samp_df: sampled probabilistic units DataFrame (columns include size_col)
        # samp_pi: numpy array of approximate inclusion probs for sampled units
        # Compute HT total: sum(cert amounts) + sum(y_i / pi_i)
        cert_total = 0.0
        if cert_df is not None and len(cert_df) > 0:
            cert_total = float(cert_df[size_col].sum())
        y = samp_df[size_col].values
        # protect against zero division
        samp_pi = np.clip(samp_pi, 1e-12, 1.0)
        ht_total = cert_total + float((y / samp_pi).sum())
        ht_mean = ht_total / N
        return ht_total, ht_mean

    # start adaptive loop
    stage = 0
    # n_total_budget: if count_certs_in_n True, n is total target (including certainties),
    # otherwise n refers only to probabilistic draws. We'll interpret `start` similarly.
    n_total_target = start if count_certs_in_n else None
    # We will iterate by increasing 'n_prob' (number of probabilistic draws); compute n_prob each stage
    n_prob = start if not count_certs_in_n else max(0, start - len(certainties))

    # Main loop: iterate until budget exceeded or stop condition met
    min_stages = 2 if force_two_stages else 1
    last_sample = None

    while True:
        stage += 1

        # Recompute rest_total and single-draw probs on current remainder
        if len(rest) == 0:
            # nothing left to sample
            # finalize with existing certainties only
            samp_df = pd.DataFrame(columns=df.columns)
            samp_pi = np.array([])
            total_hat, mean_hat = combined_ht_estimator(certainties, samp_df, samp_pi)
            stats_key = f"stage_{stage}_final_no_rest"
            stats[stats_key] = {
                'n_total': len(certainties),
                'n_prob': 0,
                'total_hat': total_hat,
                'mean_hat': mean_hat,
                'se': 0.0,
                'moe': 0.0,
                'rel_moe': 0.0
            }
            # assemble sampled_df
            sampled_df = certainties.copy()
            sampled_df["pi_approx"] = 1.0
            sampled_df["weight"] = 1.0
            sampled_df = sampled_df.drop(columns=["_share_full"], errors="ignore")
            return sampled_df.reset_index(drop=True), stats

        rest_total = rest[size_col].sum()
        rest["p_single"] = rest[size_col] / rest_total
        rest_idx = rest.index.values

        # Determine current budget for probabilistic draws (n_prob) and total n_total
        if count_certs_in_n:
            # n_total_target is interpreted as total sample size desired this stage (start, start+batch, ...)
            # In stage loop, we set n_total_current = min(max_n, previous + batch growth)
            # If not set yet, set n_total_target to start then increment by batch later.
            if n_total_target is None:
                n_total_target = start
            # ensure n_total_target never exceeds max_n
            n_total_target = min(n_total_target, max_n)
            # available budget for probabilistic draws this stage:
            n_prob = max(0, n_total_target - len(certainties))
        else:
            # n_prob is tracked directly and bounded by max_n
            n_prob = min(n_prob, max_n)

        # Now dynamic certainty detection for this stage: compute approx pi_if_n for each rest unit
        # Use n_draw = n_prob (we will draw n_prob probabilistic units after certainties)
        n_draw = min(n_prob, len(rest))
        p_all = rest["p_single"].values
        # approximate prob of being selected at least once in n_draw draws from rest
        pi_if_n = 1.0 - (1.0 - p_all) ** n_draw if n_draw > 0 else np.zeros_like(p_all)

        # Mark those that would effectively be certainties at this stage
        # Use threshold very close to 1 to handle numeric precision
        certainty_mask = pi_if_n >= (1.0 - 1e-12)
        if certainty_mask.any():
            # Move these to certainties immediately
            cert_indices = rest.index.values[certainty_mask]
            new_certs = rest.loc[cert_indices].copy()
            # append
            if len(certainties) == 0:
                certainties = new_certs.copy()
            else:
                certainties = pd.concat([certainties, new_certs], ignore_index=True)
            # drop them from rest and restart stage (do not consume probabilistic draws for them)
            rest = rest.drop(index=cert_indices).reset_index(drop=True)
            # If certainties are counted in n_total, they effectively consume part of the budget,
            # so ensure n_total_target and n_prob reflect that on next iteration.
            # continue to next loop iteration to recompute p_single and pi_if_n on the reduced rest
            # (this handles cascades)
            continue

        # If no immediate certainties, proceed to draw n_draw probabilistic units from rest
        if n_draw <= 0:
            # no probabilistic draws allowed (budget exhausted), finalize
            samp_df = pd.DataFrame(columns=df.columns)
            samp_pi = np.array([])
            total_hat, mean_hat = combined_ht_estimator(certainties, samp_df, samp_pi)
            stats_key = f"stage_{stage}_no_budget"
            stats[stats_key] = {
                'n_total': len(certainties),
                'n_prob': 0,
                'total_hat': total_hat,
                'mean_hat': mean_hat,
                'se': 0.0,
                'moe': 0.0,
                'rel_moe': 0.0
            }
            sampled_df = certainties.copy()
            sampled_df["pi_approx"] = 1.0
            sampled_df["weight"] = 1.0
            sampled_df = sampled_df.drop(columns=["_share_full", "p_single"], errors="ignore")
            return sampled_df.reset_index(drop=True), stats

        # Draw without replacement using p_single as probabilities
        selected = rng.choice(rest_idx, size=n_draw, replace=False, p=rest["p_single"].values)
        samp_df = rest.loc[selected].copy().reset_index(drop=True)

        # For the drawn units compute pi_approx = 1 - (1 - p_i)^n_draw
        p_i = samp_df["p_single"].values
        pi_approx = np.clip(1.0 - (1.0 - p_i) ** n_draw, 1e-12, 1.0)

        # Combined HT estimator for total and mean
        total_hat, mean_hat = combined_ht_estimator(certainties, samp_df, pi_approx)

        # Bootstrap to estimate SE of total_hat (resample only the probabilistic sample; keep certainties fixed)
        m = len(samp_df)
        if m == 0:
            se = 0.0
        else:
            boot_est = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                bidx = rng.integers(0, m, size=m)  # sample indices with replacement from the prob sample
                yb = samp_df[size_col].values[bidx]
                pib = pi_approx[bidx]
                # HT total for bootstrap replicate (certainties fixed)
                ht_total_b = float((yb / pib).sum()) + float(certainties[size_col].sum() if len(certainties) > 0 else 0.0)
                boot_est[b] = ht_total_b / N
            se = float(np.std(boot_est, ddof=1))

        moe = 1.96 * se
        rel_moe = moe / abs(mean_hat) if mean_hat != 0 else np.inf

        # record stats (n_total = certs + n_draw if counting certs, else total differs)
        n_total_now = len(certainties) + n_draw if count_certs_in_n else (len(certainties) + n_draw)
        stats_key = f"stage_nprob_{n_draw}"
        stats[stats_key] = {
            'n_total': n_total_now,
            'n_prob': n_draw,
            'total_hat': total_hat,
            'mean_hat': mean_hat,
            'se': se,
            'moe': moe,
            'rel_moe': rel_moe
        }

        last_sample = (samp_df.copy(), pi_approx.copy(), certainties.copy())

        # stopping rule
        if (rel_moe <= target_rel_moe) and (stage >= min_stages):
            break

        # increment stage targets
        if count_certs_in_n:
            # increase total target (so next stage more budget overall)
            # If you want to increase probabilistic draws by batch instead, change logic here
            n_total_target = min(max_n, (n_total_target if n_total_target is not None else start) + batch)
            # on next iteration n_prob will be recomputed as n_total_target - len(certainties)
        else:
            n_prob = min(max_n, n_prob + batch)

        # loop continues

    # After loop, assemble final sampled_df: combine certainties + last probabilistic sample
    samp_df, pi_approx, certainties_snapshot = last_sample
    samp_df = samp_df.copy()
    samp_df["_orig_index"] = samp_df["_orig_index"].astype(int)
    samp_df["pi_approx"] = pi_approx
    samp_df["weight"] = 1.0 / samp_df["pi_approx"]

    certainties = certainties_snapshot.copy()
    certainties["_orig_index"] = certainties["_orig_index"].astype(int)
    certainties["pi_approx"] = 1.0
    certainties["weight"] = 1.0

    sampled_df = pd.concat([certainties, samp_df], ignore_index=True).reset_index(drop=True)
    sampled_df = sampled_df.sort_values("_orig_index").reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=["_share_full", "p_single"], errors="ignore")

    return sampled_df, stats
