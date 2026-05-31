import numpy as np
import pandas as pd


def sensitivity_analysis(df, weights, S=200, K=10, scale=0.25, seed=123):
    FEATS = list(weights.keys())
    W0 = pd.Series(weights, dtype=float)
    W0 = W0 / W0.abs().sum()  # L1 normalize

    X = df[FEATS].copy()
    score_base = (X * W0).sum(axis=1)
    rank_base  = score_base.rank(ascending=False, method="min").astype(int)

    # ---- 1) Global ±25% perturbations on ALL weights ----
    rng = np.random.default_rng(seed)
    K = min(K, len(df))  # top-K overlap metric


    spearmans, topk_overlap, abs_rank_mean, abs_rank_p90 = [], [], [], []
    for _ in range(S):
        noise = rng.uniform(1-scale, 1+scale, size=len(W0))
        Wp = pd.Series(W0.values * noise, index=W0.index)
        Wp = Wp / Wp.abs().sum()  # re-normalize

        score_p = (X * Wp).sum(axis=1)
        rank_p  = score_p.rank(ascending=False, method="min").astype(int)

        # similarity metrics
        spearmans.append(score_base.corr(score_p, method="spearman"))
        top_base = set(score_base.nlargest(K).index)
        top_pert = set(score_p.nlargest(K).index)
        topk_overlap.append(len(top_base & top_pert) / float(K))
        dr = (rank_p - rank_base).abs()
        abs_rank_mean.append(float(dr.mean()))
        abs_rank_p90.append(float(dr.quantile(0.90)))

    sens_global = pd.DataFrame({
        "spearman_rho_mean": [np.nanmean(spearmans)],
        "spearman_rho_p10":  [np.nanquantile(spearmans, 0.10)],
        "spearman_rho_p90":  [np.nanquantile(spearmans, 0.90)],
        f"top{K}_overlap_mean": [np.mean(topk_overlap)],
        f"top{K}_overlap_p10":  [np.quantile(topk_overlap, 0.10)],
        f"top{K}_overlap_p90":  [np.quantile(topk_overlap, 0.90)],
        "abs_rank_change_mean": [np.mean(abs_rank_mean)/ len(df)],
        "abs_rank_change_p90":  [np.mean(abs_rank_p90)/ len(df)],
    }).round(3)

    # ---- 2) One-at-a-time ±25% per feature ----
    rows = []
    for f in FEATS:
        Wp = W0.copy()
        Wp[f] *= 1.5
        Wp = Wp / Wp.abs().sum()
        s_up  = (X * Wp).sum(axis=1)
        r_up  = s_up.rank(ascending=False, method="min").astype(int)
        dr_up = (r_up - rank_base).abs()

        Wp = W0.copy()
        Wp[f] *= 0.5
        Wp = Wp / Wp.abs().sum()
        s_dn  = (X * Wp).sum(axis=1)
        r_dn  = s_dn.rank(ascending=False, method="min").astype(int)
        dr_dn = (r_dn - rank_base).abs()

        rows.append({
            "feature": f,
            "mean_abs_rank_change_up":   float(dr_up.mean() / len(df)),
            "p90_abs_rank_change_up":    float(dr_up.quantile(0.90) / len(df)),
            "mean_abs_rank_change_down": float(dr_dn.mean() / len(df)),
            "p90_abs_rank_change_down":  float(dr_dn.quantile(0.90) / len(df)),
            "spearman_up":               float(score_base.corr(s_up, method="spearman")),
            "spearman_down":             float(score_base.corr(s_dn, method="spearman")),
        })

    sens_oaat = pd.DataFrame(rows).sort_values("p90_abs_rank_change_up", ascending=False).round(3)
    return sens_global, sens_oaat
