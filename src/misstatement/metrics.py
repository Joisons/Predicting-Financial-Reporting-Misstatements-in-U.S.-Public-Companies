"""Evaluation metrics: discrimination, investigation-budget metrics, calibration and firm-cluster bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score


def ndcg_at_k(y, s, k=0.01):
    """NDCG@k as in Bao et al. (2020): discounted gain of the top-k% list relative to the ideal list."""
    y, s = np.asarray(y), np.asarray(s)
    n = max(int(np.floor(len(y) * k)), 1)
    order = np.argsort(-s, kind='mergesort')[:n]
    disc = 1 / np.log2(np.arange(2, n + 2))
    dcg = (y[order] * disc).sum()
    idcg = (np.sort(y)[::-1][:n] * disc).sum()
    return dcg / idcg if idcg > 0 else np.nan


def topk(y, s, k):
    """Hits, sensitivity (capture), precision and lift when the top k share of firms is investigated."""
    y, s = np.asarray(y), np.asarray(s)
    n = max(int(np.floor(len(y) * k)), 1)
    hits = y[np.argsort(-s, kind='mergesort')[:n]].sum()
    base = max(y.mean(), 1e-12)
    return dict(hits=int(hits), sens=hits / max(y.sum(), 1), prec=hits / n, lift=(hits / n) / base, n_inv=n)


def calibration(y, p):
    """Brier score, calibration slope and intercept (logistic recalibration), expected/observed ratio."""
    import statsmodels.api as sm
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        slope = intercept = np.nan
    return dict(brier=brier_score_loss(y, p), cal_slope=slope, cal_intercept=intercept,
                exp_obs=p.sum() / max(y.sum(), 1))


def evaluate(y, s, ks=(0.01, 0.05)):
    y = np.asarray(y)
    ok = 0 < y.sum() < len(y)
    out = dict(n=len(y), n_pos=int(y.sum()), auc=roc_auc_score(y, s) if ok else np.nan,
               ap=average_precision_score(y, s) if ok else np.nan, ndcg1=ndcg_at_k(y, s, 0.01))
    for k in ks:
        for name, v in topk(y, s, k).items():
            out[f'{name}@{k*100:g}'] = v
    return out


def per_year(pred, keys, label='y'):
    """Metrics per model configuration and test year from a long prediction frame."""
    rows = []
    for k, g in pred.groupby(keys + ['fyear']):
        rows.append(dict(zip(keys + ['fyear'], k), **evaluate(g[label].values, g.score.values)))
    return pd.DataFrame(rows)


def summarise(py, keys, years):
    """Mean annual AUC/NDCG and hits summed over test years (the convention of Bao et al.)."""
    s = py[py.fyear.isin(years)].groupby(keys).agg(
        auc=('auc', 'mean'), ap=('ap', 'mean'), ndcg1=('ndcg1', 'mean'),
        hits1=('hits@1', 'sum'), hits5=('hits@5', 'sum'), pos=('n_pos', 'sum'), n=('n', 'sum'),
        inv1=('n_inv@1', 'sum'), inv5=('n_inv@5', 'sum')).reset_index()
    s['capture1'] = s.hits1 / s.pos
    s['capture5'] = s.hits5 / s.pos
    s['prec1'] = s.hits1 / s.inv1
    s['prec5'] = s.hits5 / s.inv5
    return s


def budget_curve(pred, keys, ks=(0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20), label='y'):
    """Capture and precision at a within-year investigation budget k, pooled over test years."""
    rows = []
    for key, g in pred.groupby(keys):
        for k in ks:
            hits = inv = 0
            for _, gy in g.groupby('fyear'):
                n = max(int(len(gy) * k), 1)
                hits += gy.nlargest(n, 'score')[label].sum(); inv += n
            rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)), k=k, hits=int(hits),
                             capture=hits / g[label].sum(), precision=hits / inv, investigated=inv))
    return pd.DataFrame(rows)


def _mean_auc(df, label):
    v = [roc_auc_score(g[label], g.score) for _, g in df.groupby('fyear') if 0 < g[label].sum() < len(g)]
    return np.mean(v) if v else np.nan


def cluster_bootstrap(pred_a, pred_b=None, cluster='firm', B=500, seed=0, label='y', stat='auc', k=0.01):
    """Firm-cluster bootstrap of mean annual AUC (or top-k capture). With pred_b, returns the paired difference.
    pred_a / pred_b must hold the same firm-years in the same order."""
    rng = np.random.default_rng(seed)
    a = pred_a.reset_index(drop=True)
    b = pred_b.reset_index(drop=True) if pred_b is not None else None
    firms = a[cluster].unique()
    idx = a.groupby(cluster).indices

    def f(df):
        if stat == 'auc':
            return _mean_auc(df, label)
        hits = sum(gy.nlargest(max(int(len(gy) * k), 1), 'score')[label].sum() for _, gy in df.groupby('fyear'))
        return hits / max(df[label].sum(), 1)
    point = f(a) - (f(b) if b is not None else 0)
    draws = []
    for _ in range(B):
        s = np.concatenate([idx[x] for x in rng.choice(firms, len(firms))])
        v = f(a.iloc[s]) - (f(b.iloc[s]) if b is not None else 0)
        draws.append(v)
    draws = np.array(draws)
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean()) if b is not None else np.nan
    return dict(estimate=point, lo=np.nanpercentile(draws, 2.5), hi=np.nanpercentile(draws, 97.5), p_boot=p)


def psi(ref, cur, bins=10):
    """Population stability index of `cur` against decile bins of `ref`."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    q = np.unique(np.quantile(ref[~np.isnan(ref)], np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return np.nan
    e = np.clip(np.histogram(np.clip(ref, q[0], q[-1]), q)[0] / len(ref), 1e-4, None)
    a = np.clip(np.histogram(np.clip(cur, q[0], q[-1]), q)[0] / len(cur), 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())
