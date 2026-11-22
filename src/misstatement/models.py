"""Model zoo, train-only preprocessing, serial-fraud label treatments and the rolling-origin evaluator."""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import RUSBoostClassifier
import xgboost as xgb
from .config import SEED, GAP
import os
NJOBS = int(os.environ.get('MISSTATE_NJOBS', 1))   # threads per model; 1 avoids oversubscription when notebooks run in parallel


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Prep:
    """Preprocessing fitted on the training window only: optional signed log, 1/99 winsorising,
    median imputation and standardisation. Prevents test-period information entering the model."""

    def __init__(self, cols, log=False):
        self.cols, self.log = list(cols), log

    def fit(self, X):
        Z = X[self.cols].astype(float)
        if self.log:
            Z = slog(Z)
        self.lo, self.hi = Z.quantile(0.01), Z.quantile(0.99)
        Z = Z.clip(self.lo, self.hi, axis=1)
        self.med = Z.median().fillna(0)
        Z = Z.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X):
        Z = X[self.cols].astype(float)
        if self.log:
            Z = slog(Z)
        Z = Z.clip(self.lo, self.hi, axis=1).fillna(self.med)
        return ((Z - self.mu) / self.sd).values


def make_model(name, seed=SEED, **kw):
    if name == 'LR':
        return LogisticRegression(C=kw.get('C', 1.0), max_iter=3000)
    if name == 'RUSBoost':   # Bao et al. (2020) configuration, 1,000 trees
        return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                                  n_estimators=kw.get('n_estimators', 1000), learning_rate=0.1, random_state=seed)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=kw.get('n_estimators', 300), min_samples_leaf=5,
                                      max_features='sqrt', class_weight='balanced_subsample', n_jobs=NJOBS,
                                      random_state=seed)
    if name == 'GBM':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    if name == 'XGB':
        return xgb.XGBClassifier(n_estimators=kw.get('n_estimators', 400), max_depth=kw.get('max_depth', 4),
                                 learning_rate=kw.get('learning_rate', 0.05), subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=kw.get('min_child_weight', 5), reg_lambda=1.0, tree_method='hist',
                                 n_jobs=NJOBS, random_state=seed, eval_metric='logloss')
    raise ValueError(name)


RATIO_SETS = {'ratio', 'accrual', 'xratio'}   # already-scaled ratios: no log transform in the logit
SCALED = {'LR', 'RF', 'RUSBoost'}          # models that get the Prep pipeline (trees with NaN support do not)


def fit_predict(name, fset_cols, train, y, test, seed=SEED, log_lr=True, **kw):
    if name == 'Size':                     # no-model baseline: rank by firm size
        return np.log1p(test['at'].clip(lower=0).fillna(0)).values, None
    m = make_model(name, seed, **kw)
    if name in SCALED:
        pp = Prep(fset_cols, log=(name == 'LR' and log_lr)).fit(train)
        m.fit(pp.transform(train), y)
        return m.predict_proba(pp.transform(test))[:, 1], (m, pp)
    m.fit(train[fset_cols].values.astype(float), y)
    return m.predict_proba(test[fset_cols].values.astype(float))[:, 1], (m, None)


def training_labels(train, test, treatment, y_col='misstate', aaer_col='p_aaer', firm_col='gvkey'):
    """Serial-fraud handling.
    naive     : keep every training label (look-ahead leakage when a fraud spans train and test years)
    recode    : Bao et al. (2020) intent - training firm-years of AAERs present in the test year set to 0
    firm_excl : drop all training firm-years of firms that are fraudulent in the test year."""
    y = train[y_col].values.copy()
    keep = np.ones(len(train), bool)
    pos = test[test[y_col] == 1]
    if treatment == 'recode':
        y[train[aaer_col].isin(pos[aaer_col].dropna().unique()).values] = 0
    elif treatment == 'firm_excl':
        keep = ~train[firm_col].isin(pos[firm_col].unique()).values
    elif treatment != 'naive':
        raise ValueError(treatment)
    return y, keep


def tune_xgb(train, y, cols, seed=SEED, grid=None):
    """Time-ordered tuning inside the training window: fit on years <= v-GAP, validate on year v
    (the last training year). Returns the best parameter dict by validation AUC."""
    grid = grid or [dict(max_depth=d, n_estimators=n, min_child_weight=w)
                    for d in (2, 4, 6) for n in (200, 500) for w in (5, 20)]
    v = train.fyear.max()
    tr, va = train.fyear <= v - GAP, train.fyear == v
    if y[va.values].sum() == 0:
        return dict(max_depth=4, n_estimators=400, min_child_weight=5)
    best, best_auc = None, -1
    for g in grid:
        m = make_model('XGB', seed, **g)
        m.fit(train.loc[tr, cols].values.astype(float), y[tr.values])
        auc = roc_auc_score(y[va.values], m.predict_proba(train.loc[va, cols].values.astype(float))[:, 1])
        if auc > best_auc:
            best, best_auc = g, auc
    return best


def rolling(d, configs, test_years, fsets, y_col='misstate', aaer_col='p_aaer', firm_col='gvkey',
            seeds=(SEED,), gap=GAP, min_year=None, verbose=True, label_date_col=None):
    """Expanding-window, rolling-origin evaluation: train on years <= T-gap, score year T.
    configs: list of (model, feature_set_name, treatment). Returns long frame of out-of-sample scores.
    label_date_col: if given, a training firm-year counts as positive only if its label became public
    (e.g. the restating 10-K/A was filed) by 31 December of the test year - i.e. before the test-year 10-K."""
    out = []
    for T in test_years:
        train = d[(d.fyear <= T - gap) & ((d.fyear >= min_year) if min_year else True)]
        test = d[d.fyear == T]
        if test[y_col].sum() == 0 or train[y_col].sum() == 0:
            continue
        for (mname, fset, treat) in configs:
            cols = fsets[fset]
            y, keep = training_labels(train, test, treat, y_col, aaer_col, firm_col)
            if label_date_col is not None:
                known = pd.to_datetime(train[label_date_col]) <= pd.Timestamp(f'{T}-12-31')
                y = np.where(known.values, y, 0)
            tr, yy = train[keep], y[keep]
            for seed in (seeds if mname in ('RUSBoost', 'XGB', 'RF', 'GBM', 'XGB-tuned') else (SEED,)):
                t0 = time.time()
                if mname == 'XGB-tuned':
                    best = tune_xgb(tr, yy, cols, seed)
                    s, _ = fit_predict('XGB', cols, tr, yy, test, seed, **best)
                else:
                    s, _ = fit_predict(mname, cols, tr, yy, test, seed, log_lr=(fset not in RATIO_SETS))
                out.append(pd.DataFrame(dict(fyear=T, firm=test[firm_col].values, y=test[y_col].values,
                                             model=mname, fset=fset, treat=treat, seed=seed, score=s)))
                if verbose:
                    print(f'{T} {mname:9s} {fset:6s} {treat:9s} seed={seed} {time.time()-t0:5.1f}s', flush=True)
    return pd.concat(out, ignore_index=True)


def cached(path, fn, *a, **kw):
    """Load a parquet cache if it exists, otherwise compute and save it."""
    import os
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = fn(*a, **kw)
    df.to_parquet(path)
    return df
