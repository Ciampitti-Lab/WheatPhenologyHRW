"""Cross-validation strategies and bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans


# ─── bootstrap CIs ───────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, n_iter=1000, seed=42, ci=95):
    """Bootstrap CIs for RMSE, MAE, R², and ±10/15 day windows.

    Returns {metric: (median, ci_low, ci_high)}.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    rmse_l, mae_l, r2_l, w10_l, w15_l = [], [], [], [], []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        rmse_l.append(np.sqrt(np.mean((yt - yp) ** 2)))
        mae_l.append(np.mean(np.abs(yt - yp)))
        denom = np.sum((yt - yt.mean()) ** 2)
        r2_l.append(1 - np.sum((yt - yp) ** 2) / denom if denom > 0 else 0)
        err = np.abs(yt - yp)
        w10_l.append(np.mean(err <= 10) * 100)
        w15_l.append(np.mean(err <= 15) * 100)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        'RMSE': (np.median(rmse_l), np.percentile(rmse_l, lo), np.percentile(rmse_l, hi)),
        'MAE':  (np.median(mae_l),  np.percentile(mae_l, lo),  np.percentile(mae_l, hi)),
        'R2':   (np.median(r2_l),   np.percentile(r2_l, lo),   np.percentile(r2_l, hi)),
        'w10':  (np.median(w10_l),  np.percentile(w10_l, lo),  np.percentile(w10_l, hi)),
        'w15':  (np.median(w15_l),  np.percentile(w15_l, lo),  np.percentile(w15_l, hi)),
    }


# ─── cross-validation strategies ─────────────────────────────────────────────

def loyo_cv(df, feat_cols, target_col, model_factory, scale=True):
    """Leave-One-Year-Out CV. Returns (y_pred, y_true) as np arrays."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    df2 = df.dropna(subset=[target_col]).copy()
    all_pred, all_true = [], []
    for yr in sorted(df2['year'].unique()):
        tr = df2[df2['year'] != yr]
        te = df2[df2['year'] == yr]
        steps = [('imp', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('sc', StandardScaler()))
        steps.append(('m', model_factory()))
        pipe = Pipeline(steps)
        pipe.fit(tr[feat_cols], tr[target_col])
        pred = pipe.predict(te[feat_cols])
        if pred.ndim > 1:
            pred = pred.ravel()
        all_pred.extend(pred)
        all_true.extend(te[target_col].values)
    return np.array(all_pred), np.array(all_true)


def loro_cv(df, feat_cols, target_col, model_factory, n_regions=5, scale=True, seed=42):
    """Leave-One-Region-Out CV. Regions defined by K-Means on (lat, lon)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    df2 = df.dropna(subset=[target_col]).copy()
    coords = df2.groupby('field_id')[['latitude', 'longitude']].first().reset_index()
    km = KMeans(n_clusters=n_regions, random_state=seed, n_init=10).fit(coords[['latitude', 'longitude']])
    coords['region'] = km.labels_
    df2 = df2.merge(coords[['field_id', 'region']], on='field_id', how='left')

    all_pred, all_true, all_region = [], [], []
    for r in sorted(df2['region'].unique()):
        tr = df2[df2['region'] != r]
        te = df2[df2['region'] == r]
        steps = [('imp', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('sc', StandardScaler()))
        steps.append(('m', model_factory()))
        pipe = Pipeline(steps)
        pipe.fit(tr[feat_cols], tr[target_col])
        pred = pipe.predict(te[feat_cols])
        if pred.ndim > 1:
            pred = pred.ravel()
        all_pred.extend(pred)
        all_true.extend(te[target_col].values)
        all_region.extend([r] * len(te))
    return np.array(all_pred), np.array(all_true), np.array(all_region)


def field_holdout_split(df, test_size=0.2, val_size=0.5, seed=42):
    """80/10/10 split grouped by field_id (no field appears in more than one fold)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, temp_idx = next(gss1.split(df, groups=df['field_id']))
    df_tr = df.iloc[tr_idx]
    df_temp = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    val_idx, te_idx = next(gss2.split(df_temp, groups=df_temp['field_id']))
    return df_tr, df_temp.iloc[val_idx], df_temp.iloc[te_idx]


def forward_temporal_cv(df, feat_cols, target_col, model_factory,
                        train_years_list, test_year_list, scale=True):
    """Walk-forward CV: train on past years, test on the next year.

    train_years_list : list[set]  e.g. [{2013,2014,2015}, {2013,2014,2015,2016}]
    test_year_list   : list[int]  e.g. [2016, 2017]
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    results = []
    df2 = df.dropna(subset=[target_col]).copy()
    for tr_years, te_year in zip(train_years_list, test_year_list):
        tr = df2[df2['year'].isin(tr_years)]
        te = df2[df2['year'] == te_year]
        steps = [('imp', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('sc', StandardScaler()))
        steps.append(('m', model_factory()))
        pipe = Pipeline(steps)
        pipe.fit(tr[feat_cols], tr[target_col])
        pred = pipe.predict(te[feat_cols])
        if pred.ndim > 1:
            pred = pred.ravel()
        yt = te[target_col].values
        rmse = np.sqrt(np.mean((yt - pred) ** 2))
        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - np.sum((yt - pred) ** 2) / denom if denom > 0 else 0
        results.append({
            'train_years': sorted(tr_years), 'test_year': te_year,
            'n_train': len(tr), 'n_test': len(te),
            'RMSE': rmse, 'R2': r2,
            'w10': np.mean(np.abs(yt - pred) <= 10) * 100,
        })
    return pd.DataFrame(results)
