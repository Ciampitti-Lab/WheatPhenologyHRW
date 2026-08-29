"""R14b - Re-run LOYO and LOSO with the corrected state assignment.

The state one-hot encoders and, more importantly, the leave-one-state-out
folds were built from a latitude/longitude box that gave the Texas
Panhandle to Oklahoma (R14a). This rebuilds the encoders from the spatial
join and re-runs both cross-validation schemes, so the size of the error
is measured rather than assumed.

LOYO is expected to move little: the encoders carry ~0 % of the grouped
permutation importance (R08). LOSO is expected to move, because the folds
themselves change: Texas goes from 168 to 458 fields and Oklahoma from
929 to 630.

Output: data/revision/R14_loyo_compare.csv
        data/revision/R14_loso_compare.csv
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from _common import ORDER, STAGE_LABEL, cohort, metrics, r2, save, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble, STATE_OH

REV = ROOT / 'data' / 'revision'
STUDY = ['TX', 'OK', 'KS', 'NE', 'CO']
# adopted model per stage; anthesis and maturity adopt the FT-Transformer,
# which needs a GPU, so the pre-specified LightGBM stands in for them and the
# comparison there is like-for-like between the two state assignments.
MODEL = {'emergence': 'LightGBM', 'tillering': 'ElasticNet',
         'jointing': 'LightGBM', 'flag_leaf': 'XGBoost', 'boot': 'LightGBM',
         'heading': 'ElasticNet', 'anthesis': 'LightGBM', 'maturity': 'LightGBM'}
WES_ARM = {'tillering': False}   # tillering adopts machine-learning-only


def corrected(fe):
    """Return a copy with state and the one-hot encoders rebuilt."""
    look = pd.read_csv(REV / 'R14_state_lookup.csv')
    look['field_id'] = look['field_id'].astype(str)
    m = look.set_index('field_id')['state_true'].to_dict()
    d = fe.copy()
    d['state_corrected'] = d['field_id'].astype(str).map(m)
    for c in STATE_OH:
        d[c] = 0.0
    for st in d['state_corrected'].dropna().unique():
        col = f'state_{st}'
        if col in d.columns:
            d.loc[d.state_corrected == st, col] = 1.0
    d['state'] = d['state_corrected']
    return d


def loyo(d, feat, tgt, model):
    T, P = [], []
    for y in sorted(d.year.unique()):
        tr, te = d[d.year != y], d[d.year == y]
        if len(tr) < 50 or len(te) < 5:
            continue
        P.extend(fold_pred_ensemble(tr, te, feat, tgt, model))
        T.extend(te[tgt].values)
    return np.asarray(T, float), np.asarray(P, float)


def loso(d, feat, tgt, model, held):
    tr, te = d[d.state != held].copy(), d[d.state == held].copy()
    if len(tr) < 50 or len(te) < 5:
        return None
    for c in STATE_OH:
        if c in feat:
            te[c] = 0.0
    p = fold_pred_ensemble(tr, te, feat, tgt, model)
    return r2(te[tgt].values, p), len(te)


def main():
    fe_pub, cols = cohort()
    fe_cor = corrected(fe_pub)
    print('field counts by state')
    print(pd.DataFrame({
        'published': fe_pub.drop_duplicates('field_id').state.value_counts(),
        'corrected': fe_cor.drop_duplicates('field_id').state.value_counts()
    }).fillna(0).astype(int).to_string(), '\n')

    rows_y, rows_s = [], []
    for st in ORDER:
        model = MODEL[st]
        wes = WES_ARM.get(st, True)
        for tag, fe in [('published', fe_pub), ('corrected', fe_cor)]:
            d, tgt = stage_frame(fe, st)
            feat = [c for c in cols(st, wes) if c in d.columns]
            T, P = loyo(d, feat, tgt, model)
            m = metrics(T, P)
            rows_y.append(dict(stage=st, assignment=tag, model=model,
                               R2=m['R2'], RMSE=m['RMSE'], n=len(d)))
            for held in STUDY:
                out = loso(d, feat, tgt, model, held)
                if out is None:
                    continue
                rows_s.append(dict(stage=st, assignment=tag, state=held,
                                   R2=out[0], n_test=out[1]))
        a, b = rows_y[-2]['R2'], rows_y[-1]['R2']
        print(f'  {STAGE_LABEL[st]:10s} LOYO published={a:+.3f} corrected={b:+.3f} '
              f'delta={b - a:+.3f}', flush=True)

    Y = pd.DataFrame(rows_y); S = pd.DataFrame(rows_s)
    save(Y.round(4), 'R14_loyo_compare.csv'); save(S.round(4), 'R14_loso_compare.csv')

    py = Y.pivot_table(index='stage', columns='assignment', values='R2').reindex(ORDER)
    py['delta'] = py.corrected - py.published
    print('\n=== LOYO ===');  print(py.round(3).to_string())
    print(f'  max |delta| = {py.delta.abs().max():.3f}')

    ps = S.pivot_table(index=['stage', 'state'], columns='assignment', values='R2')
    ps['delta'] = ps.corrected - ps.published
    print('\n=== LOSO, reproductive stages ===')
    rep = ['flag_leaf', 'boot', 'heading', 'anthesis']
    print(ps.loc[ps.index.get_level_values(0).isin(rep)].round(3).to_string())
    print(f'\n  LOSO max |delta| = {ps.delta.abs().max():.3f}')


if __name__ == '__main__':
    main()
