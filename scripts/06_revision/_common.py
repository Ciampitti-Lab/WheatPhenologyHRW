"""Shared helpers for the JAG revision analyses (round 1, JAG-D-26-03614).

Everything here reuses the *published* pipeline core in
`scripts.utils.deep_models` so that any number produced for the revision
is directly comparable with the submitted manuscript. Nothing in this
module changes the modelling logic; it only adds bookkeeping the
reviewers asked for (feature-group membership, cohort provenance,
per-fold prediction capture).

Reviewer mapping is documented per script in `scripts/06_revision/`.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.utils.config import CFG
from scripts.utils.deep_models import (  # noqa: F401  (re-exported)
    ADOPT, EARLY, LOSO_STATES, MODELS, ORDER, SPRING, STAGE_MAP, STATE_OH, WE,
    boot_ci, fold_pred_ensemble, is_win, load_cohort, loyo, metrics, r2,
    stage_frame,
)

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
OUT = REPO_ROOT / 'data' / 'revision'
OUT.mkdir(parents=True, exist_ok=True)

# Human-readable stage labels used in every revision table/figure.
STAGE_LABEL = {
    'emergence': 'Emergence', 'tillering': 'Tillering', 'jointing': 'Jointing',
    'flag_leaf': 'Flag leaf', 'boot': 'Boot', 'heading': 'Heading',
    'anthesis': 'Anthesis', 'maturity': 'Maturity',
}
REPRODUCTIVE = ['flag_leaf', 'boot', 'heading', 'anthesis']


def cohort():
    """The exact frame + column selector the manuscript pipeline uses."""
    return load_cohort(WORK, PHENO)


# --------------------------------------------------------------------------
# Feature-group membership (reviewer #6, minor 2: "a complete input-feature
# table ... data source, temporal window, spatial resolution").
# The grouping mirrors the one behind manuscript Figure 5 and is the single
# definition reused by every revision script, so the ablation groups and the
# importance groups can no longer drift apart.
# --------------------------------------------------------------------------
GROUP_RULES = [
    ('WES',        lambda c: c in WE),
    ('State',      lambda c: c in STATE_OH),
    ('HLS',        lambda c: c.startswith(('NDVI', 'EVI', 'GCVI', 'NDRE', 'DL_c'))),
    ('LST',        lambda c: c.startswith(('lst_day', 'lst_night'))),
    ('ThermalTime', lambda c: c in ('GDD_at_SOS', 'GDD_M2_at_SOS', 'GDD_eff_at_SOS',
                                    'fV_at_SOS', 'VD_at_SOS', 'photoperiod_at_SOS',
                                    'sowing_doy_used_actual', 'gdd_cum_pa')),
    ('Site',       lambda c: c in ('latitude', 'longitude', 'ph_top')),
]


def feature_group(col):
    """Map one feature column to its provenance group (Daymet = default)."""
    for name, rule in GROUP_RULES:
        if rule(col):
            return name
    return 'Daymet'


def group_map(feature_cols):
    """dict: group -> [columns], restricted to `feature_cols`."""
    out = {}
    for c in feature_cols:
        out.setdefault(feature_group(c), []).append(c)
    return out


def loyo_predictions(d, feat, tgt, model):
    """LOYO like `deep_models.loyo` but also returns the held-out year and
    state of every prediction, which the per-region diagnostics need."""
    T, P, Y, S = [], [], [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        pr = fold_pred_ensemble(tr, te, feat, tgt, model)
        P.extend(pr); T.extend(te[tgt].values)
        Y.extend(te['year'].values); S.extend(te['state'].values)
    return (np.asarray(T, float), np.asarray(P, float),
            np.asarray(Y), np.asarray(S))


def save(df, name, index=False):
    """Write a revision result next to the others and echo a preview."""
    p = OUT / name
    df.to_csv(p, index=index)
    print(f'  -> {p.relative_to(REPO_ROOT)}  ({len(df)} rows)', flush=True)
    return p
