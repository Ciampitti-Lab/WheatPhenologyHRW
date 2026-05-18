"""Rebuild F1 (study-area map) from the canonical v3 training cohort.

The previously shipped F1_study_area.png carried a stale subtitle and
legend (4,538 fields; TX 159 / OK 849 / KS 3,370 / NE 45 / CO 114 / NM 1)
from a pre-v3 cohort. This regenerates it from
features_v3_realsowing_train.parquet so the figure agrees with the
manuscript's training cohort (8,465 field-years; New Mexico folded into
Texas to match the five-state modelling framing of Section 2).

Outputs (overwrites in place; downstream figures/manuscript pick these up):
    docs/figures/F1_study_area.{pdf,png}
    ../paper-overleaf/figures/F1_study_area.{pdf,png}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import plotly.graph_objects as go

EXT = _WORK
TRAIN_FEAT = EXT / 'features_v3_realsowing_train.parquet'
DOCS = ROOT / 'docs' / 'figures'
OVERLEAF = ROOT.parent / 'paper-overleaf' / 'figures'

STATE_COLOR = {'TX': '#d62728', 'OK': '#ff7f0e', 'KS': '#2ca02c',
               'NE': '#1f77b4', 'CO': '#9467bd'}
STATE_ORDER = ['TX', 'OK', 'KS', 'NE', 'CO']


def main():
    ft = pd.read_parquet(TRAIN_FEAT, columns=['field_id', 'state',
                                              'latitude', 'longitude'])
    ft['field_id'] = ft['field_id'].astype(str)
    # Fold the single New Mexico field into Texas (matches manuscript Sec. 2)
    ft['state'] = ft['state'].replace({'NM': 'TX'})
    # One point per unique field (centroid of its field-year rows)
    fld = (ft.groupby(['field_id', 'state'])[['latitude', 'longitude']]
             .mean().reset_index())
    n_total = fld['field_id'].nunique()
    per_state = fld.groupby('state')['field_id'].nunique().to_dict()
    print(f'F1 cohort: {n_total} unique fields  ' +
          '  '.join(f'{s}={per_state.get(s,0)}' for s in STATE_ORDER), flush=True)

    fig = go.Figure()
    for s in STATE_ORDER:
        sub = fld[fld['state'] == s]
        fig.add_trace(go.Scattergeo(
            lon=sub['longitude'], lat=sub['latitude'],
            mode='markers',
            marker=dict(size=3, color=STATE_COLOR[s], opacity=0.55),
            name=f'{s} (n = {per_state.get(s, 0):,})',
            hoverinfo='skip',
        ))
    fig.update_geos(
        scope='usa', resolution=50,
        showland=True, landcolor='#f4f4f4',
        showlakes=False, subunitcolor='#bdbdbd', subunitwidth=0.6,
        countrycolor='#9e9e9e',
        lataxis_range=[31, 43.5], lonaxis_range=[-107, -94],
    )
    fig.update_layout(
        title=dict(
            text=('<b>Study area: the U.S. Hard Red Winter wheat belt</b><br>'
                  f'<span style="font-size:13px">{n_total:,} winter-wheat '
                  'fields with per-field phenology observations '
                  '(training cohort, 2013/14&#8211;2016/17)</span>'),
            x=0.5, xanchor='center'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.08,
                    xanchor='center', x=0.5),
        template='plotly_white',
        margin=dict(l=10, r=10, t=70, b=40),
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    for out in (DOCS, OVERLEAF):
        out.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out / 'F1_study_area.pdf'), width=900, height=650)
        fig.write_image(str(out / 'F1_study_area.png'), width=900, height=650, scale=2)
        print(f'wrote {out}/F1_study_area.{{pdf,png}}', flush=True)


if __name__ == '__main__':
    main()
