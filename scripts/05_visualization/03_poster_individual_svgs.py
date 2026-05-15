"""Generate each poster figure as a standalone, publication-quality SVG.

Outputs to `poster_figures/individual/`:
    F3_r2_heatmap.svg
    F4_strategy_comparison.svg
    F5_pred_vs_obs.svg
    F6_us_plains_map.svg
    F7_key_findings.svg

The wheat-lifecycle figure (F1) and the framework diagram (F2) are produced
externally (AI image generators / vector editor) — see `poster_individual_prompts.md`.

Usage:
    python scripts/05_visualization/03_poster_individual_svgs.py
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils.config import get_config

cfg = get_config()
OUT = ROOT / 'poster_figures' / 'individual'
OUT.mkdir(parents=True, exist_ok=True)

# ─── Purdue palette ──────────────────────────────────────────────────────
GOLD       = '#CEB888'   # Boilermaker Gold (primary)
BLACK      = '#000000'
AGED_GOLD  = '#6B5C36'
STEAM      = '#C0BFC0'
DUST       = '#B19F71'
CREAM      = '#FAF8F2'
DARK_GOLD  = '#4A3B17'
LIGHT_GOLD = '#E8DEC2'

# ─── Common figure styling ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family':         'serif',
    'font.serif':          ['Source Serif Pro', 'Charter', 'Georgia'],
    'font.size':           11,
    'axes.titlesize':      14,
    'axes.titleweight':    'bold',
    'axes.labelsize':      11,
    'axes.linewidth':      1.0,
    'axes.edgecolor':      BLACK,
    'axes.labelcolor':     BLACK,
    'xtick.color':         BLACK,
    'ytick.color':         BLACK,
    'xtick.labelsize':     10,
    'ytick.labelsize':     10,
    'legend.fontsize':     10,
    'legend.frameon':      False,
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
    'savefig.facecolor':   'white',
    'savefig.bbox':        'tight',
    'svg.fonttype':        'none',          # editable text in SVG
    'pdf.fonttype':        42,
})

STAGE_ORDER  = ['emergence','tillering','jointing','flag_leaf','boot','heading','anthesis','maturity']
STAGE_LABELS = {'emergence':'Emergence','tillering':'Tillering','jointing':'Jointing',
                'flag_leaf':'Flag leaf','boot':'Boot','heading':'Heading',
                'anthesis':'Anthesis','maturity':'Maturity'}


# ============================================================================
# F3 — R² heatmap (stages × models, Hybrid)
# ============================================================================
def make_f3_heatmap():
    df = pd.read_csv(cfg.paths.multi_stage_models)
    pv = df[df['strategy']=='C_Hybrid'].pivot(index='stage', columns='model', values='R2')
    pv = pv.reindex(STAGE_ORDER)
    pv.index = [STAGE_LABELS[s] for s in pv.index]

    cmap = LinearSegmentedColormap.from_list(
        'purdue_gold', [CREAM, LIGHT_GOLD, GOLD, AGED_GOLD], N=256)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pv.values, cmap=cmap, vmin=-0.2, vmax=0.85, aspect='auto')

    # Cell text + SOTA borders
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.iloc[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=BLACK, fontsize=10, fontweight='bold')
            if v >= 0.78:
                ax.add_patch(mpatches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                                 edgecolor=BLACK, linewidth=2))
    ax.set_xticks(range(pv.shape[1]))
    ax.set_xticklabels(pv.columns, rotation=0)
    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels(pv.index)
    ax.set_title('Per-stage R² across 5 ML models  (Hybrid: WES + features)',
                 pad=15, color=BLACK)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('R²', color=BLACK)
    cb.outline.set_edgecolor(BLACK)
    cb.ax.axhline(0.78, color=BLACK, linewidth=1.5, linestyle='--')

    fig.savefig(OUT/'F3_r2_heatmap.svg', format='svg')
    plt.close(fig)
    print(f'✓ F3 → {OUT/"F3_r2_heatmap.svg"}')


# ============================================================================
# F4 — Strategy comparison bar chart (A vs B vs C vs D)
# ============================================================================
def make_f4_strategy():
    df = pd.read_csv(cfg.paths.multi_stage_models)
    strat_d = pd.read_csv(cfg.paths.strategy_d_pcse) if Path(cfg.paths.strategy_d_pcse).exists() else None

    # Accept both new (A_WES) and legacy (A_APTT-V) labels for backward compatibility
    a_r2 = df[df['strategy'].isin(['A_WES', 'A_APTT-V'])].set_index('stage')['R2'].reindex(STAGE_ORDER).values
    b_best = (df[df['strategy']=='B_ML-only']
              .sort_values('R2', ascending=False)
              .groupby('stage', as_index=False).first()
              .set_index('stage')[['R2','RMSE','R2_lo','R2_hi']]
              .reindex(STAGE_ORDER))
    c_best = (df[df['strategy']=='C_Hybrid']
              .sort_values('R2', ascending=False)
              .groupby('stage', as_index=False).first()
              .set_index('stage')[['R2','RMSE','R2_lo','R2_hi']]
              .reindex(STAGE_ORDER))
    d_dict = {}
    if strat_d is not None:
        for _, r in strat_d.iterrows():
            d_dict[r['stage']] = r['R2']

    x = np.arange(len(STAGE_ORDER))
    bw = 0.20

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0.78, color=AGED_GOLD, linestyle=':', linewidth=1, label='SOTA threshold (0.78)')
    ax.axhline(0,    color=BLACK, linewidth=0.8)

    bars_a = ax.bar(x - 1.5*bw, a_r2,         bw, label='A — WES (physics)',  color=STEAM,    edgecolor=BLACK, linewidth=0.8)
    bars_b = ax.bar(x - 0.5*bw, b_best['R2'], bw, label='B — ML only',           color=AGED_GOLD, edgecolor=BLACK, linewidth=0.8)
    bars_c = ax.bar(x + 0.5*bw, c_best['R2'], bw, label='C — Hybrid (best)',     color=GOLD,     edgecolor=BLACK, linewidth=0.8)

    # Strategy D — only for stages where WOFOST predicts
    d_stages = [s for s in STAGE_ORDER if s in d_dict]
    d_x      = [STAGE_ORDER.index(s) for s in d_stages]
    d_y      = [d_dict[s] for s in d_stages]
    bars_d = ax.bar(np.array(d_x) + 1.5*bw, d_y, bw, label='D — PCSE-WOFOST (physics)', color=BLACK, edgecolor=BLACK, linewidth=0.8)

    # Value labels on Hybrid (winner)
    for i, v in enumerate(c_best['R2']):
        if not np.isnan(v):
            ax.text(i + 0.5*bw, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color=BLACK)

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], rotation=15, ha='right')
    ax.set_ylabel('R²')
    ax.set_ylim(-1.2, 1.0)
    ax.set_title('Strategy comparison — physics, ML, and hybrid across 8 stages', pad=12)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor=BLACK)

    fig.savefig(OUT/'F4_strategy_comparison.svg', format='svg')
    plt.close(fig)
    print(f'✓ F4 → {OUT/"F4_strategy_comparison.svg"}')


# ============================================================================
# F5 — Predicted vs observed scatter (4 critical stages)
# ============================================================================
def make_f5_pred_vs_obs():
    """Re-runs LOYO ElasticNet quickly to obtain per-stage prediction arrays."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression

    feat = pd.read_parquet(cfg.paths.features)
    pheno = pd.read_parquet(cfg.paths.phenology_matched)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year']     = feat['year'].astype(int)
    if 'state' in feat.columns: feat = feat.drop(columns=['state'])

    STAGE_MAP = dict(cfg.phenology_stages)
    SPRING_ONLY = set(cfg.spring_only_stages)
    obs = None
    for st, labels in STAGE_MAP.items():
        s = pheno[pheno['growth_stage'].isin(labels)].copy()
        if st in SPRING_ONLY:
            s = s[s['dos'] > cfg.spring_dos_min]
        s['hy'] = s['growing_season'].str.split('-').str[1].astype(int)
        e = s.groupby(['FIELDID','hy'])['dos'].min().reset_index()
        e['field_id'] = e['FIELDID'].astype(str); e['year'] = e['hy'].astype(int)
        e = e[['field_id','year','dos']].rename(columns={'dos':f'{st}_dos_obs'})
        obs = e if obs is None else obs.merge(e, on=['field_id','year'], how='outer')
    fwt = feat.merge(obs, on=['field_id','year'], how='left')

    META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used'] + [f'{s}_dos_obs' for s in STAGE_MAP]
    ndre = [c for c in fwt.columns if c.startswith('NDRE')]
    redund = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy','VD_from_emergence_at_SOS',
              'fV_from_emergence_at_SOS','days_emergence_to_SOS']
    cols  = [c for c in fwt.columns if c not in META and c not in ndre and c not in redund]

    def loyo(target):
        d2 = fwt.dropna(subset=[target])
        yp_all, yt_all = [], []
        for yr in sorted(d2['year'].unique()):
            tr = d2[d2['year']!=yr]; te = d2[d2['year']==yr]
            if len(tr)<30 or len(te)<5: continue
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc', StandardScaler()),
                             ('sel', SelectKBest(f_regression, k=min(60, len(cols)))),
                             ('m', ElasticNetCV(cv=5, l1_ratio=[0.5], max_iter=10000))])
            pipe.fit(tr[cols], tr[target])
            yp = pipe.predict(te[cols]).ravel()
            yp_all.extend(yp); yt_all.extend(te[target].values)
        return np.array(yt_all), np.array(yp_all)

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 9), sharex=True, sharey=True)
    crit = ['flag_leaf','boot','heading','anthesis']
    for ax, st in zip(axs.flat, crit):
        yt, yp = loyo(f'{st}_dos_obs')
        n = len(yt)
        rmse = float(np.sqrt(np.mean((yt-yp)**2)))
        denom = float(np.sum((yt-yt.mean())**2))
        r2 = 1 - np.sum((yt-yp)**2)/denom if denom>0 else 0
        w10 = float(np.mean(np.abs(yt-yp)<=10))*100
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())

        ax.fill_between([lo,hi], [lo-10,hi-10], [lo+10,hi+10],
                        color=LIGHT_GOLD, alpha=0.4, edgecolor='none', label='±10 d envelope')
        ax.plot([lo,hi],[lo,hi], '--', color=BLACK, linewidth=0.8, label='1:1 line')
        ax.scatter(yt, yp, s=10, c=GOLD, edgecolors=BLACK, linewidths=0.2, alpha=0.55)

        txt = f'R² = {r2:.2f}\nRMSE = {rmse:.1f} d\nn = {n}\n±10 d = {w10:.0f}%'
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=BLACK, lw=0.7))
        ax.set_title(STAGE_LABELS[st], color=BLACK)

    for ax in axs[1,:]: ax.set_xlabel('Observed DOS')
    for ax in axs[:,0]: ax.set_ylabel('Predicted DOS')
    fig.suptitle('Predicted vs observed (LOYO CV, ElasticNet hybrid)', y=0.98)
    fig.savefig(OUT/'F5_pred_vs_obs.svg', format='svg')
    plt.close(fig)
    print(f'✓ F5 → {OUT/"F5_pred_vs_obs.svg"}')


# ============================================================================
# F6 — US Plains hexbin density map
# ============================================================================
def make_f6_us_map():
    import geopandas as gpd
    feat = pd.read_parquet(cfg.paths.features)
    feat['field_id'] = feat['field_id'].astype(str)

    states_path = Path(cfg.paths.states_shp)
    if not states_path.exists():
        print(f'⚠ states shapefile not found at {states_path} — skipping F6')
        return
    states = gpd.read_file(states_path)
    us = states[states['admin']=='United States of America'].copy()
    target = us[us['postal'].isin(['TX','OK','KS','NE','CO'])]

    fig, ax = plt.subplots(figsize=(9, 7))
    us.plot(ax=ax, color=CREAM, edgecolor=STEAM, linewidth=0.4)
    target.plot(ax=ax, color=STEAM, edgecolor=BLACK, linewidth=0.8)

    # Hexbin density of fields
    hb = ax.hexbin(feat['longitude'], feat['latitude'], gridsize=40,
                   cmap=LinearSegmentedColormap.from_list('gold_density', [CREAM, GOLD, AGED_GOLD]),
                   mincnt=1, edgecolors='none')

    # State labels
    for code in ['TX','OK','KS','NE','CO']:
        s = target[target['postal']==code]
        if len(s) == 0: continue
        c = s.geometry.centroid.iloc[0]
        ax.text(c.x, c.y, code, ha='center', va='center',
                fontsize=14, fontweight='bold', color=BLACK,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=BLACK, lw=0.5, alpha=0.85))

    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Field density', color=BLACK)
    cb.outline.set_edgecolor(BLACK)

    ax.set_xlim(-107, -94)
    ax.set_ylim(28, 43.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title('6,120 wheat fields × 5 growing seasons (2013–2017)', pad=10)

    fig.savefig(OUT/'F6_us_plains_map.svg', format='svg')
    plt.close(fig)
    print(f'✓ F6 → {OUT/"F6_us_plains_map.svg"}')


# ============================================================================
# F7 — Key findings tiles
# ============================================================================
def make_f7_key_findings():
    tiles = [
        ('0.83',  'Anthesis R²',           '(vs 0.70 SOTA)'),
        ('5.1 d', 'Anthesis RMSE',         '(vs 7.7 d SOTA)'),
        ('94 %',  'Predictions within ±10 d', '(4 critical stages)'),
        ('6 120', 'Wheat fields',          '× 5 growing seasons'),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (big, sub, foot) in zip(axs, tiles):
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, 0.06), 0.92, 0.88, boxstyle='round,pad=0.02',
            facecolor=CREAM, edgecolor=BLACK, linewidth=1.2))
        ax.text(0.5, 0.66, big, ha='center', va='center',
                fontsize=44, fontweight='bold', color=AGED_GOLD,
                family='serif')
        ax.text(0.5, 0.34, sub, ha='center', va='center',
                fontsize=14, fontweight='bold', color=BLACK)
        ax.text(0.5, 0.20, foot, ha='center', va='center',
                fontsize=10, color=BLACK, style='italic')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    fig.suptitle('Key findings', x=0.07, y=0.97, ha='left',
                 fontsize=18, fontweight='bold', color=BLACK)
    fig.savefig(OUT/'F7_key_findings.svg', format='svg')
    plt.close(fig)
    print(f'✓ F7 → {OUT/"F7_key_findings.svg"}')


if __name__ == '__main__':
    print(f'Output dir: {OUT}')
    make_f3_heatmap()
    make_f4_strategy()
    make_f5_pred_vs_obs()
    make_f6_us_map()
    make_f7_key_findings()
    print('\nAll individual SVG figures generated.')
    print('For F1 (wheat lifecycle) and F2 (framework diagram), see poster_individual_prompts.md.')
