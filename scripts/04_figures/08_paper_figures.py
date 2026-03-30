"""Paper figures (plotly).

Single source of truth for the publication-ready figures. Produces five PNGs
in docs/figures/, suitable for direct inclusion in the manuscript:

    F3_per_stage_scatter.png         Predicted vs. observed (LOYO, 8 stages)
    F4_strategy_comparison.png       Physiology-informed vs. ML-only per stage
    F5_feature_importance.png        Feature-group importance heatmap
    F6_loso_transferability.png      Leave-one-state-out generalization
    F7_phenology_trends.png          State-level phenology trends 2018-2024

All figures use a consistent palette, no jargon ("hybrid wins", "v3", ...),
and serif-style titles for journal compatibility. Static export through kaleido.
"""

# --- repo-portable paths (no hardcoded cluster paths) --------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
# ------------------------------------------------------------------------

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

EXT = _WORK
FIG_DIR = ROOT / 'docs' / 'figures'

# Data sources
PHASE_E      = EXT / 'v3_results' / 'multi_stage_models_a6_gs.parquet'
LOYO_PREDS   = EXT / 'v3_loyo_predictions.parquet'
LOSO_RES     = EXT / 'v3_loso_results.csv'
FEAT_IMP     = FIG_DIR / 'feature_importance_per_stage.csv'
EXT_PREDS    = EXT / 'predictions_all_stages_2018_2024.parquet'
TRENDS       = EXT / 'v3_trends_per_stage_per_state.csv'

# ─── Styling ─────────────────────────────────────────────────────
STAGES_ORDER = ['emergence', 'tillering', 'jointing', 'flag_leaf',
                'boot', 'heading', 'anthesis', 'maturity']
STAGE_LABEL = {
    'emergence': 'Emergence', 'tillering': 'Tillering', 'jointing': 'Jointing',
    'flag_leaf': 'Flag leaf', 'boot': 'Boot', 'heading': 'Heading',
    'anthesis': 'Anthesis', 'maturity': 'Maturity',
}
STATES_ORDER = ['TX', 'OK', 'KS', 'NE', 'CO']
STATE_COLOR = {
    'TX': '#2E5C8A', 'OK': '#B23A48', 'KS': '#8B6F47',
    'NE': '#2C7A66', 'CO': '#C9A961', 'NM': '#6B4E7A',
}

PURDUE_GOLD = '#CEB888'
PURDUE_BROWN = '#8E6F3E'
INK = '#1A1A1A'
MUTED = '#666666'

LAYOUT_BASE = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='Helvetica, Arial, sans-serif', size=20, color=INK),
    margin=dict(l=85, r=35, t=100, b=80),
)

AXIS_BASE = dict(
    showline=True, linewidth=1.4, linecolor=INK, mirror=False,
    ticks='outside', tickfont=dict(size=18), ticklen=5,
    gridcolor='#EEEEEE', gridwidth=0.5,
    title_font=dict(size=20),
)

OUTPUT_FORMAT = 'pdf'   # vector output for LaTeX inclusion

# ─── Helpers ─────────────────────────────────────────────────────
def save(fig, name, width=1100, height=700):
    """Save figure as PDF (vector) and a PNG companion for previewing."""
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    pdf_out = FIG_DIR / f'{stem}.pdf'
    png_out = FIG_DIR / f'{stem}.png'
    fig.write_image(str(pdf_out), width=width, height=height)
    fig.write_image(str(png_out), width=width, height=height, scale=2)
    print(f'  → {pdf_out}')
    print(f'  → {png_out}')
    return pdf_out


# ─── F3: Predicted vs. observed (LOYO) ───────────────────────────
def figure_per_stage_scatter():
    print('F3 — per-stage scatter (LOYO)')
    preds = pd.read_parquet(LOYO_PREDS)
    # Physiologically plausible day-of-season ranges per stage (DOS = days since 1 Jul
    # of harvest_year-1). Observations or predictions outside the range are dropped
    # as labelling artefacts (e.g. mismatched harvest years) before computing the
    # scatter metrics shown in each panel.
    DOS_BOUNDS = {
        'emergence': (40, 280), 'tillering': (140, 300), 'jointing': (190, 320),
        'flag_leaf': (240, 340), 'boot': (260, 345), 'heading': (270, 355),
        'anthesis': (280, 355), 'maturity': (280, 365),
    }
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=[f'<b>{STAGE_LABEL[s]}</b>' for s in STAGES_ORDER],
                        horizontal_spacing=0.08, vertical_spacing=0.20)
    for k, stage in enumerate(STAGES_ORDER):
        r, c = k // 4 + 1, k % 4 + 1
        sub = preds[preds['stage'] == stage]
        if len(sub) == 0:
            continue
        lo_b, hi_b = DOS_BOUNDS[stage]
        sub_v = sub[(sub['observed'].between(lo_b, hi_b)) &
                    (sub['predicted'].between(lo_b, hi_b))]
        if len(sub_v) < 30:
            continue
        y_obs, y_pred = sub_v['observed'].values, sub_v['predicted'].values
        ss_res = np.sum((y_obs - y_pred) ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))

        lo = min(y_obs.min(), y_pred.min())
        hi = max(y_obs.max(), y_pred.max())
        pad = (hi - lo) * 0.06
        lo, hi = lo - pad, hi + pad

        fig.add_trace(go.Scatter(
            x=y_obs, y=y_pred, mode='markers',
            marker=dict(size=6, color=PURDUE_BROWN, opacity=0.35,
                        line=dict(width=0)),
            showlegend=False, hoverinfo='skip',
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode='lines',
            line=dict(color=INK, width=1.3, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ), row=r, col=c)
        ax_id = '' if k == 0 else str(k + 1)
        fig.add_annotation(
            xref=f'x{ax_id} domain', yref=f'y{ax_id} domain',
            x=0.04, y=0.96, xanchor='left', yanchor='top',
            text=f'<b>R² = {r2:.2f}</b><br>RMSE = {rmse:.1f} d<br>n = {len(sub_v):,}',
            showarrow=False, font=dict(size=14, color=INK),
            bgcolor='rgba(255,255,255,0.92)', borderpad=6,
            bordercolor='#BBBBBB', borderwidth=0.8,
        )
        fig.update_xaxes(range=[lo, hi], row=r, col=c, **AXIS_BASE)
        fig.update_yaxes(range=[lo, hi], row=r, col=c, **AXIS_BASE)
        if r == 2:
            fig.update_xaxes(title_text='Observed DOS', title_font=dict(size=15),
                             row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text='Predicted DOS', title_font=dict(size=15),
                             row=r, col=c)

    fig.update_layout(
        title=dict(text='<b>Predicted versus observed day-of-season for the eight phenological stages</b><br>'
                        '<span style="font-size:14px;color:#555">Leave-one-year-out cross-validation; '
                        'dashed line shows the 1:1 reference.</span>',
                   x=0.02, xanchor='left', y=0.98, font=dict(size=20)),
        height=720, width=1200, **LAYOUT_BASE,
    )
    for ann in fig['layout']['annotations'][:8]:
        ann['font'] = dict(size=20, color=INK)
    save(fig, 'F3_per_stage_scatter', width=1200, height=720)


# ─── F4: Feature-strategy comparison ─────────────────────────────
def figure_strategy_comparison():
    """F4 — horizontal grouped bar chart: per-stage R² of the two strategies.

    Stages are arranged vertically (no axis-label rotation). Each stage has
    two bars side by side: machine-learning only on top, physiology-informed
    on the bottom of the pair. ΔR² is shown at the right edge of each pair.
    """
    print('F4 — strategy comparison (horizontal grouped bars)')
    res = pd.read_parquet(PHASE_E)
    best = res.loc[res.groupby(['stage', 'strategy'])['R2'].idxmax()].reset_index(drop=True)
    pivot = best.pivot_table(index='stage', columns='strategy', values='R2').reindex(STAGES_ORDER)
    pivot = pivot.iloc[::-1]   # emergence at top after reversal

    y_labels = [STAGE_LABEL[s] for s in pivot.index]
    ml_vals = pivot['B_ML-only'].values
    hy_vals = pivot['C_Hybrid'].values
    deltas = hy_vals - ml_vals

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_labels, x=ml_vals,
        orientation='h',
        name='Machine-learning only',
        marker=dict(color=MUTED, line=dict(color=INK, width=0.8)),
        text=[f'{v:.2f}' for v in ml_vals],
        textposition='inside', insidetextanchor='end',
        textfont=dict(color='white', size=16, family='Helvetica, Arial, sans-serif'),
        hovertemplate='%{y}<br>ML-only R²: %{x:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=y_labels, x=hy_vals,
        orientation='h',
        name='Physiology-informed (Wang–Engel–Streck + ML)',
        marker=dict(color=PURDUE_BROWN, line=dict(color=INK, width=0.8)),
        text=[f'{v:.2f}' for v in hy_vals],
        textposition='inside', insidetextanchor='end',
        textfont=dict(color='white', size=16, family='Helvetica, Arial, sans-serif'),
        hovertemplate='%{y}<br>Physiology-informed R²: %{x:.2f}<extra></extra>',
    ))

    # ΔR² annotation at the right margin
    x_anno = max(max(ml_vals), max(hy_vals)) + 0.06
    for ylab, d in zip(y_labels, deltas):
        fig.add_annotation(
            x=x_anno, y=ylab,
            text=f'<b>{d:+.2f}</b>',
            showarrow=False,
            font=dict(size=19, color='#1B6E63' if d > 0 else '#A03939'),
            xanchor='left', yanchor='middle',
        )
    fig.add_annotation(
        x=x_anno, y=1.04, yref='paper', xanchor='left',
        text='<b>ΔR²</b>', showarrow=False,
        font=dict(size=17, color=INK),
    )

    fig.update_layout(
        title=dict(text='<b>Per-stage LOYO R² of the two feature strategies</b><br>'
                        '<span style="font-size:16px;color:#555">'
                        'Best score per (stage, strategy) across five candidate models. '
                        'ΔR² shown on the right.</span>',
                   x=0.02, xanchor='left', y=0.97, font=dict(size=22)),
        xaxis=dict(title='LOYO R²', range=[0, x_anno + 0.14], **AXIS_BASE),
        yaxis=dict(title='', tickfont=dict(size=20), showline=True,
                   linecolor=INK, linewidth=1.4,
                   ticks='outside', ticklen=5),
        barmode='group',
        bargap=0.25, bargroupgap=0.10,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.16,
                    bgcolor='rgba(255,255,255,0)', font=dict(size=17)),
        height=720, width=1100, **LAYOUT_BASE,
    )
    save(fig, 'F4_strategy_comparison', width=1100, height=720)


# ─── F5: Feature-group importance ────────────────────────────────
def figure_feature_importance():
    print('F5 — feature-group importance')
    imp = pd.read_csv(FEAT_IMP)
    grp = imp.groupby(['stage', 'group'])['importance_norm'].sum().reset_index()
    GROUP_ORDER = [
        'Wang–Engel–Streck simulator',
        'GDD / vernalization',
        'Daymet meteorology',
        'MODIS LST',
        'HLS phenometrics',
        'HLS windowed statistics',
        'Site and state encoders',
    ]
    # Map raw group names to publication labels
    RENAME = {
        'WES (thermal-time)':   'Wang–Engel–Streck simulator',
        'GDD/Vernalization':    'GDD / vernalization',
        'Daymet':               'Daymet meteorology',
        'MODIS LST':            'MODIS LST',
        'HLS phenometrics':     'HLS phenometrics',
        'HLS windowed':         'HLS windowed statistics',
        'Site/state':           'Site and state encoders',
    }
    grp['group'] = grp['group'].map(RENAME).fillna(grp['group'])
    pv = grp.pivot_table(index='stage', columns='group',
                         values='importance_norm', fill_value=0)
    pv = pv.reindex(STAGES_ORDER).reindex(columns=[g for g in GROUP_ORDER if g in pv.columns])

    z = (pv.values * 100).round(0)
    text = [[f'{v:.0f}' if v >= 1 else '' for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(pv.columns),
        y=[STAGE_LABEL[s] for s in pv.index],
        colorscale=[[0, '#FFFFFF'], [0.4, '#E8DCBC'], [0.8, '#A38847'], [1.0, '#5A4520']],
        zmin=0, zmax=60,
        colorbar=dict(title=dict(text='Importance<br>(% of total)', side='right',
                                  font=dict(size=14)),
                      thickness=18, len=0.75, tickfont=dict(size=13)),
        text=text, texttemplate='%{text}',
        textfont=dict(size=15, color=INK),
        hovertemplate='Stage: %{y}<br>Feature group: %{x}<br>Importance: %{z:.0f}%<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='<b>Importance of feature groups across the eight phenological stages</b><br>'
                        '<span style="font-size:14px;color:#555">'
                        'Normalised feature importance (best model per stage); gain-based for tree '
                        'ensembles, |β̂| for linear models.</span>',
                   x=0.02, xanchor='left', y=0.97, font=dict(size=20)),
        xaxis=dict(tickangle=-25, showline=True, linecolor=INK,
                   tickfont=dict(size=14), side='bottom'),
        yaxis=dict(showline=True, linecolor=INK, tickfont=dict(size=14),
                   autorange='reversed'),
        height=560, width=1050, **LAYOUT_BASE,
    )
    save(fig, 'F5_feature_importance', width=1050, height=560)


# ─── F6: LOSO transferability ────────────────────────────────────
def figure_loso():
    print('F6 — LOSO transferability')
    df = pd.read_csv(LOSO_RES)
    pv = df.pivot_table(index='stage', columns='state', values='R2')
    pv = pv.reindex(STAGES_ORDER)[STATES_ORDER]

    z = pv.values
    text = [[f'{v:.2f}' if not pd.isna(v) else '—' for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f'{s}<br><span style="font-size:11px;color:#666">held out</span>' for s in STATES_ORDER],
        y=[STAGE_LABEL[s] for s in pv.index],
        colorscale=[[0, '#9C2424'], [0.3, '#E0B5B5'], [0.5, '#FFFFFF'],
                    [0.7, '#A3C9BE'], [1.0, '#1B6E63']],
        zmin=-0.2, zmax=0.85,
        colorbar=dict(title=dict(text='R² (LOSO)', side='right', font=dict(size=14)),
                      thickness=18, len=0.75, tickfont=dict(size=13)),
        text=text, texttemplate='%{text}',
        textfont=dict(size=16, color=INK),
        hovertemplate='Stage: %{y}<br>Held-out state: %{x}<br>R²: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='<b>Leave-one-state-out transferability across the Hard Red Winter belt</b><br>'
                        '<span style="font-size:14px;color:#555">'
                        'Models trained on the four remaining states and evaluated on the held-out state.</span>',
                   x=0.02, xanchor='left', y=0.97, font=dict(size=20)),
        xaxis=dict(showline=True, linecolor=INK, tickfont=dict(size=14)),
        yaxis=dict(showline=True, linecolor=INK, tickfont=dict(size=14),
                   autorange='reversed'),
        height=600, width=900, **LAYOUT_BASE,
    )
    save(fig, 'F6_loso_transferability', width=900, height=600)


# ─── F7: Phenology trends, all stages ────────────────────────────
def figure_phenology_trends():
    print('F7 — phenology trends, all stages')
    preds = pd.read_parquet(EXT_PREDS)
    trends = pd.read_csv(TRENDS)
    PLAUS = {'emergence': (260, 340), 'tillering': (30, 110), 'jointing': (50, 120),
             'flag_leaf': (90, 140), 'boot': (95, 150), 'heading': (100, 160),
             'anthesis': (110, 175), 'maturity': (150, 210)}

    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=[f'<b>{STAGE_LABEL[s]}</b>' for s in STAGES_ORDER],
                        horizontal_spacing=0.07, vertical_spacing=0.22,
                        shared_xaxes=True)

    legend_states_done = set()
    rng = np.random.RandomState(42)
    for k, stage in enumerate(STAGES_ORDER):
        r, c = k // 4 + 1, k % 4 + 1
        col = f'{stage}_doy_pred'
        lo, hi = PLAUS[stage]
        sub = preds[preds[col].between(lo, hi)]
        years = np.array(sorted(sub['year'].unique()), dtype=float)
        y_grid = np.linspace(years.min(), years.max(), 60)

        for st in STATES_ORDER:
            sst = sub[sub['state'] == st]
            if len(sst) < 30:
                continue
            x = sst['year'].values.astype(float)
            y = sst[col].values.astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            # 500-resample bootstrap CI on the regression line for the ribbon
            boots = []
            n = len(x)
            for _ in range(400):
                idx = rng.choice(n, n, replace=True)
                s, i = np.polyfit(x[idx], y[idx], 1)
                boots.append(s * y_grid + i)
            boots = np.asarray(boots)
            lo_band = np.percentile(boots, 2.5, axis=0)
            hi_band = np.percentile(boots, 97.5, axis=0)
            line_y = slope * y_grid + intercept

            color = STATE_COLOR.get(st, '#888888')
            r_rgb, g_rgb, b_rgb = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
            fill = f'rgba({r_rgb},{g_rgb},{b_rgb},0.13)'
            show_legend = st not in legend_states_done
            legend_states_done.add(st)

            # Ribbon (upper then lower with fill='tonexty')
            fig.add_trace(go.Scatter(
                x=y_grid, y=hi_band, mode='lines',
                line=dict(color=color, width=0), showlegend=False, hoverinfo='skip',
                legendgroup=st,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=y_grid, y=lo_band, mode='lines',
                line=dict(color=color, width=0), showlegend=False, hoverinfo='skip',
                fill='tonexty', fillcolor=fill, legendgroup=st,
            ), row=r, col=c)
            # OLS line
            fig.add_trace(go.Scatter(
                x=y_grid, y=line_y, mode='lines',
                line=dict(color=color, width=3),
                name=st, legendgroup=st, showlegend=show_legend,
                hovertemplate=f'{st}<br>Year %{{x:.0f}}<br>Predicted DOY %{{y:.1f}}<extra></extra>',
            ), row=r, col=c)
            # Annual means as faint markers (low-key, visual anchor without clutter)
            agg = sst.groupby('year')[col].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=agg['year'], y=agg[col], mode='markers',
                marker=dict(size=7, color=color, opacity=0.55,
                            line=dict(color='white', width=0.6)),
                showlegend=False, hoverinfo='skip', legendgroup=st,
            ), row=r, col=c)

        fig.update_xaxes(row=r, col=c, **AXIS_BASE,
                         tickmode='array', tickvals=list(range(2018, 2025)),
                         range=[2017.7, 2024.3])
        fig.update_yaxes(row=r, col=c, **AXIS_BASE)
        if r == 2:
            fig.update_xaxes(title_text='Harvest year', row=r, col=c,
                             title_font=dict(size=18))
        if c == 1:
            fig.update_yaxes(title_text='Predicted day of year', row=r, col=c,
                             title_font=dict(size=18))

    fig.update_layout(
        title=dict(text='<b>Trends in predicted phenology timing across the Hard Red Winter belt, 2018–2024</b><br>'
                        '<span style="font-size:16px;color:#555">'
                        'Per-state ordinary least-squares trend (solid) with 95 % bootstrap confidence band; '
                        'small dots are annual state means.</span>',
                   x=0.02, xanchor='left', y=0.98, font=dict(size=22)),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.16,
                    bgcolor='rgba(255,255,255,0)', font=dict(size=18)),
        height=820, width=1300,
        margin=dict(l=85, r=35, t=110, b=140),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Helvetica, Arial, sans-serif', size=20, color=INK),
    )
    for ann in fig['layout']['annotations'][:8]:
        ann['font'] = dict(size=20, color=INK)
    save(fig, 'F7_phenology_trends', width=1300, height=820)


# ─── F7 companion: trend-slope summary heatmap ────────────────────
def figure_trend_summary():
    print('F8 — trend-slope summary heatmap')
    trends = pd.read_csv(TRENDS)
    pv = trends.pivot_table(index='stage', columns='state',
                            values='slope_d_per_yr').reindex(STAGES_ORDER)[STATES_ORDER]
    sig = trends.copy()
    sig['is_sig'] = (sig['ci_lo'] > 0) | (sig['ci_hi'] < 0)
    sig_mark = sig.pivot_table(index='stage', columns='state',
                               values='is_sig').reindex(STAGES_ORDER)[STATES_ORDER]

    z = pv.values
    text = []
    for i in range(z.shape[0]):
        row = []
        for j in range(z.shape[1]):
            v = z[i, j]
            if pd.isna(v):
                row.append('—')
            else:
                star = '*' if sig_mark.iloc[i, j] == True else ''
                row.append(f'{v:+.2f}{star}')
        text.append(row)

    vmax = np.nanmax(np.abs(z))
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=STATES_ORDER,
        y=[STAGE_LABEL[s] for s in pv.index],
        colorscale=[[0, '#1B4F8C'], [0.5, '#FFFFFF'], [1.0, '#A03939']],
        zmin=-vmax, zmax=vmax,
        colorbar=dict(title=dict(text='Slope<br>(days yr⁻¹)', side='right',
                                  font=dict(size=14)),
                      thickness=18, len=0.75, tickfont=dict(size=13)),
        text=text, texttemplate='%{text}',
        textfont=dict(size=16, color=INK),
        hovertemplate='Stage: %{y}<br>State: %{x}<br>Slope: %{z:+.2f} d yr⁻¹<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='<b>Linear trends in predicted phenology timing, 2018–2024</b><br>'
                        '<span style="font-size:14px;color:#555">'
                        'Slope of predicted day-of-year against harvest year, per state and stage. '
                        '* denotes 95 % bootstrap CI excluding zero.</span>',
                   x=0.02, xanchor='left', y=0.97, font=dict(size=20)),
        xaxis=dict(showline=True, linecolor=INK, tickfont=dict(size=15)),
        yaxis=dict(showline=True, linecolor=INK, tickfont=dict(size=14),
                   autorange='reversed'),
        height=620, width=880, **LAYOUT_BASE,
    )
    save(fig, 'F8_trend_summary', width=880, height=620)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_per_stage_scatter()
    figure_strategy_comparison()
    figure_feature_importance()
    figure_loso()
    figure_phenology_trends()
    figure_trend_summary()


if __name__ == '__main__':
    main()
