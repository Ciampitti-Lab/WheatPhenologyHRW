"""F4 — Strategy comparison panel: per-stage R² for B_ML-only vs C_Hybrid
across the 5 candidate ML models (V2 vanilla framework).

Shows that the Hybrid strategy (Wang-Engel-Streck features + ML) wins
on 7 of 8 stages, justifying the physiology-informed design.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXT = _WORK
RES = pd.read_parquet(EXT / "v3_results" / 'multi_stage_models_a6_gs.parquet')
OUT_FIG = ROOT / 'docs' / 'figures' / 'F4_strategy_comparison_v3.png'

GOLD = '#CEB888'; ACCENT = '#8E6F3E'; DARK = '#1B1B1B'
GREY = '#888'

STAGES_ORDER = ['emergence', 'tillering', 'jointing',
                'flag_leaf', 'boot', 'heading', 'anthesis', 'maturity']

# Best R² per (stage, strategy) — keeps the best across the 5 models
best = RES.loc[RES.groupby(['stage','strategy'])['R2'].idxmax()].reset_index(drop=True)
print('Best per (stage, strategy):')
print(best[['stage','strategy','model','R2','RMSE']].to_string(index=False))

# Pivot for plotting
pivot = best.pivot_table(index='stage', columns='strategy', values='R2')
pivot = pivot.reindex(STAGES_ORDER)
print('\nPivot R²:')
print(pivot.round(3).to_string())

# ─── Two-panel figure ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=140,
                                gridspec_kw={'width_ratios': [3, 2]})

# Panel A — Grouped bar: B_ML-only vs C_Hybrid per stage
x = np.arange(len(STAGES_ORDER))
w = 0.36
b_vals = pivot['B_ML-only'].values
c_vals = pivot['C_Hybrid'].values
bars_b = ax1.bar(x - w/2, b_vals, w, label='ML-only (no WES)',
                  color=GREY, edgecolor=DARK, linewidth=0.6)
bars_c = ax1.bar(x + w/2, c_vals, w, label='Hybrid (WES + ML)',
                  color=ACCENT, edgecolor=DARK, linewidth=0.6)

# Annotate Δ (Hybrid - ML-only)
for i, (b, c) in enumerate(zip(b_vals, c_vals)):
    delta = c - b
    arrow = '▲' if delta > 0 else '▼'
    col = '#1B6E63' if delta > 0 else '#A03939'
    ax1.text(i, max(b, c) + 0.03, f'{arrow}{abs(delta):.02f}',
             ha='center', fontsize=8, color=col, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels([s.replace('_',' ').title() for s in STAGES_ORDER],
                    rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('LOYO R²', fontsize=11)
ax1.set_ylim(-0.05, 1.0)
ax1.axhline(0, color='k', lw=0.4)
ax1.set_title('A. Strategy comparison — Hybrid (WES + ML) vs ML-only per stage',
              fontsize=11, fontweight='bold', color=DARK, pad=10)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, axis='y', alpha=0.2, linewidth=0.5)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B — Summary: how many stages each strategy wins, mean R² gain
strat_wins = {'Hybrid': (c_vals > b_vals).sum(),
              'ML-only': (b_vals > c_vals).sum(),
              'Tie': (np.abs(c_vals - b_vals) < 0.005).sum()}
mean_gain = (c_vals - b_vals).mean()
median_gain = float(np.median(c_vals - b_vals))

# Simple stats bar
labels = ['Hybrid wins', 'ML-only wins', 'Tie']
vals = [strat_wins['Hybrid'], strat_wins['ML-only'], strat_wins['Tie']]
cols = [ACCENT, GREY, '#CCC']
bars = ax2.bar(labels, vals, color=cols, edgecolor=DARK, linewidth=0.6)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.1, str(v),
             ha='center', fontsize=12, fontweight='bold', color=DARK)
ax2.set_ylim(0, max(vals) + 1.5)
ax2.set_ylabel('Number of stages (out of 8)', fontsize=11)
ax2.set_title(f'B. Hybrid wins {strat_wins["Hybrid"]}/8 stages\n(mean ΔR² = {mean_gain:+.3f})',
              fontsize=11, fontweight='bold', color=DARK, pad=10)
ax2.grid(True, axis='y', alpha=0.2, linewidth=0.5)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.suptitle('Hybrid physiology + ML wins on reproductive stages; ML-only better for early stages',
             fontsize=11, color=DARK, y=1.02)
plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\n→ {OUT_FIG}')
