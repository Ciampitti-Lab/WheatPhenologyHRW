"""R14g - Regenerate Figures 4 and 6 from the corrected-state results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

REV = Path('/home/vmangidi/repositories/WheatPhenologyHRW/data/revision')
OUT = Path('/home/vmangidi/repositories/paper-overleaf/figures')
ORDER = ['emergence','tillering','jointing','flag_leaf','boot','heading','anthesis','maturity']
LAB = {'emergence':'Emergence','tillering':'Tillering','jointing':'Jointing',
       'flag_leaf':'Flag Leaf','boot':'Boot','heading':'Heading',
       'anthesis':'Anthesis','maturity':'Maturity'}
STATES = ['TX','OK','KS','NE','CO']


def figure4():
    t = pd.read_csv(REV/'R14_strategy_tally.csv').set_index('stage').reindex(ORDER)
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.5),
                           gridspec_kw={'width_ratios':[2.5,1]})
    x = np.arange(len(ORDER)); w = 0.38
    a = ax[0]
    a.bar(x-w/2, t['B_ML-only'], w, color='#9a9a9a', label='ML-only (no WES)')
    a.bar(x+w/2, t['C_Hybrid'], w, color='#8c6d3f', label='Hybrid (WES + ML)')
    for i, d in enumerate(t.delta):
        top = max(t['B_ML-only'].iloc[i], t['C_Hybrid'].iloc[i])
        mark = '▲' if d > 0.005 else ('▼' if d < -0.005 else '')
        col = '#2e7d5b' if d > 0.005 else ('#b3402f' if d < -0.005 else '#666666')
        a.text(i, top+0.022, f'{mark}{abs(d):.2f}' if mark else f'≈{abs(d):.2f}',
               ha='center', fontsize=8, color=col, fontweight='bold')
    a.set_xticks(x); a.set_xticklabels([LAB[s] for s in ORDER], rotation=28, ha='right')
    a.set_ylabel('LOYO $R^2$'); a.set_ylim(0, 1.0)
    a.set_title('A. Best model per strategy, by stage', loc='left',
                fontsize=11, fontweight='bold')
    a.legend(fontsize=9, frameon=False, loc='upper left')
    a.grid(axis='y', ls=':', color='#cccccc', lw=0.6); a.set_axisbelow(True)
    for sp in ('top','right'): a.spines[sp].set_visible(False)

    b = ax[1]
    c = t.winner.value_counts()
    cats = ['Hybrid','ML-only','tie']
    vals = [int(c.get(k,0)) for k in cats]
    cols = ['#8c6d3f','#9a9a9a','#c9c9c9']
    b.bar(range(3), vals, color=cols, width=0.62)
    for i,v in enumerate(vals):
        b.text(i, v+0.08, str(v), ha='center', fontweight='bold', fontsize=11)
    b.set_xticks(range(3))
    b.set_xticklabels(['Hybrid\nhigher','ML-only\nhigher','Tie'], fontsize=9)
    b.set_ylabel('Stages (of 8)'); b.set_ylim(0, 7)
    b.set_title(f'B. Hybrid higher in {vals[0]}/8\n(mean $\\Delta R^2$='
                f'{t.delta.mean():+.2f})', loc='left', fontsize=11, fontweight='bold')
    for sp in ('top','right'): b.spines[sp].set_visible(False)
    fig.tight_layout()
    for e in ('pdf','png'):
        fig.savefig(OUT/f'F4_strategy_comparison.{e}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    print('F4:', dict(zip(cats, vals)))


def figure6():
    L = pd.read_csv(REV/'R14_loso_final.csv')
    m = L.pivot_table(index='stage', columns='state', values='R2').reindex(ORDER)[STATES]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    clip = m.clip(-1, 1)
    im = ax.imshow(clip.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(ORDER)):
        for j, st in enumerate(STATES):
            v = m.iloc[i, j]
            if pd.isna(v):
                continue
            txt = '$<\\,-1.00$' if v < -1 else f'{v:.2f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8.5,
                    fontweight='bold',
                    color='white' if abs(clip.iloc[i, j]) > 0.55 else '#222222')
    ax.set_xticks(range(len(STATES))); ax.set_xticklabels(STATES)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LAB[s] for s in ORDER])
    ax.set_xlabel('Held-out state')
    ax.set_title('Leave-one-state-out transferability ($R^2$;\nadopted model per stage)',
                 fontsize=11, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('$R^2$ (clipped to $[-1, 1]$)')
    fig.tight_layout()
    for e in ('pdf','png'):
        fig.savefig(OUT/f'F6_loso_transferability.{e}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    print('F6 written')
    print(m.round(3).to_string())


if __name__ == '__main__':
    figure4(); figure6()
