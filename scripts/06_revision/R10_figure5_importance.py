"""R10 - Redraw Figure 5 from the uniform grouped permutation importance
        (reviewer #6, minor 3).

Replaces the submitted figure, which normalized split gain, |beta-hat| and
permutation importance onto one axis. Two panels:

  A  absolute loss in LOYO R^2 when each source group is permuted, which
     keeps the measure on the same scale as every other number in the
     paper and makes stages comparable to one another;
  B  the same as a share of total positive importance per stage, which is
     what the submitted figure attempted to show.

Reads data/revision/R08_grouped_permutation.csv (produced by R08).

Output: paper-overleaf/figures/F5_feature_importance.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path('/home/vmangidi/repositories/WheatPhenologyHRW/data/revision/'
           'R08_grouped_permutation.csv')
OUT = Path('/home/vmangidi/repositories/paper-overleaf/figures')

ORDER = ['emergence', 'tillering', 'jointing', 'flag_leaf', 'boot',
         'heading', 'anthesis', 'maturity']
LABEL = {'emergence': 'Emergence', 'tillering': 'Tillering',
         'jointing': 'Jointing', 'flag_leaf': 'Flag leaf', 'boot': 'Boot',
         'heading': 'Heading', 'anthesis': 'Anthesis', 'maturity': 'Maturity'}
GROUPS = ['WES', 'HLS', 'Daymet', 'ThermalTime', 'LST', 'Site', 'State']
GNAME = {'WES': 'WES simulator', 'HLS': 'HLS phenometrics',
         'Daymet': 'Daymet meteorology', 'ThermalTime': 'Thermal-time state',
         'LST': 'MODIS LST', 'Site': 'Site geometry',
         'State': 'State encoders'}
COLOR = {'WES': '#8c6d3f', 'HLS': '#d9c79f', 'Daymet': '#6b9e78',
         'ThermalTime': '#a05f8e', 'LST': '#4f7fa8', 'Site': '#c2543a',
         'State': '#b9b9b9'}


def main():
    d = pd.read_csv(SRC)
    imp = d.pivot_table(index='stage', columns='group',
                        values='drop_mean').reindex(ORDER)[GROUPS]
    err = d.pivot_table(index='stage', columns='group',
                        values='drop_sd').reindex(ORDER)[GROUPS]
    pos = imp.clip(lower=0)
    share = 100 * pos.div(pos.sum(axis=1), axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 9.2),
                             gridspec_kw={'height_ratios': [1.3, 1]})
    y = np.arange(len(ORDER))[::-1]

    # --- Panel A: absolute loss, grouped bars -----------------------------
    ax = axes[0]
    h = 0.118
    for k, yy in enumerate(y):
        if k % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color='#f2f0ec', zorder=0)
    for i, g in enumerate(GROUPS):
        off = (i - (len(GROUPS) - 1) / 2) * h
        ax.barh(y + off, imp[g].values, height=h, color=COLOR[g],
                edgecolor='white', linewidth=0.3, label=GNAME[g],
                xerr=err[g].values, error_kw=dict(lw=0.6, ecolor='#555555'))
    ax.axvline(0, color='#444444', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[s] for s in ORDER])
    ax.set_xlabel(r'Loss in LOYO $R^2$ when the group is permuted')
    ax.set_title('A. Absolute importance', loc='left', fontsize=11,
                 fontweight='bold')
    ax.grid(axis='x', ls=':', color='#cccccc', lw=0.6)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # --- Panel B: share, stacked ------------------------------------------
    ax = axes[1]
    left = np.zeros(len(ORDER))
    for g in GROUPS:
        v = share[g].values
        ax.barh(y, v, left=left, height=0.62, color=COLOR[g],
                edgecolor='white', linewidth=0.5, label=GNAME[g])
        for yi, (l, w) in enumerate(zip(left, v)):
            if w >= 8:
                ax.text(l + w / 2, y[yi], f'{w:.0f}', ha='center',
                        va='center', fontsize=7.5,
                        color='white' if g in ('WES', 'Site', 'Daymet')
                        else '#333333')
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[s] for s in ORDER])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of total positive importance (%)')
    ax.set_title('B. Relative composition', loc='left', fontsize=11,
                 fontweight='bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('Grouped permutation importance by input source\n'
                 '(LightGBM, identical protocol at every stage)',
                 fontsize=12.5, fontweight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0.062, 1, 0.955])

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'F5_feature_importance.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    print(f'wrote {OUT}/F5_feature_importance.{{pdf,png}}')
    print(share.round(1).to_string())


if __name__ == '__main__':
    main()
