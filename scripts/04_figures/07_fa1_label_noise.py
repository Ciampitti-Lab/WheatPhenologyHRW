"""Figure A1: within-field-year scatter of vegetative-stage onset.

For each training (field, harvest-year) with >=2 same-stage
observations inside a 21 d span, take the SD of the recorded onset DOS,
and plot its distribution for the three vegetative stages. Runs off the
public data_public/ subset.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.utils.config import CFG

LABELS = REPO / 'data_public' / 'processed' / 'phenology_labels.parquet'
OUT = REPO / 'docs' / 'figures'

# The three vegetative stages, drawn from the canonical vocabulary.
VEG = ['tillering', 'jointing', 'emergence']
VOCAB = {s: list(CFG.phenology_stages[s]) for s in VEG}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ph = pd.read_parquet(LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, stage in zip(axes, VEG):
        sub = ph[ph['growth_stage'].isin(VOCAB[stage])]
        g = (sub.groupby(['field_id', 'harvest_year'])['dos']
                .agg(['count', 'std', 'min', 'max']))
        w = g[(g['count'] >= 2) & (g['max'] - g['min'] <= 21)]
        sd = w['std'].dropna()
        m = sd.mean()
        ax.hist(sd, bins=range(0, 22, 1), color='#5d4a86',
                edgecolor='white', linewidth=0.4)
        ax.axvline(m, ls='--', color='#9e3b3b', lw=1.5)
        ax.annotate(f'mean = {m:.1f} d', xy=(m, 0.93), xycoords=('data',
                    'axes fraction'), ha='left', va='top',
                    fontsize=9, color='#9e3b3b',
                    xytext=(4, 0), textcoords='offset points')
        ax.set_title(f'{stage.capitalize()} '
                     f'(n = {len(sd):,} field-years)', fontsize=10)
        ax.set_xlabel('Within-field-year SD of\nrecorded stage-onset day (d)')
        ax.set_xlim(0, 21)
    axes[0].set_ylabel('Field-years')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'FA1_label_noise.{ext}', dpi=200,
                    bbox_inches='tight')
    print(f'-> {(OUT / "FA1_label_noise.png").relative_to(REPO)} (+ .pdf)')
    for stage in VEG:
        sub = ph[ph['growth_stage'].isin(VOCAB[stage])]
        g = (sub.groupby(['field_id', 'harvest_year'])['dos']
                .agg(['count', 'std', 'min', 'max']))
        w = g[(g['count'] >= 2) & (g['max'] - g['min'] <= 21)]
        print(f'   {stage:10s} n={len(w.dropna()):4d}  '
              f'mean within-FY SD = {w["std"].mean():.1f} d')


if __name__ == '__main__':
    main()
