"""R18 - Regenerate Supplementary Table S1, the full results grid.

The submitted S1 was written from the pre-correction run: 67 of its 112 R^2
cells disagree with data/revision/R14_grid_full.csv, and its asterisk sat on
the tillering ElasticNet cell that audit item 14 replaced with Ridge. Nothing
in the verification harness read it, because it is the only table in the paper
that reports the whole grid rather than a derived quantity.

Emits the tabular body only, for splicing between \\midrule and \\end{longtable}
in sections/supplementary.tex. The asterisk marks the adopted cell of ADOPT.

Output: data/revision/R18_supp_grid_rows.tex
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from scripts.utils.deep_models import ADOPT, ORDER, MODELS

REV = Path(__file__).resolve().parents[2] / 'data' / 'revision'
LABEL = {'FT': 'FT-Transformer'}
STAGE = {s: s.replace('_', ' ').title() for s in ORDER}


def fmt(v, adopted):
    t = f'{v:.2f}'
    return f'{t}$^{{\\ast}}$' if adopted else t


def main():
    g = pd.read_csv(REV / 'R14_grid_full.csv').set_index(['stage', 'strategy', 'model'])
    out = []
    for i, s in enumerate(ORDER):
        if i:
            out.append('\\midrule')
        for j, m in enumerate(MODELS):
            cells = []
            for arm in ('B_ML-only', 'C_Hybrid'):
                r = g.loc[(s, arm, m)]
                cells += [fmt(r.R2, ADOPT[s] == (arm, m)), f'{r.RMSE:.1f}']
            out.append(f'{STAGE[s] if j == 0 else "":10s} & {LABEL.get(m, m):15s} & '
                       + ' & '.join(f'{c:>14s}' for c in cells) + ' \\\\')
    p = REV / 'R18_supp_grid_rows.tex'
    p.write_text('\n'.join(out) + '\n')
    print(f'-> {p}  ({len(ORDER) * len(MODELS)} rows, '
          f'{sum(1 for s in ORDER)} stages x {len(MODELS)} models)')
    print('\n'.join(out[:9]))


if __name__ == '__main__':
    main()
