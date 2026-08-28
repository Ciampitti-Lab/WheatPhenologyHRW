"""R07 - Complete input-feature inventory  (reviewer #6, minor comment 2).

Reviewer #6: "A complete input-feature table would improve clarity,
including data source, temporal window, spatial resolution, and whether
the same feature set is used across all phenological stages."

The table is generated from the live feature selector rather than typed
by hand, so it cannot drift from the code as the previous prose
description did. Emits LaTeX (longtable) for Supplementary Table S1 and
a CSV for the repository.

Output: data/revision/R07_feature_inventory.csv
        data/revision/R07_feature_inventory.tex
"""
from _common import EARLY, cohort, feature_group, save
import pandas as pd

SOURCE = {
    'HLS': ('HLS v2.0 L30/S30', '30 m', 'Full season (1 Jul--31 Jul)'),
    'Daymet': ('Daymet v4', '1 km', 'see window column'),
    'LST': ('MODIS MOD11A2', '1 km', 'see window column'),
    'ThermalTime': ('Daymet v4 (derived)', '1 km', 'Sowing to start of season'),
    'WES': ('Wang--Engel--Streck simulator', 'per field', 'Sowing to maturity'),
    'State': ('USDA state boundary', 'state', 'static'),
    'Site': ('Field centroid', 'per field', 'static'),
}
DESC = {
    'HLS': 'Vegetation-index phenometrics and double-logistic shape terms',
    'Daymet': 'Aggregated daily weather (GDD, frost/heat-day counts, precipitation, radiation)',
    'LST': 'Day- and night-time land-surface temperature aggregates',
    'ThermalTime': 'Thermal-time, vernalization and photoperiod state at start of season',
    'WES': 'Simulated day-of-season for each of the eight stages',
    'State': 'One-hot state indicator',
    'Site': 'Latitude and longitude of the field centroid',
}


def window_of(col):
    if col.endswith('_gf'):
        return 'Grain filling (15 Apr--30 Jun)'
    if col.endswith(('_pa', '_pa_late')):
        return 'Pre-anthesis (1 Mar--14 Apr)'
    if 'winter' in col:
        return 'Winter (Dec--Feb)'
    if 'greenup' in col:
        return 'Green-up'
    if 'pre_SOS' in col or 'at_SOS' in col:
        return 'Sowing to start of season'
    if 'post_anthesis' in col or col.endswith('_pa'):
        return 'Post-anthesis'
    return 'Full season'


def main():
    _, cols = cohort()
    full = cols('heading', True)      # widest set: hybrid, non-early stage
    early = set(cols('emergence', True))

    rows = []
    for c in sorted(full):
        g = feature_group(c)
        src, res, _ = SOURCE[g]
        rows.append(dict(
            feature=c, group=g, source=src, resolution=res,
            window=window_of(c),
            available_to=('all stages' if c in early
                          else 'reproductive + maturity only'),
            withheld_from_early=int(c not in early)))
    df = pd.DataFrame(rows)
    save(df, 'R07_feature_inventory.csv')

    print('=== inventory summary ===')
    s = (df.groupby('group')
           .agg(n=('feature', 'size'),
                withheld=('withheld_from_early', 'sum')).sort_index())
    s['source'] = [SOURCE[g][0] for g in s.index]
    s['resolution'] = [SOURCE[g][1] for g in s.index]
    print(s.to_string())
    print(f'\ntotal features: {len(df)}   '
          f'withheld from {sorted(EARLY)}: {df.withheld_from_early.sum()}')

    # ---- LaTeX (grouped summary table, S1) --------------------------------
    lines = [
        r'\begin{longtable}{p{2.1cm}p{3.4cm}p{1.5cm}p{4.2cm}r}',
        r"\caption{Complete input inventory. ``$n$'' is the number of columns "
        r'contributed by each source to the widest (physiology-informed, '
        r'non-early-stage) feature vector. Window-aggregated weather features '
        r'close after the early stages have already occurred and are withheld '
        r'from the emergence, tillering and jointing models (last column); all '
        r'other features are available to every stage. Feature-level detail is '
        r'in the repository file \texttt{R07\_feature\_inventory.csv}.}\\',
        r'\label{tab:s1-features}\\',
        r'\toprule',
        r'Source & Product & Resolution & Content & $n$ \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'Source & Product & Resolution & Content & $n$ \\',
        r'\midrule',
        r'\endhead',
    ]
    for g in ['HLS', 'Daymet', 'LST', 'ThermalTime', 'WES', 'State', 'Site']:
        sub = df[df.group == g]
        if not len(sub):
            continue
        src, res, _ = SOURCE[g]
        lines.append(f'{g} & {src} & {res} & {DESC[g]} & {len(sub)} \\\\')
    lines += [r'\midrule',
              f'\\textbf{{Total}} & & & & \\textbf{{{len(df)}}} \\\\',
              r'\bottomrule', r'\end{longtable}']
    tex = '\n'.join(lines)
    p = ('/home/vmangidi/repositories/WheatPhenologyHRW/data/revision/'
         'R07_feature_inventory.tex')
    with open(p, 'w') as f:
        f.write(tex + '\n')
    print(f'\n  -> {p}')
    print('\n' + tex)


if __name__ == '__main__':
    main()
