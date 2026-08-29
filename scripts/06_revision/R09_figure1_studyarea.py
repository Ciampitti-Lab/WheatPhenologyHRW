"""R09 - Redraw Figure 1, the study-area map  (reviewer #6, minor 4).

Reviewer #6: "Figure 1 is overly simple and would benefit from clearer
geographic context and more complete map elements."

The submitted figure was a scatter of field centroids over an unlabelled
state outline, with no projection statement, no scale, no orientation and
no wider locator. This version adds: an equal-area projection appropriate
to the belt, labelled state polygons, a graticule, a scale bar, a north
arrow, an inset locating the study area within the conterminous US, and a
legend carrying the per-state field counts, which make the Kansas
concentration -- central to the transferability discussion -- legible
directly from the figure.

Needs the geospatial environment:
  /depot/ciampitti/apps/envs/vmangidi_ww_protein_prediction/bin/python

Output: paper-overleaf/figures/F1_study_area.{pdf,png}
"""
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.lines import Line2D


WORK = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
            'extension_2018_2024')
OUT = Path('/home/vmangidi/repositories/paper-overleaf/figures')
SHP = Path('/home/vmangidi/.local/share/cartopy/shapefiles/natural_earth/'
           'cultural/ne_50m_admin_1_states_provinces.shp')

STATE_COLOR = {'TX': '#c2543a', 'OK': '#d9a441', 'KS': '#3f7d4e',
               'NE': '#3b6ea5', 'CO': '#7a5ba6'}
ORDER = ['TX', 'OK', 'KS', 'NE', 'CO']
FULL = {'TX': 'Texas', 'OK': 'Oklahoma', 'KS': 'Kansas',
        'NE': 'Nebraska', 'CO': 'Colorado'}
PROJ = ccrs.AlbersEqualArea(central_longitude=-99, central_latitude=37.5,
                            standard_parallels=(33, 45))


def scale_bar(ax, length_km=200, loc=(0.08, 0.06)):
    """Scale bar drawn in projected (metre) coordinates."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + (x1 - x0) * loc[0]
    by = y0 + (y1 - y0) * loc[1]
    L = length_km * 1000
    ax.plot([bx, bx + L], [by, by], color='black', lw=2.4,
            solid_capstyle='butt', zorder=12)
    ax.plot([bx, bx + L / 2], [by, by], color='white', lw=1.4,
            solid_capstyle='butt', zorder=13)
    for xx, lab in [(bx, '0'), (bx + L, f'{length_km} km')]:
        ax.text(xx, by + (y1 - y0) * 0.012, lab, ha='center', va='bottom',
                fontsize=7.5, zorder=13)


def north_arrow(ax, loc=(0.94, 0.12)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * loc[0]
    y = y0 + (y1 - y0) * loc[1]
    dy = (y1 - y0) * 0.055
    ax.annotate('', xy=(x, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.3),
                zorder=13)
    ax.text(x, y + dy * 1.15, 'N', ha='center', va='bottom', fontsize=9,
            fontweight='bold', zorder=13)


def main():
    # State comes from the spatial join of R14a, not the latitude box the
    # pipeline originally used, which handed the Texas Panhandle to Oklahoma.
    look = pd.read_csv(Path('/home/vmangidi/repositories/WheatPhenologyHRW/'
                            'data/revision/R14_state_lookup.csv'))
    look['field_id'] = look['field_id'].astype(str)
    ft = pd.read_parquet(WORK / 'features_v3_realsowing_train.parquet',
                         columns=['field_id', 'latitude', 'longitude'])
    ft['field_id'] = ft['field_id'].astype(str)
    ft = ft.merge(look[['field_id', 'state_true']], on='field_id', how='left')
    ft = ft.rename(columns={'state_true': 'state'})
    # New Mexico is shown with Texas, as the nearest study state; the three
    # centroids outside the study area entirely are excluded.
    ft['state'] = ft['state'].replace({'NM': 'TX'})
    ft = ft[ft['state'].isin(ORDER)]
    fld = (ft.groupby(['field_id', 'state'])[['latitude', 'longitude']]
             .mean().reset_index())
    counts = fld.groupby('state')['field_id'].nunique().to_dict()
    total = fld['field_id'].nunique()
    print(f'{total} unique fields: ' +
          ', '.join(f'{s} {counts.get(s, 0)}' for s in ORDER))

    fig = plt.figure(figsize=(9.2, 7.0))
    ax = fig.add_axes([0.03, 0.06, 0.80, 0.88], projection=PROJ)
    ax.set_extent([-107.5, -94.0, 31.0, 43.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f4f2ee')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#dce6ee')
    ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='#dce6ee',
                   edgecolor='#9fb4c4', lw=0.3)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), edgecolor='#9fb4c4',
                   lw=0.4)

    reader = shpreader.Reader(str(SHP))
    for rec in reader.records():
        if rec.attributes.get('admin') != 'United States of America':
            continue
        pc = rec.attributes.get('postal')
        face = (STATE_COLOR[pc] if pc in STATE_COLOR else 'none')
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                          facecolor=face, alpha=0.13 if pc in STATE_COLOR else 0,
                          edgecolor='#4a4a4a', linewidth=0.7, zorder=2)
        if pc in STATE_COLOR:
            c = rec.geometry.centroid
            ax.text(c.x, c.y, FULL[pc], transform=ccrs.PlateCarree(),
                    fontsize=10.5, fontweight='bold', color='#333333',
                    ha='center', va='center', alpha=0.55, zorder=3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#b9b9b9',
                      alpha=0.6, linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.rotate_labels = False          # Albers otherwise stacks them vertically
    gl.xlabel_style = gl.ylabel_style = {'size': 8, 'color': '#555555'}

    for s in ORDER:
        d = fld[fld.state == s]
        ax.scatter(d.longitude, d.latitude, transform=ccrs.PlateCarree(),
                   s=5.5, c=STATE_COLOR[s], edgecolor='none', alpha=0.75,
                   zorder=6, rasterized=True)

    scale_bar(ax)
    north_arrow(ax)

    handles = [Line2D([], [], marker='o', ls='none', ms=6.5,
                      mfc=STATE_COLOR[s], mec='none',
                      label=f'{FULL[s]} ($n = {counts.get(s, 0):,}$)')
               for s in ORDER]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 0.98),
              frameon=True, framealpha=0.95, edgecolor='#cccccc',
              fontsize=9, title='Fields with ground\nphenology observations',
              title_fontsize=9.5, borderpad=0.8, labelspacing=0.7)

    # locator inset
    inset = fig.add_axes([0.845, 0.10, 0.145, 0.20],
                         projection=ccrs.AlbersEqualArea(
                             central_longitude=-96, central_latitude=37.5))
    inset.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    inset.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#eeece8')
    inset.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor='#dce6ee')
    inset.add_feature(cfeature.STATES.with_scale('110m'),
                      edgecolor='#b0b0b0', lw=0.25)
    inset.add_feature(cfeature.COASTLINE.with_scale('110m'),
                      edgecolor='#8a8a8a', lw=0.4)
    for rec in shpreader.Reader(str(SHP)).records():
        if (rec.attributes.get('admin') == 'United States of America'
                and rec.attributes.get('postal') in STATE_COLOR):
            inset.add_geometries([rec.geometry], ccrs.PlateCarree(),
                                 facecolor='#c2543a', alpha=0.65,
                                 edgecolor='none', zorder=4)
    inset.set_title('Study area', fontsize=8, pad=2)

    ax.set_title('The U.S. Hard Red Winter wheat belt study area',
                 fontsize=13, fontweight='bold', pad=12)
    ax.text(0.5, -0.105,
            f'{total:,} winter-wheat fields with at least one ground phenology '
            f'observation, 2013/14–2016/17.\n'
            'Albers equal-area conic projection, standard parallels '
            '33°N and 45°N.',
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5,
            color='#555555')

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'F1_study_area.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    print(f'wrote {OUT}/F1_study_area.{{pdf,png}}')


if __name__ == '__main__':
    main()
