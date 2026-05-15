"""Feature pipeline for 2018-2024 extension years.

Replicates the full A6 feature schema (115 columns) for inference on
the extension dataset, mirroring scripts/02_features/01..07 notebooks
but consolidated into a single inference-only pass over the
(field_id, harvest_year) tuples that survived the CDL wheat filter.

Output: extension_2018_2024/features_extension_2018_2024.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Bring in the production utils
ROOT = Path('/home/vmangidi/repositories/WheatPhenologyHRW')
sys.path.insert(0, str(ROOT))
from scripts.utils.thermal import (simulate_wes, beta_temp_response,
                                   streck_fV, gdd_method2_daily)
from scripts.utils.features import (smooth_vi, extract_phenometrics,
                                    fit_double_logistic, photoperiod_hours)

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
HLS_PATH    = EXT / 'hls_phenology_merged_2018_2024.parquet'
DAYMET_PATH = EXT / 'daymet_full_2018_2024.parquet'
LST_PATH    = EXT / 'modis_lst_2018_2024.parquet'
VALID_PATH  = EXT / 'valid_field_years_2019_2024.parquet'
OUT         = EXT / 'features_extension_2019_2024.parquet'


# ─── helper: state inference from (lat, lon) ─────────────────────────────────
# Same simple bounding-box rule used in 07_spatial_features.ipynb.
def infer_state(lat, lon):
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    # Rough rectangles for the HRW belt — TX/OK/KS/NE/CO/NM
    if lon < -103.5 and lat < 37.0:    return 'NM'
    if lon < -103.5:                   return 'CO'
    if lat < 34.5:                     return 'TX'
    if lat < 37.0:                     return 'OK'
    if lat < 40.0:                     return 'KS'
    return 'NE'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='If >0, process only the first N field-years (smoke test).')
    ap.add_argument('--out', type=str, default=str(OUT),
                    help='Override output parquet path.')
    ap.add_argument('--valid-path', type=str, default=str(VALID_PATH),
                    help='Override valid-field-years parquet path.')
    ap.add_argument('--daymet-path', type=str, default=str(DAYMET_PATH),
                    help='Override Daymet parquet path.')
    ap.add_argument('--lst-path', type=str, default=str(LST_PATH),
                    help='Override MODIS LST parquet path.')
    args = ap.parse_args()
    valid_path  = Path(args.valid_path)
    daymet_path = Path(args.daymet_path)
    lst_path    = Path(args.lst_path)

    t0 = time.time()
    print('=== Loading inputs ===')
    valid = pd.read_parquet(valid_path)
    valid['field_id'] = valid['field_id'].astype(str)
    if args.limit > 0:
        valid = valid.head(args.limit).copy()
        print(f'[SMOKE] limited to first {len(valid):,} field-years')
    print(f'Valid field-years: {len(valid):,}')

    hls = pd.read_parquet(HLS_PATH)
    hls['field_id'] = hls['field_id'].astype(str)
    print(f'HLS observations:  {len(hls):,}')

    # Load Daymet with float32 + filter to fields in valid set to keep
    # memory under SLURM cgroup limit. We compute per-field-year
    # cumulatives inside the loop rather than globally.
    valid_fields = set(valid['field_id'].unique())
    print(f'Loading Daymet (filtered to {len(valid_fields):,} valid fields)...')
    wx = pd.read_parquet(daymet_path,
                         columns=['FIELDID','date','Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq'])
    wx = wx.rename(columns={'FIELDID': 'field_id'})
    wx['field_id'] = wx['field_id'].astype(str)
    wx = wx[wx['field_id'].isin(valid_fields)].copy()
    for c in ['Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq']:
        wx[c] = wx[c].astype('float32')
    wx['date'] = pd.to_datetime(wx['date'])
    wx['year'] = wx['date'].dt.year.astype('int16')
    wx['doy']  = wx['date'].dt.dayofyear.astype('int16')
    wx['month']= wx['date'].dt.month.astype('int8')
    wx['T_mean'] = ((wx['Tmin'] + wx['Tmax']) / 2.0).astype('float32')
    wx = wx.sort_values(['field_id', 'date']).reset_index(drop=True)
    print(f'Daymet rows:       {len(wx):,}  '
          f'mem ~{wx.memory_usage(deep=True).sum()/1e9:.2f} GB')

    lst = pd.read_parquet(lst_path)
    lst = lst.rename(columns={'FIELDID': 'field_id'})
    lst['field_id'] = lst['field_id'].astype(str)
    lst['date'] = pd.to_datetime(lst['date'])
    lst['cy'] = lst['date'].dt.year
    lst['month'] = lst['date'].dt.month
    lst['gs_start_year'] = np.where(lst['month'] >= 7, lst['cy'], lst['cy'] - 1)
    lst['harvest_year']  = lst['gs_start_year'] + 1
    lst['gs_start_date'] = pd.to_datetime(lst['gs_start_year'].astype(str) + '-07-01')
    lst['dos'] = (lst['date'] - lst['gs_start_date']).dt.days + 1
    print(f'MODIS LST rows:    {len(lst):,}')

    # Per-field-year cumulatives are computed inside the loop, on-demand,
    # so we don't bloat memory with 5 extra cumulative columns × 15.6M rows.

    # Vectorised β-temperature response (used inline below per field-year)
    Tmin_v, Topt_v, Tmax_v = 1.3, 4.9, 15.7
    alpha = np.log(2) / np.log((Tmax_v - Tmin_v) / (Topt_v - Tmin_v))
    xopt = Topt_v - Tmin_v
    def beta_vec(T_arr):
        x = T_arr - Tmin_v
        with np.errstate(invalid='ignore'):
            val = (2 * np.power(x, alpha) * (xopt ** alpha) - np.power(x, 2 * alpha)) / (xopt ** (2 * alpha))
        return np.where((T_arr <= Tmin_v) | (T_arr >= Tmax_v), 0.0,
                        np.clip(val, 0.0, 1.0))

    # Window definitions — same DOS ranges as the training notebooks
    DAYMET_WINDOWS = {
        'srad_greenup_mean':       {'var':'srad', 'agg':'mean', 'dos':(240, 290)},
        'srad_grain_mean':         {'var':'srad', 'agg':'mean', 'dos':(295, 355)},
        'ptq_grain_mean':          {'var':'ptq',  'agg':'mean', 'dos':(295, 355)},
        'swe_max_winter':          {'var':'swe',  'agg':'max',  'dos':(150, 240)},
    }
    EVENT_FEATURES = {
        'days_above_5C_winter':    lambda d: ((d['T_mean']>5) & (d['dos']>=150) & (d['dos']<=220)).sum(),
        'frost_days_pre_jointing': lambda d: ((d['Tmin']<0) & (d['dos']>=200) & (d['dos']<=260)).sum(),
        'heat_days_post_anthesis': lambda d: ((d['Tmax']>30) & (d['dos']>=295) & (d['dos']<=360)).sum(),
        'frost_events_spring':     lambda d: ((d['Tmin']<-5) & (d['dos']>=240) & (d['dos']<=280)).sum(),
    }
    LST_WINDOWS = {
        'lst_day_emergence':    {'col':'lst_day_C',   'agg':'mean', 'dos':(90, 150)},
        'lst_day_winter_mean':  {'col':'lst_day_C',   'agg':'mean', 'dos':(150, 220)},
        'lst_day_jointing':     {'col':'lst_day_C',   'agg':'mean', 'dos':(240, 290)},
        'lst_day_anthesis_max': {'col':'lst_day_C',   'agg':'max',  'dos':(290, 330)},
        'lst_night_winter_min': {'col':'lst_night_C', 'agg':'min',  'dos':(150, 220)},
    }

    # Grain-filling + post-anthesis fixed DOS windows
    GF_WINDOW = (300, 350)   # grain fill ≈ May-Jun
    PA_WINDOW = (295, 360)   # post-anthesis

    print(f'\n=== Iterating over {len(valid):,} field-years ===')
    rows = []
    for i, fy in enumerate(valid.itertuples(index=False)):
        fid, hy, lat, lon = fy.field_id, int(fy.harvest_year), fy.lat, fy.lon

        # HLS for this field × calendar year (harvest_year)
        sub = hls[(hls['field_id'] == fid) & (hls['year'] == hy)].copy()
        if len(sub) < 5:
            continue

        row = {'field_id': fid, 'year': hy, 'flag_true_doy': np.nan,
               'n_obs': int(len(sub)),
               'latitude': lat, 'longitude': lon}

        # ── HLS phenometrics per VI ────────────────────────────────────────
        for vi in ['NDVI', 'EVI', 'GCVI', 'NDRE']:
            if vi not in sub.columns:
                continue
            vsub = sub[sub[vi].notna()][['doy', vi]].copy()
            if len(vsub) < 5:
                row.update({f'{vi}_{k}': np.nan for k in
                    ['base','peak','amplitude','POS','greenup_rate','SOS',
                     'greenup_midpoint','duration_greenup','integrated','LeftShoulder']})
                continue
            doys, sm = smooth_vi(vsub['doy'].values, vsub[vi].values)
            if doys is None:
                continue
            row.update(extract_phenometrics(doys, sm, vi))

        # Use NDVI smoothed for double-logistic + dormancy break + senescence
        ndvi_sub = sub[sub['NDVI'].notna()][['doy', 'NDVI']]
        ndvi_doys, ndvi_sm = (None, None)
        if len(ndvi_sub) >= 5:
            ndvi_doys, ndvi_sm = smooth_vi(ndvi_sub['doy'].values, ndvi_sub['NDVI'].values)

        # ── Beck double logistic (NDVI) ────────────────────────────────────
        if ndvi_sm is not None:
            popt = fit_double_logistic(ndvi_doys, ndvi_sm)
            if popt is not None:
                row['DL_c3_greenup_steepness'] = popt[2]
                row['DL_c4_greenup_midpoint']  = popt[3]
                row['DL_c5_senesc_steepness']  = popt[4]
                row['DL_c6_senesc_midpoint']   = popt[5]
            else:
                for k in ['DL_c3_greenup_steepness','DL_c4_greenup_midpoint',
                          'DL_c5_senesc_steepness','DL_c6_senesc_midpoint']:
                    row[k] = np.nan
        else:
            for k in ['DL_c3_greenup_steepness','DL_c4_greenup_midpoint',
                      'DL_c5_senesc_steepness','DL_c6_senesc_midpoint']:
                row[k] = np.nan

        # ── Thermal-time / vernalisation features at SOS ───────────────────
        sos = row.get('NDVI_SOS', np.nan)
        wx_fy = wx[(wx['field_id'] == fid) & (wx['year'] == hy)].copy()
        wx_fy_pre = wx[(wx['field_id'] == fid) &
                       (wx['date'] >= pd.Timestamp(f'{hy-1}-07-01')) &
                       (wx['date'] <  pd.Timestamp(f'{hy}-07-01'))].copy()

        # Per-field-year derivatives (computed on-demand to keep memory low)
        for sub_w in (wx_fy, wx_fy_pre):
            if len(sub_w) == 0:
                continue
            sub_w['gdd_daily']    = np.maximum(0, sub_w['T_mean'].values)
            sub_w['gdd_m2_daily'] = gdd_method2_daily(
                sub_w['Tmin'].values, sub_w['Tmax'].values, 0, 35)
            sub_w['vd_daily']     = beta_vec(sub_w['T_mean'].values)
            sub_w['gdd_cum']      = sub_w['gdd_daily'].cumsum()
            sub_w['gdd_m2_cum']   = sub_w['gdd_m2_daily'].cumsum()
            sub_w['vd_cum']       = sub_w['vd_daily'].cumsum()
            sub_w['fV']           = streck_fV(sub_w['vd_cum'].values)
            sub_w['gdd_eff_daily']= sub_w['gdd_m2_daily'] * sub_w['fV']
            sub_w['gdd_eff_cum']  = sub_w['gdd_eff_daily'].cumsum()
            # DOS = days since Jul 1 of harvest_year-1
            gs_start = pd.Timestamp(f'{hy-1}-07-01')
            sub_w['dos'] = (sub_w['date'] - gs_start).dt.days + 1
        if not np.isnan(sos) and len(wx_fy) > 0:
            at_sos = wx_fy[wx_fy['doy'] <= int(sos)]
            row['photoperiod_at_SOS'] = photoperiod_hours(lat, sos)
            row['GDD_at_SOS']         = at_sos['gdd_cum'].iloc[-1] if len(at_sos) else np.nan
            row['GDD_M2_at_SOS']      = at_sos['gdd_m2_cum'].iloc[-1] if len(at_sos) else np.nan
            row['VD_at_SOS']          = at_sos['vd_cum'].iloc[-1] if len(at_sos) else np.nan
            row['fV_at_SOS']          = at_sos['fV'].iloc[-1] if len(at_sos) else np.nan
            row['GDD_eff_at_SOS']     = at_sos['gdd_eff_cum'].iloc[-1] if len(at_sos) else np.nan
            ppt_at = wx_fy[wx_fy['doy'] <= int(sos)]['prcp'].sum()
            ppt_gu = wx_fy[(wx_fy['doy'] >= int(sos)) &
                           (wx_fy['doy'] <= row.get('NDVI_POS', 200))]['prcp'].sum()
            row['PPT_at_SOS']  = float(ppt_at)
            row['PPT_greenup'] = float(ppt_gu)
        else:
            for k in ['photoperiod_at_SOS','GDD_at_SOS','GDD_M2_at_SOS','VD_at_SOS',
                      'fV_at_SOS','GDD_eff_at_SOS','PPT_at_SOS','PPT_greenup']:
                row[k] = np.nan

        # ── WES simulation ─────────────────────────────────────────────────
        if len(wx_fy_pre) > 0:
            wx_pre = wx_fy_pre.rename(columns={'date': 'date'})
            sowing_doy = 258  # Sep 15 default
            we_out = simulate_wes(wx_pre, lat=lat, sowing_doy=sowing_doy,
                                  sowing_year=hy, return_dos=False)
            row.update(we_out)
        else:
            for s in ['emergence','tillering','jointing','flag_leaf','boot',
                      'heading','anthesis','maturity']:
                row[f'WE_{s}_doy'] = np.nan

        # ── Daymet windows (DOS-anchored) ──────────────────────────────────
        wx_seas = wx_fy_pre  # full growing season (DOS 1..365)
        for fname, spec in DAYMET_WINDOWS.items():
            mask = (wx_seas['dos'] >= spec['dos'][0]) & (wx_seas['dos'] <= spec['dos'][1])
            v = wx_seas.loc[mask, spec['var']].dropna()
            if len(v) == 0:
                row[fname] = np.nan
            else:
                row[fname] = float(getattr(v, spec['agg'])())

        for fname, expr in EVENT_FEATURES.items():
            row[fname] = int(expr(wx_seas))

        # heat_days_pre_SOS / cold_days_pre_SOS / temp range / extremes
        if not np.isnan(sos) and len(wx_fy) > 0:
            pre = wx_fy[wx_fy['doy'] <= int(sos)]
            row['heat_days_pre_SOS'] = int((pre['Tmax'] > 25).sum())
            row['cold_days_pre_SOS'] = int((pre['Tmin'] < -10).sum())
            row['max_Tmax_pre_SOS']  = float(pre['Tmax'].max()) if len(pre) else np.nan
            row['min_Tmin_pre_SOS']  = float(pre['Tmin'].min()) if len(pre) else np.nan
            pos = row.get('NDVI_POS', np.nan)
            if not np.isnan(pos):
                gu = wx_fy[(wx_fy['doy'] >= int(sos)) & (wx_fy['doy'] <= int(pos))]
                row['temp_range_greenup'] = float(gu['Tmax'].mean() - gu['Tmin'].mean()) if len(gu) else np.nan
            else:
                row['temp_range_greenup'] = np.nan
        else:
            for k in ['heat_days_pre_SOS','cold_days_pre_SOS','max_Tmax_pre_SOS',
                      'min_Tmin_pre_SOS','temp_range_greenup']:
                row[k] = np.nan

        # Emergence anchor (use WES emergence; sowing_doy fixed)
        row['emergence_doy']     = row.get('WE_emergence_doy', np.nan)
        row['sowing_doy_used']   = 258
        if not np.isnan(row['emergence_doy']) and not np.isnan(sos) and len(wx_fy) > 0:
            em = int(row['emergence_doy'])
            mask = (wx_fy['doy'] >= em) & (wx_fy['doy'] <= int(sos))
            sub_e = wx_fy[mask]
            row['VD_from_emergence_at_SOS'] = float(sub_e['vd_daily'].sum())
            row['fV_from_emergence_at_SOS'] = float(streck_fV(sub_e['vd_daily'].sum()))
            row['days_emergence_to_SOS']    = int(sos) - em
        else:
            for k in ['VD_from_emergence_at_SOS','fV_from_emergence_at_SOS',
                      'days_emergence_to_SOS']:
                row[k] = np.nan

        # NDVI pre-dormancy peak (DOS 100-150 in original) → DOY equivalent
        if ndvi_sm is not None:
            row['NDVI_pre_dormancy_peak'] = float(ndvi_sm[99:150].max()) if len(ndvi_sm) > 150 else np.nan
            # Dormancy break DOY: first DOY (200-260) where NDVI rises >5% from min
            if len(ndvi_sm) > 260:
                seg = ndvi_sm[199:260]
                seg_min = seg.min() if len(seg) else np.nan
                if not np.isnan(seg_min):
                    risen = np.where(seg >= seg_min * 1.05)[0]
                    row['dormancy_break_DOS'] = int(risen[0] + 200) if len(risen) > 0 else np.nan
                else:
                    row['dormancy_break_DOS'] = np.nan
            else:
                row['dormancy_break_DOS'] = np.nan
            # Senescence rate post POS = mean negative slope from POS to POS+60
            pos_v = row.get('NDVI_POS', np.nan)
            if not np.isnan(pos_v):
                p = int(pos_v)
                sen = ndvi_sm[p:p+60] if len(ndvi_sm) > p + 5 else None
                if sen is not None and len(sen) > 5:
                    slope = np.polyfit(np.arange(len(sen)), sen, 1)[0]
                    row['senescence_rate_post_POS'] = float(slope)
                else:
                    row['senescence_rate_post_POS'] = np.nan
            else:
                row['senescence_rate_post_POS'] = np.nan
        else:
            for k in ['NDVI_pre_dormancy_peak','dormancy_break_DOS','senescence_rate_post_POS']:
                row[k] = np.nan

        # ── MODIS LST windows ──────────────────────────────────────────────
        ls = lst[(lst['field_id'] == fid) & (lst['harvest_year'] == hy)]
        for fname, spec in LST_WINDOWS.items():
            v = ls[(ls['dos'] >= spec['dos'][0]) & (ls['dos'] <= spec['dos'][1])][spec['col']].dropna()
            if len(v) == 0:
                row[fname] = np.nan
            else:
                row[fname] = float(getattr(v, spec['agg'])())

        # ── Grain-fill + post-anthesis windows ─────────────────────────────
        gf = wx_seas[(wx_seas['dos'] >= GF_WINDOW[0]) & (wx_seas['dos'] <= GF_WINDOW[1])]
        row['srad_cum_gf']      = float(gf['srad'].sum()) if len(gf) else np.nan
        row['heat_days_30']     = int((gf['Tmax'] > 30).sum())
        row['heat_days_32']     = int((gf['Tmax'] > 32).sum())
        row['heat_days_35']     = int((gf['Tmax'] > 35).sum())
        row['prcp_cum_gf']      = float(gf['prcp'].sum()) if len(gf) else np.nan
        row['prcp_max_day_gf']  = float(gf['prcp'].max()) if len(gf) else np.nan
        # 2-week window with maximum prcp inside grain fill
        if len(gf) >= 14:
            roll = gf['prcp'].rolling(14, min_periods=14).sum()
            row['prcp_2week_gf'] = float(roll.max())
        else:
            row['prcp_2week_gf'] = np.nan

        pa = wx_seas[(wx_seas['dos'] >= PA_WINDOW[0]) & (wx_seas['dos'] <= PA_WINDOW[1])]
        row['frost_days_pa']      = int((pa['Tmin'] < 0).sum())
        row['heat_days_25_pa']    = int((pa['Tmax'] > 25).sum())
        row['heat_days_30_pa']    = int((pa['Tmax'] > 30).sum())
        row['gdd_cum_pa']         = float(pa['gdd_daily'].sum()) if len(pa) else np.nan
        row['srad_cum_pa']        = float(pa['srad'].sum()) if len(pa) else np.nan
        row['prcp_cum_pa']        = float(pa['prcp'].sum()) if len(pa) else np.nan
        row['prcp_max_day_pa']    = float(pa['prcp'].max()) if len(pa) else np.nan
        if len(pa) >= 14:
            row['prcp_2week_late_pa'] = float(pa['prcp'].rolling(14, min_periods=14).sum().max())
        else:
            row['prcp_2week_late_pa'] = np.nan

        # ── Soil placeholder (not extracted for extension; imputer handles) ─
        row['ph_top'] = np.nan

        # ── Spatial state encoding ────────────────────────────────────────
        st = infer_state(lat, lon)
        row['state'] = st
        for s in ['CO','KS','NE','NM','OK','TX']:
            row[f'state_{s}'] = 1 if st == s else 0

        rows.append(row)
        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(valid)}]  {elapsed/60:.1f} min  '
                  f'({(i+1)/elapsed*60:.0f} fy/min)')

    feat = pd.DataFrame(rows)
    out = Path(args.out)
    feat.to_parquet(out, index=False)
    print(f'\n→ {out}')
    print(f'   {len(feat):,} rows × {len(feat.columns)} cols')
    print(f'   Wall time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
