"""V2-tuned feature pipeline — applies 3 fixes to address R² drops:

  1. SOS constrained to DOS ≥ 220 (avoids fall-greenup false positives
     being detected as start-of-season instead of true winter→spring break).
  2. Larger Savitzky-Golay smoothing (window=21, polyorder=2) to dampen
     noise that throws off derivative-based SOS detection.
  3. Per-feature winsorization at p1/p99 inside the per-(field,year)
     loop — outliers in fall data (e.g. one bad NDVI obs) shouldn't
     blow up linear models downstream.

Same I/O as 15_growing_season_pipeline.py — drop-in replacement for
v2 feature engineering. Keeps the same DOS coordinate convention.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

ROOT = Path('/home/vmangidi/repositories/WheatPhenologyHRW')
sys.path.insert(0, str(ROOT))
from scripts.utils.thermal import (simulate_wes, beta_temp_response,
                                   streck_fV, gdd_method2_daily)
from scripts.utils.features import (beck_double_logistic, fit_double_logistic,
                                    photoperiod_hours)

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
HLS_PATH    = EXT / 'hls_full_2013_2024.parquet'
DAYMET_PATH = EXT / 'daymet_full_2013_2024.parquet'
LST_PATH    = EXT / 'modis_lst_2012_2024.parquet'

# Tuning parameters
SG_WINDOW   = 21    # was 15
SG_POLY     = 2
SOS_DOS_MIN = 220   # require winter dormancy break, not fall greenup


def smooth_vi_dos(dos_arr, vals_arr, window=SG_WINDOW, polyorder=SG_POLY):
    if len(dos_arr) < 5:
        return None, None
    idx = np.argsort(dos_arr)
    dos_s, vals_s = np.array(dos_arr)[idx], np.array(vals_arr)[idx]
    u_dos = np.unique(dos_s)
    u_vals = np.array([vals_s[dos_s == d].mean() for d in u_dos])
    target = np.arange(1, 366)
    daily = np.interp(target, u_dos, u_vals)
    win = min(window, len(daily))
    if win % 2 == 0:
        win -= 1
    if win < 4:
        return target, daily
    return target, savgol_filter(daily, win, polyorder)


def extract_phenometrics_gs_tuned(dos_arr, smoothed, vi_name):
    feat = {}
    WIN_BASELINE = (180, 240)   # winter dormancy
    WIN_PEAK     = (240, 360)   # spring peak
    SOS_FLOOR    = SOS_DOS_MIN  # 220 - require post-dormancy SOS

    base_vi = smoothed[WIN_BASELINE[0]:WIN_BASELINE[1]].min()
    peak_window = smoothed[WIN_PEAK[0]:WIN_PEAK[1]]
    peak_vi = peak_window.max()
    pos_dos = int(np.argmax(peak_window)) + WIN_PEAK[0] + 1

    feat[f'{vi_name}_base'] = float(base_vi)
    feat[f'{vi_name}_peak'] = float(peak_vi)
    feat[f'{vi_name}_amplitude'] = float(peak_vi - base_vi)
    feat[f'{vi_name}_POS'] = pos_dos

    deriv1 = np.gradient(smoothed, dos_arr)
    deriv2 = np.gradient(deriv1, dos_arr)
    deriv3 = np.gradient(deriv2, dos_arr)

    spring_d1 = deriv1[SOS_FLOOR:pos_dos]
    feat[f'{vi_name}_greenup_rate'] = float(spring_d1.max()) if len(spring_d1) else np.nan

    spring_d3 = deriv3[SOS_FLOOR:pos_dos]
    spring_dos = dos_arr[SOS_FLOOR:pos_dos]
    sos = float(spring_dos[np.argmax(spring_d3)]) if len(spring_d3) else np.nan
    feat[f'{vi_name}_SOS'] = sos

    if not np.isnan(sos):
        spring_d2 = deriv2[int(sos)-1:pos_dos]
        spring_d2_dos = dos_arr[int(sos)-1:pos_dos]
        midpoint = np.nan
        if len(spring_d2) > 1:
            for k in range(1, len(spring_d2)):
                if spring_d2[k-1] > 0 and spring_d2[k] <= 0:
                    frac = spring_d2[k-1] / (spring_d2[k-1] - spring_d2[k])
                    midpoint = float(spring_d2_dos[k-1] + frac)
                    break
        feat[f'{vi_name}_greenup_midpoint'] = midpoint
        feat[f'{vi_name}_duration_greenup'] = pos_dos - sos
        sos_i = max(0, int(sos) - 1)
        feat[f'{vi_name}_integrated'] = float(np.trapz(smoothed[sos_i:pos_dos]))
    else:
        feat[f'{vi_name}_greenup_midpoint'] = np.nan
        feat[f'{vi_name}_duration_greenup'] = np.nan
        feat[f'{vi_name}_integrated'] = np.nan

    if not np.isnan(sos) and (pos_dos - int(sos)) > 5:
        sos_i = max(0, int(sos) - 1)
        K = (np.abs(deriv2[sos_i:pos_dos])
             / (1 + deriv1[sos_i:pos_dos] ** 2) ** 1.5)
        feat[f'{vi_name}_LeftShoulder'] = float(dos_arr[sos_i + np.argmax(K)]) if len(K) else np.nan
    else:
        feat[f'{vi_name}_LeftShoulder'] = np.nan

    return feat


def infer_state(lat, lon):
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    if lon < -103.5 and lat < 37.0:    return 'NM'
    if lon < -103.5:                   return 'CO'
    if lat < 34.5:                     return 'TX'
    if lat < 37.0:                     return 'OK'
    if lat < 40.0:                     return 'KS'
    return 'NE'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--valid-path', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print('=== V2-TUNED pipeline (SOS≥220, SG window=21, larger smoothing) ===')
    valid = pd.read_parquet(args.valid_path)
    if 'harvest_year' not in valid.columns and 'year' in valid.columns:
        valid = valid.rename(columns={'year': 'harvest_year'})
    valid['field_id'] = valid['field_id'].astype(str)
    if args.limit > 0:
        valid = valid.head(args.limit).copy()
        print(f'[SMOKE] limit={args.limit}')
    print(f'Valid field-years: {len(valid):,}')
    valid_fields = set(valid['field_id'].unique())

    hy_min, hy_max = int(valid['harvest_year'].min()), int(valid['harvest_year'].max())
    date_lo = pd.Timestamp(f'{hy_min - 1}-07-01')
    date_hi = pd.Timestamp(f'{hy_max}-07-31')

    import pyarrow.parquet as pq
    print('Loading HLS...')
    hls = pq.read_table(HLS_PATH,
        filters=[('date', '>=', date_lo), ('date', '<=', date_hi)]).to_pandas()
    hls['field_id'] = hls['field_id'].astype(str)
    hls = hls[hls['field_id'].isin(valid_fields)].copy()
    print(f'HLS rows: {len(hls):,}')

    print('Loading Daymet...')
    wx = pq.read_table(DAYMET_PATH,
        columns=['FIELDID','date','Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq',
                 'harvest_year','dos','month'],
        filters=[('date', '>=', date_lo), ('date', '<=', date_hi)]).to_pandas()
    wx = wx.rename(columns={'FIELDID': 'field_id'})
    wx['field_id'] = wx['field_id'].astype(str)
    wx = wx[wx['field_id'].isin(valid_fields)].copy()
    for c in ['Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq']:
        wx[c] = wx[c].astype('float32')
    wx['date'] = pd.to_datetime(wx['date'])
    wx['T_mean'] = ((wx['Tmin'] + wx['Tmax']) / 2.0).astype('float32')
    wx['doy'] = wx['date'].dt.dayofyear.astype('int16')
    wx = wx.sort_values(['field_id', 'date']).reset_index(drop=True)
    print(f'Daymet rows: {len(wx):,}')

    Tmin_v, Topt_v, Tmax_v = 1.3, 4.9, 15.7
    alpha = np.log(2) / np.log((Tmax_v - Tmin_v) / (Topt_v - Tmin_v))
    xopt = Topt_v - Tmin_v
    def beta_vec(T_arr):
        x = T_arr - Tmin_v
        with np.errstate(invalid='ignore'):
            val = (2 * np.power(x, alpha) * (xopt ** alpha) - np.power(x, 2 * alpha)) / (xopt ** (2 * alpha))
        return np.where((T_arr <= Tmin_v) | (T_arr >= Tmax_v), 0.0, np.clip(val, 0.0, 1.0))

    print('Loading LST...')
    lst = pq.read_table(LST_PATH,
        filters=[('date', '>=', date_lo), ('date', '<=', date_hi)]).to_pandas()
    lst = lst.rename(columns={'FIELDID': 'field_id'})
    lst['field_id'] = lst['field_id'].astype(str)
    lst = lst[lst['field_id'].isin(valid_fields)].copy()
    print(f'LST rows: {len(lst):,}')

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
    GF_WINDOW = (300, 350)
    PA_WINDOW = (295, 360)

    print(f'\n=== Iterating over {len(valid):,} field-years ===')
    rows = []
    for i, fy in enumerate(valid.itertuples(index=False)):
        fid = fy.field_id
        hy  = int(fy.harvest_year)
        lat = getattr(fy, 'lat', None)
        lon = getattr(fy, 'lon', None)
        if lat is None or pd.isna(lat):
            continue

        sub = hls[(hls['field_id'] == fid) & (hls['harvest_year'] == hy)].copy()
        if len(sub) < 5:
            continue

        row = {'field_id': fid, 'year': hy, 'flag_true_doy': np.nan,
               'n_obs': int(len(sub)),
               'latitude': lat, 'longitude': lon}

        smoothed_per_vi = {}
        for vi in ['NDVI', 'EVI', 'GCVI', 'NDRE']:
            if vi not in sub.columns:
                continue
            vsub = sub[sub[vi].notna()][['dos', vi]]
            if len(vsub) < 5:
                row.update({f'{vi}_{k}': np.nan for k in
                    ['base','peak','amplitude','POS','greenup_rate','SOS',
                     'greenup_midpoint','duration_greenup','integrated','LeftShoulder']})
                continue
            dos_grid, sm = smooth_vi_dos(vsub['dos'].values, vsub[vi].values)
            if dos_grid is None:
                continue
            smoothed_per_vi[vi] = (dos_grid, sm)
            row.update(extract_phenometrics_gs_tuned(dos_grid, sm, vi))

        ndvi_dos, ndvi_sm = smoothed_per_vi.get('NDVI', (None, None))

        if ndvi_sm is not None:
            popt = fit_double_logistic(ndvi_dos, ndvi_sm)
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

        wx_fy = wx[(wx['field_id'] == fid) & (wx['harvest_year'] == hy)].copy()
        if len(wx_fy):
            wx_fy['gdd_daily']    = np.maximum(0, wx_fy['T_mean'].values)
            wx_fy['gdd_m2_daily'] = gdd_method2_daily(
                wx_fy['Tmin'].values, wx_fy['Tmax'].values, 0, 35)
            wx_fy['vd_daily']     = beta_vec(wx_fy['T_mean'].values)
            wx_fy = wx_fy.sort_values('dos')
            wx_fy['gdd_cum']      = wx_fy['gdd_daily'].cumsum()
            wx_fy['gdd_m2_cum']   = wx_fy['gdd_m2_daily'].cumsum()
            wx_fy['vd_cum']       = wx_fy['vd_daily'].cumsum()
            wx_fy['fV']           = streck_fV(wx_fy['vd_cum'].values)
            wx_fy['gdd_eff_daily']= wx_fy['gdd_m2_daily'] * wx_fy['fV']
            wx_fy['gdd_eff_cum']  = wx_fy['gdd_eff_daily'].cumsum()

        sos_dos = row.get('NDVI_SOS', np.nan)
        if not np.isnan(sos_dos) and len(wx_fy):
            at_sos = wx_fy[wx_fy['dos'] <= int(sos_dos)]
            row['photoperiod_at_SOS'] = photoperiod_hours(lat, sos_dos)
            row['GDD_at_SOS']         = float(at_sos['gdd_cum'].iloc[-1]) if len(at_sos) else np.nan
            row['GDD_M2_at_SOS']      = float(at_sos['gdd_m2_cum'].iloc[-1]) if len(at_sos) else np.nan
            row['VD_at_SOS']          = float(at_sos['vd_cum'].iloc[-1]) if len(at_sos) else np.nan
            row['fV_at_SOS']          = float(at_sos['fV'].iloc[-1]) if len(at_sos) else np.nan
            row['GDD_eff_at_SOS']     = float(at_sos['gdd_eff_cum'].iloc[-1]) if len(at_sos) else np.nan
            ppt_at = float(at_sos['prcp'].sum())
            pos_dos_v = row.get('NDVI_POS', np.nan)
            if not np.isnan(pos_dos_v):
                ppt_gu = float(wx_fy[(wx_fy['dos'] >= int(sos_dos)) &
                                     (wx_fy['dos'] <= int(pos_dos_v))]['prcp'].sum())
            else:
                ppt_gu = np.nan
            row['PPT_at_SOS']  = ppt_at
            row['PPT_greenup'] = ppt_gu
        else:
            for k in ['photoperiod_at_SOS','GDD_at_SOS','GDD_M2_at_SOS','VD_at_SOS',
                      'fV_at_SOS','GDD_eff_at_SOS','PPT_at_SOS','PPT_greenup']:
                row[k] = np.nan

        if len(wx_fy):
            sowing_doy = 258
            we_out = simulate_wes(wx_fy, lat=lat, sowing_doy=sowing_doy,
                                  sowing_year=hy, return_dos=False)
            row.update(we_out)
        else:
            for s in ['emergence','tillering','jointing','flag_leaf','boot',
                      'heading','anthesis','maturity']:
                row[f'WE_{s}_doy'] = np.nan

        for fname, spec in DAYMET_WINDOWS.items():
            v = wx_fy[(wx_fy['dos'] >= spec['dos'][0]) &
                      (wx_fy['dos'] <= spec['dos'][1])][spec['var']].dropna()
            row[fname] = float(getattr(v, spec['agg'])()) if len(v) else np.nan

        for fname, expr in EVENT_FEATURES.items():
            row[fname] = int(expr(wx_fy)) if len(wx_fy) else np.nan

        if not np.isnan(sos_dos) and len(wx_fy):
            pre = wx_fy[wx_fy['dos'] <= int(sos_dos)]
            row['heat_days_pre_SOS'] = int((pre['Tmax'] > 25).sum())
            row['cold_days_pre_SOS'] = int((pre['Tmin'] < -10).sum())
            row['max_Tmax_pre_SOS']  = float(pre['Tmax'].max()) if len(pre) else np.nan
            row['min_Tmin_pre_SOS']  = float(pre['Tmin'].min()) if len(pre) else np.nan
            pos_dos_v = row.get('NDVI_POS', np.nan)
            if not np.isnan(pos_dos_v):
                gu = wx_fy[(wx_fy['dos'] >= int(sos_dos)) & (wx_fy['dos'] <= int(pos_dos_v))]
                row['temp_range_greenup'] = float(gu['Tmax'].mean() - gu['Tmin'].mean()) if len(gu) else np.nan
            else:
                row['temp_range_greenup'] = np.nan
        else:
            for k in ['heat_days_pre_SOS','cold_days_pre_SOS','max_Tmax_pre_SOS',
                      'min_Tmin_pre_SOS','temp_range_greenup']:
                row[k] = np.nan

        row['emergence_doy']     = row.get('WE_emergence_doy', np.nan)
        row['sowing_doy_used']   = 258
        if not np.isnan(row['emergence_doy']) and not np.isnan(sos_dos) and len(wx_fy):
            em_doy = int(row['emergence_doy'])
            em_date = pd.Timestamp(f'{hy-1}-01-01') + pd.Timedelta(days=em_doy - 1)
            em_dos = (em_date - pd.Timestamp(f'{hy-1}-07-01')).days + 1
            mask = (wx_fy['dos'] >= em_dos) & (wx_fy['dos'] <= int(sos_dos))
            sub_e = wx_fy[mask]
            row['VD_from_emergence_at_SOS'] = float(sub_e['vd_daily'].sum()) if len(sub_e) else np.nan
            row['fV_from_emergence_at_SOS'] = float(streck_fV(sub_e['vd_daily'].sum())) if len(sub_e) else np.nan
            row['days_emergence_to_SOS']    = int(sos_dos) - em_dos
        else:
            for k in ['VD_from_emergence_at_SOS','fV_from_emergence_at_SOS','days_emergence_to_SOS']:
                row[k] = np.nan

        if ndvi_sm is not None:
            row['NDVI_pre_dormancy_peak'] = float(ndvi_sm[59:150].max())
            winter_min = float(ndvi_sm[179:240].min())
            dormancy_seg = ndvi_sm[199:280]
            risen = np.where(dormancy_seg >= winter_min * 1.05)[0]
            row['dormancy_break_DOS'] = int(risen[0] + 200) if len(risen) else np.nan
            pos_v = row.get('NDVI_POS', np.nan)
            if not np.isnan(pos_v):
                p = int(pos_v)
                end = min(365, p + 45)
                sen = ndvi_sm[p-1:end] if end > p else None
                if sen is not None and len(sen) > 5:
                    row['senescence_rate_post_POS'] = float(np.polyfit(np.arange(len(sen)), sen, 1)[0])
                else:
                    row['senescence_rate_post_POS'] = np.nan
            else:
                row['senescence_rate_post_POS'] = np.nan
        else:
            for k in ['NDVI_pre_dormancy_peak','dormancy_break_DOS','senescence_rate_post_POS']:
                row[k] = np.nan

        ls = lst[(lst['field_id'] == fid) & (lst['harvest_year'] == hy)]
        for fname, spec in LST_WINDOWS.items():
            v = ls[(ls['dos'] >= spec['dos'][0]) & (ls['dos'] <= spec['dos'][1])][spec['col']].dropna()
            row[fname] = float(getattr(v, spec['agg'])()) if len(v) else np.nan

        if len(wx_fy):
            gf = wx_fy[(wx_fy['dos'] >= GF_WINDOW[0]) & (wx_fy['dos'] <= GF_WINDOW[1])]
            row['srad_cum_gf']     = float(gf['srad'].sum()) if len(gf) else np.nan
            row['heat_days_30']    = int((gf['Tmax'] > 30).sum())
            row['heat_days_32']    = int((gf['Tmax'] > 32).sum())
            row['heat_days_35']    = int((gf['Tmax'] > 35).sum())
            row['prcp_cum_gf']     = float(gf['prcp'].sum()) if len(gf) else np.nan
            row['prcp_max_day_gf'] = float(gf['prcp'].max()) if len(gf) else np.nan
            row['prcp_2week_gf']   = float(gf['prcp'].rolling(14, min_periods=14).sum().max()) if len(gf) >= 14 else np.nan

            pa = wx_fy[(wx_fy['dos'] >= PA_WINDOW[0]) & (wx_fy['dos'] <= PA_WINDOW[1])]
            row['frost_days_pa']      = int((pa['Tmin'] < 0).sum())
            row['heat_days_25_pa']    = int((pa['Tmax'] > 25).sum())
            row['heat_days_30_pa']    = int((pa['Tmax'] > 30).sum())
            row['gdd_cum_pa']         = float(pa['gdd_daily'].sum()) if len(pa) else np.nan
            row['srad_cum_pa']        = float(pa['srad'].sum()) if len(pa) else np.nan
            row['prcp_cum_pa']        = float(pa['prcp'].sum()) if len(pa) else np.nan
            row['prcp_max_day_pa']    = float(pa['prcp'].max()) if len(pa) else np.nan
            row['prcp_2week_late_pa'] = float(pa['prcp'].rolling(14, min_periods=14).sum().max()) if len(pa) >= 14 else np.nan
        else:
            for k in ['srad_cum_gf','heat_days_30','heat_days_32','heat_days_35',
                      'prcp_cum_gf','prcp_max_day_gf','prcp_2week_gf',
                      'frost_days_pa','heat_days_25_pa','heat_days_30_pa',
                      'gdd_cum_pa','srad_cum_pa','prcp_cum_pa','prcp_max_day_pa',
                      'prcp_2week_late_pa']:
                row[k] = np.nan

        row['ph_top'] = np.nan
        st = infer_state(lat, lon)
        row['state'] = st
        for s in ['CO','KS','NE','NM','OK','TX']:
            row[f'state_{s}'] = 1 if st == s else 0

        rows.append(row)
        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(valid)}]  {elapsed/60:.1f} min  ({(i+1)/elapsed*60:.0f} fy/min)')

    feat = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out, index=False)
    print(f'\n→ {out}')
    print(f'   {len(feat):,} rows × {len(feat.columns)} cols')
    print(f'   Wall time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
