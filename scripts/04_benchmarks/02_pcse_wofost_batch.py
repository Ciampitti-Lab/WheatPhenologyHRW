"""Standalone parallel PCSE-WOFOST batch runner for HRW Wheat (2014-2017).

Output: data/results/pcse_wofost_phenology.csv
Columns per (field_id, year): wofost_emergence_doy, wofost_anthesis_doy, wofost_maturity_doy

Usage:
    python run_pcse_wofost_batch.py [--workers N] [--limit N] [--out PATH]
"""
import pandas as pd
import numpy as np
import datetime as dt
import warnings, time, os, sys, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
warnings.filterwarnings('ignore')

from pcse.input import YAMLCropDataProvider, WOFOST72SiteDataProvider, DummySoilDataProvider
from pcse.base import WeatherDataProvider, ParameterProvider, WeatherDataContainer
from pcse.models import Wofost72_PP
from pcse.util import reference_ET, daylength
import yaml as yamlpkg


class DaymetWeatherProvider(WeatherDataProvider):
    def __init__(self, df, latitude, longitude, elevation=400):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.angstA = 0.18
        self.angstB = 0.55
        df = df.sort_values('date').reset_index(drop=True)
        for _, row in df.iterrows():
            d = row['date'].date() if hasattr(row['date'], 'date') else row['date']
            tmin = float(row['Tmin'])
            tmax = float(row['Tmax'])
            dayl_h = daylength(d, self.latitude)
            irrad = float(row['srad']) * dayl_h * 3600.0
            vap_kpa = float(row['vp']) / 1000.0
            rain_cm = float(row['prcp']) / 10.0
            wind = 3.0
            try:
                e0, es0, et0 = reference_ET(d, self.latitude, self.elevation,
                                            tmin, tmax, irrad, vap_kpa*10.0,
                                            wind, self.angstA, self.angstB)
                e0  = min(max(e0,  0.0), 2.5)
                es0 = min(max(es0, 0.0), 2.5)
                et0 = min(max(et0, 0.0), 2.5)
            except Exception:
                e0 = es0 = et0 = 0.5
            wdc = WeatherDataContainer(
                LAT=self.latitude, LON=self.longitude, ELEV=self.elevation,
                DAY=d,
                IRRAD=irrad, TMIN=tmin, TMAX=tmax,
                VAP=vap_kpa*10.0, WIND=wind, RAIN=rain_cm,
                E0=e0, ES0=es0, ET0=et0, SNOWDEPTH=0.0,
            )
            self._store_WeatherDataContainer(wdc, d)


def build_agromgmt(sow_date, end_date):
    campaign_start = sow_date - dt.timedelta(days=7)
    yaml_str = f"""
- {campaign_start}:
    CropCalendar:
        crop_name: wheat
        variety_name: Winter_wheat_106
        crop_start_date: {sow_date}
        crop_start_type: sowing
        crop_end_date: {end_date}
        crop_end_type: maturity
        max_duration: 350
    TimedEvents: null
    StateEvents: null
"""
    return yamlpkg.safe_load(yaml_str)


def extract_phenology(out_df):
    res = {}
    for stage_name, dvs_thresh in [('emergence', 0.0), ('anthesis', 1.0), ('maturity', 2.0)]:
        if stage_name == 'emergence':
            mask = out_df['DVS'].notna() & (out_df['DVS'] > 0)
        else:
            mask = out_df['DVS'] >= dvs_thresh
        if not mask.any():
            res[stage_name] = np.nan
            continue
        first_day = out_df.loc[mask.idxmax(), 'day']
        res[stage_name] = first_day.timetuple().tm_yday
    return res


# Worker function — params built fresh per process to avoid pickling issues
def _build_params():
    crop = YAMLCropDataProvider()
    crop.set_active_crop('wheat', 'Winter_wheat_106')
    soil = DummySoilDataProvider()
    site = WOFOST72SiteDataProvider(WAV=10)
    return ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)


_PARAMS_CACHE = None
def get_params():
    global _PARAMS_CACHE
    if _PARAMS_CACHE is None:
        _PARAMS_CACHE = _build_params()
    return _PARAMS_CACHE


def simulate_one(args):
    """Run WOFOST for a single (field_id, year) combo."""
    fid, yr, lat, lon, wx_field_dict = args
    try:
        wx_field = pd.DataFrame(wx_field_dict)
        wx_field['date'] = pd.to_datetime(wx_field['date'])
        wx_gs = wx_field[(wx_field['date'] >= pd.Timestamp(f'{yr-1}-07-01')) &
                         (wx_field['date'] <= pd.Timestamp(f'{yr}-07-15'))]
        if len(wx_gs) < 200:
            return {'field_id': fid, 'year': yr, 'error': 'insufficient_weather'}
        wp = DaymetWeatherProvider(wx_gs, latitude=lat, longitude=lon, elevation=400)
        sow_date = dt.date(yr-1, 10, 15)
        end_date = dt.date(yr, 7, 15)
        agro = build_agromgmt(sow_date, end_date)
        wofost = Wofost72_PP(get_params(), wp, agro)
        wofost.run_till_terminate()
        out = pd.DataFrame(wofost.get_output())
        out['day'] = pd.to_datetime(out['day'])
        ph = extract_phenology(out)
        return {'field_id': fid, 'year': yr,
                'wofost_emergence_doy': ph['emergence'],
                'wofost_anthesis_doy': ph['anthesis'],
                'wofost_maturity_doy': ph['maturity']}
    except Exception as e:
        return {'field_id': fid, 'year': yr, 'error': str(e)[:120]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument('--limit', type=int, default=None, help='limit to first N field-years (for testing)')
    parser.add_argument('--out', default='data/results/pcse_wofost_phenology.csv')
    args = parser.parse_args()

    FEAT_PATH    = 'data/processed/features/features_with_external.parquet'
    WEATHER_PATH = 'data/raw/satellite/daymet_daily_weather_full.csv'

    print(f'Loading features from {FEAT_PATH} ...')
    feat = pd.read_parquet(FEAT_PATH)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    feat = feat[feat['year'] >= 2014].reset_index(drop=True)
    if args.limit:
        feat = feat.head(args.limit)
    print(f'Field-years to simulate: {len(feat):,}')

    fields_of_interest = set(feat['field_id'].unique())
    print(f'Loading weather (chunked, {len(fields_of_interest):,} fields) ...')
    wx_chunks = []
    for chunk in pd.read_csv(WEATHER_PATH, chunksize=500_000):
        chunk['FIELDID'] = chunk['FIELDID'].astype(str)
        chunk = chunk[chunk['FIELDID'].isin(fields_of_interest)]
        if len(chunk) > 0:
            wx_chunks.append(chunk)
    wx = pd.concat(wx_chunks, ignore_index=True)
    wx['date'] = pd.to_datetime(wx['date'])
    print(f'Weather rows: {len(wx):,}')
    # Pre-group by field as DICTS (for pickling to workers)
    # Daymet drops Dec 31 in non-leap years and Feb 29 in leap years.
    # Pad missing days by forward-filling from previous day so PCSE doesn't fail.
    wx_by_field = {}
    for fid, g in wx.groupby('FIELDID'):
        g = g.sort_values('date').set_index('date')
        # Reindex to full daily range, forward-fill missing days
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
        g = g.reindex(full_idx).ffill().reset_index().rename(columns={'index': 'date'})
        wx_by_field[fid] = {
            'date': g['date'].dt.strftime('%Y-%m-%d').tolist(),
            'Tmin': g['Tmin'].tolist(),
            'Tmax': g['Tmax'].tolist(),
            'srad': g['srad'].tolist(),
            'vp': g['vp'].tolist(),
            'prcp': g['prcp'].tolist(),
        }
    print(f'Pre-grouped weather for {len(wx_by_field):,} fields')

    jobs = []
    for _, r in feat.iterrows():
        fid = r['field_id']
        if fid not in wx_by_field: continue
        jobs.append((fid, int(r['year']), float(r['latitude']), float(r['longitude']),
                     wx_by_field[fid]))
    print(f'Jobs ready: {len(jobs):,}, workers: {args.workers}')

    t_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(simulate_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            results.append(fut.result())
            if (i+1) % 500 == 0 or (i+1) == len(jobs):
                elapsed = time.time() - t_start
                rate = (i+1) / elapsed
                eta = (len(jobs) - i - 1) / rate
                print(f'  {i+1}/{len(jobs)} done ({elapsed:.0f}s elapsed, {rate:.1f}/s, ETA {eta:.0f}s)')

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    n_ok = df['wofost_anthesis_doy'].notna().sum() if 'wofost_anthesis_doy' in df.columns else 0
    n_err = df.get('error', pd.Series(dtype=str)).notna().sum() if 'error' in df.columns else 0
    print(f'\n=== DONE in {(time.time()-t_start)/60:.1f} min ===')
    print(f'Total rows: {len(df)}, valid anthesis: {n_ok}, errors: {n_err}')
    print(f'Saved: {args.out}')
    if 'wofost_anthesis_doy' in df.columns:
        print('Anthesis DOY:', df['wofost_anthesis_doy'].describe(percentiles=[0.1,0.5,0.9]).round(1).to_dict())
        print('Maturity DOY:', df['wofost_maturity_doy'].describe(percentiles=[0.1,0.5,0.9]).round(1).to_dict())


if __name__ == '__main__':
    main()
