"""Thermal-time utilities and the WES (Wang-Engel-Streck) phenology simulator.

The WES core integrates a 3-phase Wang & Engel (1998) DVS-rate equation with
Streck (2003)'s sigmoidal vernalization function. Stage events are detected
at 8 fractional DVS thresholds via the Zadoks/BBCH mapping. All parameters
are taken from the original papers (no calibration to our data).

Refs:
    McMaster & Wilhelm 1997 — GDD Method 2
    Wang & Engel 1998        — beta f(T), three-phase DVS-rate, photoperiod ramp
    Streck et al. 2003       — sigmoidal f(V) = VD^5 / (22.5^5 + VD^5)
    Porter & Gawith 1999     — vernalization cardinal temperatures (1.3 / 4.9 / 15.7 °C)
    Zadoks 1974 / Lancashire 1991 — BBCH decimal scale used for the 8 thresholds
"""
import numpy as np
import pandas as pd

from .features import beta_temp_response, photoperiod_hours


# ─── thermal time ────────────────────────────────────────────────────────────

def gdd_method2_daily(Tmin, Tmax, T_base=0.0, T_upper=35.0):
    """Daily GDD using McMaster & Wilhelm 1997 Method 2 (clip before averaging)."""
    tmin = np.clip(Tmin, T_base, T_upper)
    tmax = np.clip(Tmax, T_base, T_upper)
    return np.maximum(0.0, (tmax + tmin) / 2.0 - T_base)


# ─── vernalization ───────────────────────────────────────────────────────────

def daily_vernalization(T_mean, Tmin=1.3, Topt=4.9, Tmax=15.7):
    """Daily vernalization rate (Wang & Engel 1998 Eq. 6) with Porter & Gawith cardinals."""
    return beta_temp_response(T_mean, Tmin, Topt, Tmax)


def streck_fV(VD_cumulative, VD_half=22.5, n=5):
    """Streck 2003 Eq. 4 — sigmoidal vernalization satisfaction in [0, 1]."""
    VD = np.asarray(VD_cumulative, dtype=float)
    return VD ** n / (VD_half ** n + VD ** n)


# ─── legacy single-target Wang-Engel simulator ───────────────────────────────
# Kept for the early flag-leaf-only experiments; production code uses
# simulate_wes() below.

def simulate_wang_engel(weather_df, lat, sowing_doy, sowing_year,
                        R_dev_max=0.035,
                        Tmin_v=0, Topt_v=24, Tmax_v=35,
                        Tmin_vn=1.3, Topt_vn=4.9, Tmax_vn=15.7,
                        VND=22.0,
                        photoperiod_critical=7.0, photoperiod_omega=0.28,
                        flag_leaf_threshold=0.65):
    """Forward simulation. Returns calendar DOY when DVS first hits flag_leaf_threshold."""
    if weather_df is None or len(weather_df) == 0:
        return np.nan
    wx = weather_df.sort_values('date').reset_index(drop=True)

    vd_cum = 0.0
    Sdev = 0.0
    for _, r in wx.iterrows():
        fvn = beta_temp_response(r['T_mean'], Tmin_vn, Topt_vn, Tmax_vn)
        vd_cum += fvn
        fV = streck_fV(vd_cum, VD_half=VND)

        fT = beta_temp_response(r['T_mean'], Tmin_v, Topt_v, Tmax_v)

        hp = photoperiod_hours(lat, r['doy'])
        if hp > photoperiod_critical:
            fP = max(0.0, 1 - np.exp(-photoperiod_omega * (hp - photoperiod_critical)))
        else:
            fP = 0.0

        Sdev += R_dev_max * fT * fV * fP
        if Sdev >= flag_leaf_threshold:
            return int(r['date'].timetuple().tm_yday)

    return np.nan


# ─── WES — multi-stage simulator (production) ────────────────────────────────

# DVS thresholds — fractional values mapped to BBCH stages.
# Source: standard BBCH ↔ DVS table for wheat (Asseng 2000; Boogaard 2014).
WES_STAGE_THRESHOLDS = {
    'emergence': 0.05,
    'tillering': 0.20,
    'jointing':  0.45,
    'flag_leaf': 0.65,
    'boot':      0.75,
    'heading':   0.85,
    'anthesis':  1.00,
    'maturity':  2.00,
}

# Phase-specific parameters from Wang & Engel 1998. R_max in 1/day.
# Phase 1 uses vegetative cardinals; Phase 3 shifts Topt warmer for grain fill.
WES_PHASE_PARAMS = {
    'phase1_emergence': {
        'Sdev_min': 0.0,  'Sdev_max': 0.05,
        'R_max': 0.10,
        'cardinals_T': (0, 24, 35),
        'use_fV': False,   # vernalization not yet engaged
        'use_fP': False,   # seed underground
    },
    'phase2_vegetative': {
        'Sdev_min': 0.05, 'Sdev_max': 1.00,
        'R_max': 0.025,
        'cardinals_T': (0, 24, 35),
        'use_fV': True,    # Streck f(V) gates flowering
        'use_fP': True,    # long-day plant
    },
    'phase3_reproductive': {
        'Sdev_min': 1.00, 'Sdev_max': 2.00,
        'R_max': 0.025,
        'cardinals_T': (0, 29, 40),  # warmer optimum for grain fill
        'use_fV': False,
        'use_fP': False,
    },
}


def _phase_for_sdev(Sdev):
    for name, p in WES_PHASE_PARAMS.items():
        if p['Sdev_min'] <= Sdev < p['Sdev_max']:
            return name, p
    # past maturity — stay in phase 3
    return 'phase3_reproductive', WES_PHASE_PARAMS['phase3_reproductive']


def simulate_wes(weather_df, lat, sowing_doy, sowing_year,
                 thresholds=None,
                 Tmin_vn=1.3, Topt_vn=4.9, Tmax_vn=15.7, VND=22.0,
                 photoperiod_critical=7.0, photoperiod_omega=0.28,
                 return_dos=True):
    """Run WES forward simulation for one field-year.

    Daily Euler integration of dDVS/dt = R_max(phi) * f(T) * f(V) * f(P).
    Returns predicted DOY (or DOS) for each stage in `thresholds`.

    weather_df : DataFrame with ['date', 'doy', 'T_mean'], sorted by date
    lat        : field latitude in degrees (for photoperiod)
    sowing_doy : DOY of fall sowing — used only for the growing-season anchor
    sowing_year: harvest_year (i.e. season ends in this calendar year)
    return_dos : if True, return DOS (days from Jul 1 of harvest_year-1)
    """
    if thresholds is None:
        thresholds = WES_STAGE_THRESHOLDS
    if weather_df is None or len(weather_df) == 0:
        return {f'WE_{s}_doy': np.nan for s in thresholds}

    wx = weather_df.sort_values('date').reset_index(drop=True)
    gs_start = pd.Timestamp(f'{sowing_year - 1}-07-01')

    # Sowing date — only start integrating thermal/vernalization AFTER sowing.
    # Pre-sowing days are fallow; there's no wheat to develop.
    sow_date = pd.Timestamp(f'{sowing_year - 1}-01-01') + pd.Timedelta(days=sowing_doy - 1)
    if sow_date < gs_start:
        sow_date = gs_start

    vd_cum = 0.0
    Sdev = 0.0
    out = {f'WE_{s}_doy': np.nan for s in thresholds}
    pending = sorted(thresholds.items(), key=lambda x: x[1])

    for _, r in wx.iterrows():
        if r['date'] < sow_date:
            continue   # fallow — no wheat development before sowing
        T = r['T_mean']

        # accumulate vernalization regardless of phase (only consumed in Phase 2)
        vd_cum += beta_temp_response(T, Tmin_vn, Topt_vn, Tmax_vn)
        fV = streck_fV(vd_cum, VD_half=VND)

        _, p = _phase_for_sdev(Sdev)
        fT = beta_temp_response(T, *p['cardinals_T'])
        fV_eff = fV if p['use_fV'] else 1.0

        if p['use_fP']:
            hp = photoperiod_hours(lat, r['doy'])
            if hp > photoperiod_critical:
                fP_eff = max(0.0, 1 - np.exp(-photoperiod_omega * (hp - photoperiod_critical)))
            else:
                fP_eff = 0.0
        else:
            fP_eff = 1.0

        Sdev += p['R_max'] * fT * fV_eff * fP_eff

        # detect threshold crossings — pending is sorted ascending
        while pending and Sdev >= pending[0][1]:
            stage_name, _ = pending.pop(0)
            value = (r['date'] - gs_start).days + 1 if return_dos else r['date'].timetuple().tm_yday
            out[f'WE_{stage_name}_doy'] = int(value)
        if not pending:
            break

    return out
