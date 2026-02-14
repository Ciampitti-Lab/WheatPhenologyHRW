"""VI smoothing, phenometric extraction, and double-logistic fitting.

Refs:
    Jönsson & Eklundh 2004 (TIMESAT)  — SOS, POS, amplitude, integrated, rate, midpoint
    Beck et al. 2006                  — double logistic Eq. 3 (6 params)
    Zhao et al. 2025                  — Left-Shoulder curvature feature
    Ruan et al. 2023 (phenoC++)       — Savitzky-Golay smoothing
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# ─── smoothing ───────────────────────────────────────────────────────────────

def smooth_vi(doys, values, window=15, polyorder=2):
    """Interpolate VI to daily DOY grid (1-365) and Savitzky-Golay smooth.

    Returns (doy_grid, smoothed) or (None, None) if fewer than 5 points.
    """
    if len(doys) < 5:
        return None, None
    idx = np.argsort(doys)
    doys_s, vals_s = np.array(doys)[idx], np.array(values)[idx]
    u_doys = np.unique(doys_s)
    u_vals = np.array([vals_s[doys_s == d].mean() for d in u_doys])
    target = np.arange(1, 366)
    daily = np.interp(target, u_doys, u_vals)
    win = min(window, len(daily))
    if win % 2 == 0:
        win -= 1
    if win < 4:
        return target, daily
    return target, savgol_filter(daily, win, polyorder)


# ─── phenometrics (TIMESAT + Zhao left-shoulder) ─────────────────────────────

def extract_phenometrics(doys, smoothed, vi_name):
    """Extract 10 phenological metrics for one VI, one field-year."""
    feat = {}

    base_vi = smoothed[:60].min()
    peak_vi = smoothed[59:200].max()
    pos_doy = int(np.argmax(smoothed[59:200])) + 60

    feat[f'{vi_name}_base'] = base_vi
    feat[f'{vi_name}_peak'] = peak_vi
    feat[f'{vi_name}_amplitude'] = peak_vi - base_vi
    feat[f'{vi_name}_POS'] = pos_doy

    deriv1 = np.gradient(smoothed, doys)
    deriv2 = np.gradient(deriv1, doys)
    deriv3 = np.gradient(deriv2, doys)

    # green-up rate = max slope before peak
    spring_d1 = deriv1[59:pos_doy]
    feat[f'{vi_name}_greenup_rate'] = spring_d1.max() if len(spring_d1) > 0 else np.nan

    # SOS = argmax of 3rd derivative before POS (proxy for ~10% threshold)
    spring_d3 = deriv3[29:pos_doy]
    spring_doys = doys[29:pos_doy]
    sos = float(spring_doys[np.argmax(spring_d3)]) if len(spring_d3) > 0 else np.nan
    feat[f'{vi_name}_SOS'] = sos

    # green-up midpoint = first downward zero-cross of 2nd derivative
    spring_d2 = deriv2[59:pos_doy]
    spring_doys_d2 = doys[59:pos_doy]
    midpoint = np.nan
    if len(spring_d2) > 1:
        for k in range(1, len(spring_d2)):
            if spring_d2[k - 1] > 0 and spring_d2[k] <= 0:
                frac = spring_d2[k - 1] / (spring_d2[k - 1] - spring_d2[k])
                midpoint = float(spring_doys_d2[k - 1] + frac)
                break
    feat[f'{vi_name}_greenup_midpoint'] = midpoint

    feat[f'{vi_name}_duration_greenup'] = pos_doy - sos if not np.isnan(sos) else np.nan

    # AUC across green-up
    if not np.isnan(sos):
        sos_int = max(0, int(sos) - 1)
        pos_int = min(364, pos_doy - 1)
        feat[f'{vi_name}_integrated'] = float(np.trapz(smoothed[sos_int:pos_int + 1]))
    else:
        feat[f'{vi_name}_integrated'] = np.nan

    # Left Shoulder: max curvature K(t) = |f''| / (1 + f'^2)^1.5  (Zhao 2025)
    if not np.isnan(sos) and (pos_doy - int(sos)) > 5:
        sos_i = max(0, int(sos) - 1)
        pos_i = min(len(doys) - 1, pos_doy - 1)
        K = np.abs(deriv2[sos_i:pos_i + 1]) / (1 + deriv1[sos_i:pos_i + 1] ** 2) ** 1.5
        feat[f'{vi_name}_LeftShoulder'] = float(doys[sos_i + np.argmax(K)]) if len(K) > 0 else np.nan
    else:
        feat[f'{vi_name}_LeftShoulder'] = np.nan

    return feat


# ─── Beck 2006 double-logistic fit ───────────────────────────────────────────

def beck_double_logistic(t, c1, c2, c3, c4, c5, c6):
    """Beck 2006 Eq. 3 — 6-param double logistic.

    c1 = baseline VI       c2 = amplitude
    c3 = green-up slope    c4 = green-up midpoint (DOY)
    c5 = senescence slope  c6 = senescence midpoint (DOY)
    """
    return c1 + c2 * (
        1.0 / (1.0 + np.exp(-c3 * (t - c4)))
        - 1.0 / (1.0 + np.exp(-c5 * (t - c6)))
    )


def fit_double_logistic(doys, smoothed):
    """Multi-start DL fit with a single-logistic fallback. Returns (c1..c6) or None."""
    vi_min = smoothed[:60].min()
    vi_max = smoothed[59:200].max()
    amp = max(vi_max - vi_min, 0.01)
    pos_doy = int(np.argmax(smoothed[59:200])) + 60

    initial_guesses = [
        [vi_min, amp, 0.08, 90, 0.08, 180],
        [vi_min, amp, 0.10, 100, 0.10, 200],
        [vi_min, amp, 0.15, 110, 0.05, 190],
        [vi_min, amp, 0.05, 85, 0.12, 210],
        [vi_min, amp, 0.12, 105, 0.08, 175],
        [vi_min, amp * 0.8, 0.10, 95, 0.10, 195],
        [vi_min, amp * 1.2, 0.08, 100, 0.06, 220],
    ]
    bounds_lower = [vi_min - abs(amp), 0, 0.005, 30, 0.005, 120]
    bounds_upper = [vi_min + abs(amp), 5 * amp, 2.0, 170, 2.0, 320]

    best_popt, best_r2 = None, -999
    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(beck_double_logistic, doys, smoothed, p0=p0,
                                bounds=(bounds_lower, bounds_upper), maxfev=10000)
            pred = beck_double_logistic(doys, *popt)
            ss_res = np.sum((smoothed - pred) ** 2)
            ss_tot = np.sum((smoothed - smoothed.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if popt[3] < popt[5] and r2 > best_r2:
                best_r2, best_popt = r2, popt
        except Exception:
            continue

    if best_r2 > 0.5 and best_popt is not None:
        return best_popt

    # Single-logistic fallback
    try:
        ascending = smoothed[:pos_doy + 30]
        ascending_doys = doys[:pos_doy + 30]

        def single_logistic(t, base, a, k, t0):
            return base + a / (1.0 + np.exp(-k * (t - t0)))

        p0_s = [vi_min, amp, 0.1, 100]
        bounds_s = ([vi_min - abs(amp), 0, 0.005, 30],
                    [vi_min + abs(amp), 5 * amp, 2.0, 170])
        popt_s, _ = curve_fit(single_logistic, ascending_doys, ascending,
                              p0=p0_s, bounds=bounds_s, maxfev=10000)
        return np.array([popt_s[0], popt_s[1], popt_s[2], popt_s[3],
                         0.1, max(pos_doy + 30, popt_s[3] + 60)])
    except Exception:
        return None


# ─── photoperiod & temperature response ──────────────────────────────────────

def photoperiod_hours(lat_deg, doy):
    """Day length (hours) from latitude and DOY — astronomical formula."""
    decl = 23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    cos_ha = -np.tan(np.radians(lat_deg)) * np.tan(np.radians(decl))
    return 2.0 * np.degrees(np.arccos(np.clip(cos_ha, -1, 1))) / 15.0


def beta_temp_response(T, Tmin, Topt, Tmax):
    """Wang & Engel 1998 Eq. 6 β-function: 0 outside [Tmin, Tmax], 1.0 at Topt."""
    if T <= Tmin or T >= Tmax:
        return 0.0
    alpha = np.log(2) / np.log((Tmax - Tmin) / (Topt - Tmin))
    x = (T - Tmin)
    xopt = (Topt - Tmin)
    val = (2 * x ** alpha * xopt ** alpha - x ** (2 * alpha)) / (xopt ** (2 * alpha))
    return max(0.0, min(1.0, val))