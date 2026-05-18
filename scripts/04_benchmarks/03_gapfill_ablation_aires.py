"""
Gap-filling ablation — reviewer defense for the "no gap-fill" design decision.

Reproduces the validation protocol of Aires et al. (2026, J. Remote Sensing,
doi:10.34133/remotesensing.0878), "Gap-filled daily EVI time series validation",
on the WheatPhenologyHRW HLS stack.

Protocol (faithful to the paper):
  - Per field, split the clear-sky EVI observations into train/test (test = held-out
    observations that simulate cloud-induced gaps).
  - Reconstruct each test observation from the TRAIN observations that fall inside a
    +/-15-day window around the target date (the window size Aires et al. selected
    empirically; they swept +/-3..+/-30 d).
  - 4 methods: temporal median, 2nd-degree polynomial, single-sine harmonic, LightGBM
    (n_estimators=50, defaults) — exactly the four they compared.
  - Metrics: RMSE, MAE, R^2 over all (predicted, observed) pairs, plus the fraction
    of targets each method could actually define given clear-sky sparsity.

The point is NOT to adopt gap-filling but to quantify, on our own data, how well
reconstruction would do — so the manuscript can defend dropping it rather than
merely asserting it.

Usage:
    python scripts/04_benchmarks/03_gapfill_ablation_aires.py [--max-fields 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ on path
from utils.config import get_config

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

WINDOW_DAYS = 15            # +/-15 d, the window Aires et al. selected
TEST_FRAC = 0.25            # held-out fraction simulating cloud gaps
EVI_CLIP = (-0.2, 1.2)      # plausible EVI bounds for guarding extrapolation


def _harmonic(t, A, w, phi, C):
    return A * np.sin(w * t + phi) + C


def fit_predict(train_x, train_y, t):
    """Return dict method -> prediction at day `t`, or np.nan if undefined.

    train_x: day-of-series (int days), train_y: EVI, already restricted to the
    +/-15-d window around t.
    """
    n = len(train_x)
    out = {"median": np.nan, "poly2": np.nan, "harmonic": np.nan, "lightgbm": np.nan}
    if n == 0:
        return out

    # --- temporal median (Aires: robust, ignores temporal order) ---
    out["median"] = float(np.median(train_y))

    # --- 2nd-degree polynomial (Aires: their best method) ---
    if len(np.unique(train_x)) >= 3:
        coeffs = np.polyfit(train_x, train_y, 2)
        out["poly2"] = float(np.clip(np.polyval(coeffs, t), *EVI_CLIP))
    else:
        out["poly2"] = out["median"]  # graceful fallback (their median fallback logic)

    # --- single-sine harmonic ---
    if n >= 4 and np.ptp(train_x) > 0:
        try:
            p0 = [np.ptp(train_y) / 2 or 0.1, 2 * np.pi / 365.0,
                  0.0, float(np.mean(train_y))]
            popt, _ = curve_fit(_harmonic, train_x, train_y, p0=p0, maxfev=2000)
            out["harmonic"] = float(np.clip(_harmonic(t, *popt), *EVI_CLIP))
        except Exception:
            out["harmonic"] = out["median"]
    else:
        out["harmonic"] = out["median"]

    # --- LightGBM (Aires: defaults, n_estimators=50) ---
    if n >= 3:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=50, num_leaves=31, learning_rate=0.1,
            max_depth=-1, verbose=-1,
        )
        model.fit(train_x.reshape(-1, 1), train_y)
        out["lightgbm"] = float(np.clip(
            model.predict(np.array([[t]]))[0], *EVI_CLIP))
    else:
        out["lightgbm"] = out["median"]

    return out


def metrics(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    m = ~np.isnan(pred)
    obs, pred = obs[m], pred[m]
    if len(obs) < 2:
        return dict(n=len(obs), RMSE=np.nan, MAE=np.nan, R2=np.nan, coverage=np.nan)
    err = pred - obs
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return dict(
        n=int(len(obs)),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        MAE=float(np.mean(np.abs(err))),
        R2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fields", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hls", default=None,
                    help="override path to hls_phenology_merged.parquet")
    ap.add_argument("--out", default=None, help="override results CSV path")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = get_config()
    src = args.hls or cfg["paths"]["hls_merged"]
    if not Path(src).exists():
        # config.local.yaml may be stale (pre-rename WheatFlagLeaf path);
        # fall back to the verified depot location.
        fallback = ("/depot/ciampitti/data/WheatPhenologyHRW/"
                    "data/processed/buffer/hls_phenology_merged.parquet")
        print(f"[warn] {src} missing; falling back to {fallback}")
        src = fallback
    print(f"[load] {src}")
    df = pd.read_parquet(src, columns=["field_id", "date", "EVI"])
    # Same EVI cleaning the phenometric pipeline applies before curve fitting
    # (02_features/01: |EVI| > 1 -> NaN). Keeps the ablation apples-to-apples.
    df.loc[df["EVI"].abs() > 1, "EVI"] = np.nan
    df = df.dropna(subset=["EVI"]).copy()
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)

    fields = df["field_id"].unique()
    sample = rng.choice(fields, size=min(args.max_fields, len(fields)),
                        replace=False)
    df = df[df["field_id"].isin(sample)].sort_values(["field_id", "t"])
    print(f"[sample] {len(sample)} fields, {len(df):,} clear-sky EVI obs")

    methods = ["median", "poly2", "harmonic", "lightgbm"]
    obs_all, pred_all = [], {m: [] for m in methods}
    n_targets = 0
    t0 = time.time()

    for fid, g in df.groupby("field_id", sort=False):
        x = g["t"].to_numpy()
        y = g["EVI"].to_numpy()
        if len(x) < 6:
            continue
        is_test = rng.random(len(x)) < TEST_FRAC
        if not is_test.any():
            continue
        tr_x, tr_y = x[~is_test], y[~is_test]
        for ti, yi in zip(x[is_test], y[is_test]):
            wmask = np.abs(tr_x - ti) <= WINDOW_DAYS
            preds = fit_predict(tr_x[wmask], tr_y[wmask], ti)
            obs_all.append(yi)
            for m in methods:
                pred_all[m].append(preds[m])
            n_targets += 1

    dt = time.time() - t0
    print(f"[done] {n_targets:,} held-out targets reconstructed in {dt:.0f}s\n")

    obs_arr = np.asarray(obs_all)
    active = obs_arr > 0.2   # peak/active-season subset (comparable to Aires'
                             # greenup/maturity stage-stratified validation)

    rows = []
    for m in methods:
        p = np.asarray(pred_all[m])
        r = metrics(obs_arr, p)
        r["coverage"] = r["n"] / n_targets if n_targets else np.nan
        ra = metrics(obs_arr[active], p[active])
        r["RMSE_active"] = ra["RMSE"]
        r["R2_active"] = ra["R2"]
        r["method"] = m
        rows.append(r)
    res = pd.DataFrame(rows)[
        ["method", "n", "coverage", "RMSE", "MAE", "R2",
         "RMSE_active", "R2_active"]
    ].sort_values("RMSE").reset_index(drop=True)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"all targets n={n_targets:,} | "
          f"active-season (EVI>0.2) n={int(active.sum()):,} "
          f"({active.mean():.0%})\n")
    print(res.to_string(index=False))

    out = args.out or str(Path(src).parent.parent.parent
                          / "results" / "gapfill_ablation_aires.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
