"""Raw-signal VI-only sequence model (internal context only).

A Temporal Convolutional Network and a two-layer LSTM trained on the
daily NDVI/EVI/GCVI series alone, without the Wang-Engel-Streck, Daymet,
MODIS-LST or state features. Retained as an internal context script; the
comparison reported in the paper (Supplementary S9) is the matched-input
evaluation in 08_seq_dl_hybrid.py, where the learned representation is
combined with the same auxiliary features the engineered models use.
Needs a GPU. Output: <work_dir>/v3_results/seq_dl_baseline.csv
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
from scripts.utils.deep_models import load_cohort, stage_frame, r2, boot_ci, SEEDS
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
import torch, torch.nn as nn

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
HLS = WORK / 'hls_full_2013_2024.parquet'
OUT = WORK / 'v3_results'; OUT.mkdir(parents=True, exist_ok=True)
VIS = ['NDVI', 'EVI', 'GCVI']; T = 365
REPRO = ['emergence', 'tillering', 'jointing', 'flag_leaf',
         'boot', 'heading', 'anthesis', 'maturity']
ENG = {'emergence': 0.36, 'tillering': 0.34, 'jointing': 0.33, 'flag_leaf': 0.71,
       'boot': 0.69, 'heading': 0.73, 'anthesis': 0.82, 'maturity': 0.44}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def _smooth(dos, vals, window=15, poly=2):
    if len(dos) < 5:
        return None
    i = np.argsort(dos); ds, vs = np.asarray(dos)[i], np.asarray(vals)[i]
    u = np.unique(ds); uv = np.array([vs[ds == d].mean() for d in u])
    daily = np.interp(np.arange(1, T + 1), u, uv)
    w = min(window, len(daily)); w -= (w % 2 == 0)
    return daily if w < 4 else savgol_filter(daily, w, poly)


def build_seqs():
    h = pd.read_parquet(HLS, columns=['field_id', 'harvest_year', 'dos'] + VIS)
    h['field_id'] = h['field_id'].astype(str)
    h = h[h['harvest_year'].isin(CFG.study_area.harvest_years)]
    seqs = {}
    for (fid, hy), g in h.groupby(['field_id', 'harvest_year']):
        ch = []
        for v in VIS:
            sub = g[['dos', v]].dropna()
            s = _smooth(sub['dos'].values, sub[v].values) if len(sub) >= 5 else None
            if s is None:
                ch = None; break
            ch.append(s.astype('float32'))
        if ch is not None:
            seqs[(fid, int(hy))] = np.stack(ch, 1)
    print(f'built {len(seqs)} field-year sequences', flush=True)
    return seqs


class TempCNN(nn.Module):
    def __init__(s, c=3, t=T, f=64, k=5, d=.2):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(d))
        s.cnn = nn.Sequential(blk(c, f), blk(f, f), blk(f, f))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(f * t, 256), nn.ReLU(),
                               nn.Dropout(d), nn.Linear(256, 1))
    def forward(s, x):
        return s.head(s.cnn(x.transpose(1, 2)))


class LSTMReg(nn.Module):
    def __init__(s, c=3, h=64):
        super().__init__()
        s.lstm = nn.LSTM(c, h, num_layers=2, batch_first=True, dropout=.2)
        s.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(),
                               nn.Dropout(.2), nn.Linear(64, 1))
    def forward(s, x):
        o, _ = s.lstm(x); return s.head(o[:, -1])


def _train_pred(Mk, Xtr, ytr, Xva, yva, Xte, seed, mx=150, pat=20):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Mk().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev).view(-1, 1)
    Xv = torch.tensor(Xva, device=dev); n = len(Xt); bs = 128
    best = 1e18; bad = 0; bw = None
    for ep in range(mx):
        m.train(); pm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = pm[i:i + bs]; opt.zero_grad()
            nn.functional.mse_loss(m(Xt[j]), yt[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(m(Xv).cpu().view(-1), torch.tensor(yva)).item()
        if v < best - 1e-4:
            best = v; bad = 0
            bw = {k: p.detach().clone() for k, p in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    if bw:
        m.load_state_dict(bw)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte, device=dev)).cpu().numpy().ravel()


def _loyo(d, seqs, tgt, Mk):
    P, Tr = [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        iy = sorted(tr['year'].unique()); vy = iy[-1]
        it, iv = tr[tr['year'] != vy], tr[tr['year'] == vy]

        def pack(df):
            X, y = [], []
            for _, r in df.iterrows():
                s = seqs.get((r['field_id'], int(r['year'])))
                if s is not None:
                    X.append(s); y.append(r[tgt])
            return np.array(X, 'float32'), np.array(y, 'float32')
        Xtr, ytr = pack(it); Xva, yva = pack(iv)
        Xall, yall = pack(tr); Xte, yte = pack(te)
        if len(Xtr) < 50 or len(Xte) < 5 or len(Xva) < 5:
            continue
        mu = Xall.reshape(-1, 3).mean(0); sd = Xall.reshape(-1, 3).std(0) + 1e-6
        ny, sy = yall.mean(), yall.std() + 1e-6
        nm = lambda A: ((A - mu) / sd).astype('float32')
        ps = [_train_pred(Mk, nm(Xtr), (ytr - ny) / sy, nm(Xva),
                          (yva - ny) / sy, nm(Xte), s) * sy + ny for s in SEEDS]
        P.extend(np.mean(ps, 0)); Tr.extend(yte)
    return np.array(Tr), np.array(P)


def main():
    t0 = time.time()
    fe, _ = load_cohort(WORK, PHENO)
    seqs = build_seqs()
    rows = []
    for s in REPRO:
        d, tgt = stage_frame(fe, s)
        for mn, Mk in [('TempCNN', TempCNN), ('LSTM', LSTMReg)]:
            Tr, P = _loyo(d, seqs, tgt, Mk)
            if len(Tr) < 5:
                continue
            R = r2(Tr, P); lo, hi = boot_ci(Tr, P)
            rows.append(dict(stage=s, model=mn, R2=round(R, 4),
                             R2_lo=round(lo, 4), R2_hi=round(hi, 4),
                             n=len(Tr), eng_best=ENG[s],
                             d_vs_eng=round(R - ENG[s], 4)))
            print(f"{s:10s} {mn:8s} R2={R:.3f} CI[{lo:.3f},{hi:.3f}] "
                  f"eng={ENG[s]:.2f} d={R - ENG[s]:+.3f} n={len(Tr)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'seq_dl_baseline.csv', index=False)
    df = pd.DataFrame(rows)
    print('\n=== sequence-DL vs engineered + physiology (adopted best) ===',
          flush=True)
    print(df.to_string(index=False), flush=True)
    if len(df):
        print(f"\nseq-DL beats engineered best in "
              f"{int((df.d_vs_eng > 0).sum())}/{len(df)} cells", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
