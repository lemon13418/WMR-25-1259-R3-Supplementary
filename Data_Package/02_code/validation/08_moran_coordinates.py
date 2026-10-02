# -*- coding: utf-8 -*-
"""
08_moran_coordinates.py
Residual spatial autocorrelation (Moran's I) with row-standardized weights.
Coordinates: bundled in 04_outputs/coordinates.csv (41 prefecture cities).
Inputs: output OOF residual file (City/Year/resid), produced by 10_diagnostics_supplement.py
        (04_outputs/validation/oof_resid_with_city.csv). Run after script 10.
Outputs: ./04_outputs/validation/morans_I_rowstd.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, '04_outputs', 'validation')
os.makedirs(OUT, exist_ok=True)

cn = pd.read_csv(os.path.join(ROOT, '04_outputs', 'coordinates.csv'))
cc = dict(zip(cn['city'], zip(cn['lat'], cn['lon'])))
resid = pd.read_csv(os.path.join(ROOT, '04_outputs', 'validation', 'oof_resid_with_city.csv'))
cities = sorted(resid['City'].unique())
assert all(c in cc for c in cities), [c for c in cities if c not in cc]
lat = np.array([cc[c][0] for c in cities]); lon = np.array([cc[c][1] for c in cities])

a = np.sin(np.deg2rad(lat[:, None] - lat) / 2) ** 2 + np.cos(np.deg2rad(lat[:, None])) * \
    np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon[:, None] - lon) / 2) ** 2
d = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
Wbin = (d <= 250).astype(float)
np.fill_diagonal(Wbin, 0)
W = Wbin / (Wbin.sum(axis=1, keepdims=True) + 1e-12)  # row-standardized

idx = {c: i for i, c in enumerate(cities)}
resid['ci'] = resid['City'].map(idx)

def moran(z, ci):
    zc = z - z.mean()
    if (zc ** 2).sum() <= 0:
        return np.nan
    Ws = W[np.ix_(ci, ci)]
    return (zc @ Ws @ zc) / (zc @ zc)

rng = np.random.default_rng(42)
rows = []
for yr, g in resid.groupby('Year'):
    z = g['resid'].to_numpy(); ci = g['ci'].to_numpy()
    obs = moran(z, ci)
    perm = np.array([moran(rng.permutation(z), ci) for _ in range(999)])
    rows.append({'year': int(yr), 'n': len(g), 'morans_I': obs,
                 'p_perm': (np.abs(perm) >= np.abs(obs)).mean(), 'EI': -1 / (len(cities) - 1)})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, 'morans_I_rowstd.csv'), index=False, encoding='utf-8-sig')
print(out.round(4).to_string(index=False))