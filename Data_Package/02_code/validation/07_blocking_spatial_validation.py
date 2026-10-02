# -*- coding: utf-8 -*-
"""
07_blocking_spatial_validation.py
Blocking and spatial validation for WMR-25-1259.R3 (deterministic, no early stopping).
Run from the Data_Package root:  python 02_code/validation/07_*.py
Inputs: 01_data/*.csv ; best hyperparameters from 05_environment/best_params_summary.json
Outputs: ./04_outputs/validation/ (LOOCV, no-ID, lagged baseline, lag substitution)
"""
import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.formula.api as smf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, '01_data')
OUT = os.path.join(ROOT, '04_outputs', 'validation')
os.makedirs(OUT, exist_ok=True)

DYN = ['PerCapitaGDP', 'SecondaryIndustryShare', 'TertiaryIndustryShare',
       'FiscalCapacityPerCapita', 'ConstructionIntensity', 'UrbanizationRate',
       'PatentsPerCapita', 'EnergyIntensity', 'PolicyCount', 'PublicAwarenessIndex']
SEED = 42
N_EST = 2000
MIN_TRAIN = 7

bp = json.load(open(os.path.join(ROOT, '05_environment', 'best_params_summary.json')))
BP = {k: v for k, v in bp['final_best_hyperparameters_XGBoost_M2_Full'].items()}

df0 = pd.read_csv(os.path.join(DATA, 'final_dataset_lpc_robust_scaled.csv'))
df0['City'] = df0['City'].astype('category')
df0['Year'] = df0['Year'].astype('category')
years = sorted(df0['Year'].unique())

def outer_folds(tmp):
    return [(
        tmp.index[tmp['Year'].isin(years[:MIN_TRAIN + i])].to_numpy(),
        tmp.index[tmp['Year'] == years[MIN_TRAIN + i]].to_numpy())
        for i in range(len(years) - MIN_TRAIN)]

def fit_pred(tr, te, feats, tmp):
    m = xgb.XGBRegressor(**BP, n_estimators=N_EST, random_state=SEED, n_jobs=-1,
                         verbosity=0, enable_categorical=True, device='cpu')
    m.fit(tmp.loc[tr, feats], tmp.loc[tr, 'Target'])
    return m.predict(tmp.loc[te, feats])

def metrics(y, p):
    v = ~np.isnan(p)
    return r2_score(y[v], p[v]), np.sqrt(mean_squared_error(y[v], p[v]))

# (a) LOOCV
y = df0['Target'].to_numpy(); oof = np.full(len(df0), np.nan)
for c in df0['City'].cat.categories:
    test = df0['City'] == c
    oof[test] = fit_pred(np.flatnonzero(~test.to_numpy()), np.flatnonzero(test.to_numpy()),
                         DYN + ['City', 'Year'], df0)
r2lo, rmselo = metrics(y, oof)

# (b) No-ID
y0 = df0['Target'].to_numpy(); o0 = np.full(len(df0), np.nan)
for ti, ei in outer_folds(df0):
    o0[ei] = fit_pred(ti, ei, DYN, df0)
r2no, rmseno = metrics(y0, o0)

# (c) Lagged-outcome baseline (in-sample OLS, as a persistence benchmark)
df2 = df0.copy()
df2['LagY'] = df2.groupby('City', observed=False)['Target'].shift(1)
df2 = df2.dropna(subset=['LagY'])
mO = smf.ols('Target ~ C(City) + LagY', data=df2).fit()
r2lag = 1 - ((df2['Target'] - mO.fittedvalues) ** 2).sum() / ((df2['Target'] - df2['Target'].mean()) ** 2).sum()

pd.DataFrame([
    {'test': 'LOOCV (41 cities)', 'r2': r2lo, 'rmse': rmselo},
    {'test': 'no City/Year IDs', 'r2': r2no, 'rmse': rmseno},
    {'test': 'lagged-outcome baseline (in-sample OLS)', 'r2': r2lag, 'rmse': np.nan},
]).to_csv(os.path.join(OUT, 'blocking_tests.csv'), index=False, encoding='utf-8-sig')
print('blocking tests:', r2lo, r2no, r2lag)

# (d) Lag substitution (PolicyCount and Awareness replaced by one-year lags)
df1 = df0.copy()
for c in ['PolicyCount', 'PublicAwarenessIndex']:
    df1[c + '_L1'] = df1.groupby('City', observed=False)[c].shift(1)
df1 = df1.dropna(subset=['PolicyCount_L1', 'PublicAwarenessIndex_L1'])
uy = sorted(df1['Year'].unique())
oc = []
for i in range(len(uy) - MIN_TRAIN):
    oc.append((np.flatnonzero(df1['Year'].isin(uy[:MIN_TRAIN + i]).to_numpy()),
               np.flatnonzero((df1['Year'] == uy[MIN_TRAIN + i]).to_numpy())))

def run_var(feats):
    yy = df1['Target'].to_numpy(); o = np.full(len(df1), np.nan)
    for ti, ei in oc:
        m = xgb.XGBRegressor(**BP, n_estimators=N_EST, random_state=SEED, n_jobs=-1,
                             verbosity=0, enable_categorical=True, device='cpu')
        m.fit(df1.iloc[ti][feats], yy[ti])
        o[ei] = m.predict(df1.iloc[ei][feats])
    return metrics(yy, o)

base = run_var(DYN + ['City', 'Year'])
pL = run_var([c if c != 'PolicyCount' else 'PolicyCount_L1' for c in DYN] + ['City', 'Year'])
aL = run_var([c if c != 'PublicAwarenessIndex' else 'PublicAwarenessIndex_L1' for c in DYN] + ['City', 'Year'])
pd.DataFrame([
    {'variant': 'baseline (contemporaneous)', 'r2': base[0], 'rmse': base[1]},
    {'variant': 'PolicyCount replaced by lag', 'r2': pL[0], 'rmse': pL[1]},
    {'variant': 'PublicAwarenessIndex replaced by lag', 'r2': aL[0], 'rmse': aL[1]},
]).to_csv(os.path.join(OUT, 'lagged_sensitivity.csv'), index=False, encoding='utf-8-sig')
print('lag substitution:', base, pL, aL)