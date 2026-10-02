# -*- coding: utf-8 -*-
"""
11_transform_window_isolated.py
Isolated transformer-window sensitivity check for WMR-25-1259.R3.

The benchmark RobustScaler is fitted once on the first nine years (2011-2019)
and applied to the full sample, so the outer test years 2018-2019 fall inside
the transform window. Script 10 (S1) already provides a bounding re-fit on a
leave-one-test-year-out basis (a superset of the expanding training window, and
with the model retrained on that superset). This script isolates the pure
transform-window effect: fold definitions, training rows, test rows and
hyperparameters are identical to the benchmark OOF protocol; ONLY the scaler
fitting window changes.

Implementation note. Refitting a RobustScaler on a row subset of the RAW
features is exactly equivalent to recomputing, on that same row subset, the
median and IQR of the RELEASED scaled values and applying the affine
re-standardisation z' = (z - med)/iqr per column. The released scaled file is
therefore used directly and no raw-value reconstruction is involved; all ten
columns (including ConstructionIntensity) are handled identically. Columns
whose IQR is zero within a training window are left unchanged (scale = 1),
mirroring RobustScaler.

  variant_fixed: released scaled file as-is (scaler fitted on 2011-2019)
  variant_clean: per-fold scaler fitted on the outer training years only
                 (2011..test_year-1), i.e. a strictly leakage-free refit

Delta = r2_clean - r2_fixed isolates the contribution of the fixed reference
window to the reported out-of-fold metrics. Fixed hyperparameters, no early
stopping (same convention as scripts 07-10).

Run from the Data_Package root:
   python 02_code/validation/11_transform_window_isolated.py
Output: 04_outputs/validation/transformers_window_isolated.csv
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, '01_data')
OUT = os.path.join(ROOT, '04_outputs', 'validation')
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_EST = 2000
MIN_TRAIN = 7

DYN = ['PerCapitaGDP', 'SecondaryIndustryShare', 'TertiaryIndustryShare',
       'FiscalCapacityPerCapita', 'ConstructionIntensity', 'UrbanizationRate',
       'PatentsPerCapita', 'EnergyIntensity', 'PolicyCount', 'PublicAwarenessIndex']

bp = json.load(open(os.path.join(ROOT, '05_environment', 'best_params_summary.json')))
BP = dict(bp['final_best_hyperparameters_XGBoost_M2_Full'])


def run_oof(scaled, scaler_mode):
    """Benchmark OOF protocol; scaler_mode = 'fixed' | 'per_fold'."""
    df = scaled.copy()
    df['City'] = df['City'].astype('category')
    df['Year'] = df['Year'].astype('category')
    years = sorted(df['Year'].unique().astype(int))

    oof = np.full(len(df), np.nan)
    for i in range(len(years) - MIN_TRAIN):
        tr_years = years[:MIN_TRAIN + i]
        te_year = years[MIN_TRAIN + i]
        trm = df['Year'].isin(tr_years).to_numpy()
        tem = (df['Year'] == te_year).to_numpy()

        if scaler_mode == 'per_fold':
            Xw = df.copy()
            z = Xw[DYN]
            med = z.loc[trm].median()
            q75 = z.loc[trm].quantile(0.75)
            q25 = z.loc[trm].quantile(0.25)
            iqr = (q75 - q25).replace(0.0, 1.0)
            Xw[DYN] = (z - med) / iqr
        else:
            Xw = df

        m = xgb.XGBRegressor(**BP, n_estimators=N_EST, random_state=SEED, n_jobs=-1,
                             verbosity=0, enable_categorical=True, device='cpu')
        m.fit(Xw.loc[trm, DYN + ['City', 'Year']], Xw.loc[trm, 'Target'])
        oof[tem] = m.predict(Xw.loc[tem, DYN + ['City', 'Year']])

    y = df['Target'].to_numpy()
    rows = []
    for i in range(len(years) - MIN_TRAIN):
        tem = (df['Year'] == years[MIN_TRAIN + i]).to_numpy()
        rows.append({'test_year': years[MIN_TRAIN + i],
                     'r2': r2_score(y[tem], oof[tem]),
                     'rmse': float(np.sqrt(mean_squared_error(y[tem], oof[tem])))})
    ok = ~np.isnan(oof)
    rows.append({'test_year': 'overall',
                 'r2': r2_score(y[ok], oof[ok]),
                 'rmse': float(np.sqrt(mean_squared_error(y[ok], oof[ok])))})
    return pd.DataFrame(rows)


def main():
    scaled = pd.read_csv(os.path.join(DATA, 'final_dataset_lpc_robust_scaled.csv'))
    a = run_oof(scaled, 'fixed')
    b = run_oof(scaled, 'per_fold')

    res = pd.DataFrame({
        'fold_test_year': a['test_year'],
        'r2_fixed_window': a['r2'].round(4),
        'r2_train_years_only': b['r2'].round(4),
        'delta_r2': (b['r2'] - a['r2']).round(4),
        'rmse_fixed_window': a['rmse'].round(4),
        'rmse_train_years_only': b['rmse'].round(4),
    })
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUT, 'transformers_window_isolated.csv'),
               index=False, encoding='utf-8-sig')
    print('written:', os.path.join(OUT, 'transformers_window_isolated.csv'))


if __name__ == '__main__':
    main()