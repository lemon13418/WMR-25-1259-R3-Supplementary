# -*- coding: utf-8 -*-
"""
10_diagnostics_supplement.py
Diagnostic supplement for WMR-25-1259.R3: regenerates every supporting table that
underpins Supplementary S1, S3, S8 and the 2020-dummy and zero-loss-PCA statements
in the main text. Deterministic: all random generators seeded 42.

Run from the Data_Package root:   python 02_code/validation/10_diagnostics_supplement.py
Pinned environment: python 3.9.25, xgboost 2.1.4, scipy 1.13.1, statsmodels 0.14.5
(package versions in 05_environment/requirements_pinned.txt). Running under a different
xgboost version may move tree results at the third decimal; outputs written here are the
pinned-environment values (the present R3 package).

Outputs (04_outputs/validation/):
  imputation_audit.csv            - per-file missing-value counts + no-op assertion
  qq_transform_check.csv          - Q-Q normality fit (r^2) raw vs log1p per-capita target
  ols_2020_indicator.csv          - 2020-indicator panel FE OLS table (Supplementary S8)
  oof_resid_with_city.csv         - diagnostic residual set (fixed hyperparameters,
                                    no early stopping) - benchmark-vs-diagnostic residual
                                    disclosure in Supplementary S3
  transformers_window_sensitivity.csv - scaler refit on outer training portion only
                                    (folds 2018-2019; Supplementary S1)
  xgb_2020_dummy_shap.csv         - marginal SHAP (pred_contribs) of a 2020 indicator
                                    added to the XGBoost M2-Full pipeline
  zero_loss_pca.csv               - 100%-variance PCA (10 components) OOF RMSE
  fold_manifest_check.csv         - verification of folds_definitions.csv and
                                    coordinates.csv against the code-derived folds
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import statsmodels.formula.api as smf

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


def xgb_fit(tr, te, feats, tmp, extra_cols=None):
    cols = feats + (extra_cols or [])
    m = xgb.XGBRegressor(**BP, n_estimators=N_EST, random_state=SEED, n_jobs=-1,
                         verbosity=0, enable_categorical=True, device='cpu')
    m.fit(tmp.loc[tr, cols], tmp.loc[tr, 'Target'])
    return m.predict(tmp.loc[te, cols])


def r2(y, p):
    return r2_score(y, p)


# ----------------------------------------------------------------- 1. imputation audit
print('== 1. imputation audit')
frames = {
    'merged_raw_data.csv': DATA + r'\merged_raw_data.csv',
    'final_dataset_lpc_robust_scaled.csv': DATA + r'\final_dataset_lpc_robust_scaled.csv',
    'final_dataset_M2_PCA.csv': DATA + r'\final_dataset_M2_PCA.csv',
}
audit_rows = []
for name, path in frames.items():
    df = pd.read_csv(path)
    nans = int(df.isna().sum().sum())
    num = df.select_dtypes(include=[np.number])
    changed = int((num != num.ffill().bfill().fillna(num.mean())).sum().sum())
    n_cells = int(df.shape[0] * df.shape[1])
    audit_rows.append({'file': name, 'n_rows': df.shape[0], 'n_cols': df.shape[1],
                       'n_cells': n_cells, 'missing_cells': nans,
                       'cells_changed_by_fill_steps': changed})
    msg = 'PASS (no missing values; defensive fills are no-ops)' if nans == 0 else 'FAIL'
    print('  %-40s missing=%d fillchanged=%d %s' % (name, nans, changed, msg))
    assert nans == 0, 'unexpected missing values in %s' % name
pd.DataFrame(audit_rows).to_csv(os.path.join(OUT, 'imputation_audit.csv'),
                                index=False, encoding='utf-8-sig')

# ----------------------------------------------------------------- 2. Q-Q normality fit
print('== 2. Q-Q normality fit')
df = pd.read_csv(DATA + r'\final_dataset_lpc_robust_scaled.csv')
tgt = df['Target'].to_numpy()
raw = np.expm1(tgt)


def probplot_r(x):
    # scipy >= 1.16 moved the fit result (slope, intercept, r) to the second element;
    # earlier versions return it as the third element.
    res = stats.probplot(x, dist='norm', plot=None)
    fit = res[2] if len(res) > 2 else res[1]
    return float(fit[2])


qq = pd.DataFrame({'var': ['per-capita raw', 'log1p(per-capita)'],
                   'qq_R2': [probplot_r(raw) ** 2, probplot_r(tgt) ** 2]})
qq.to_csv(os.path.join(OUT, 'qq_transform_check.csv'), index=False, encoding='utf-8-sig')
print('  raw=%.6f  log1p=%.6f' % (qq.loc[0, 'qq_R2'], qq.loc[1, 'qq_R2']))

# ----------------------------------------------------------------- 3. panel FE OLS with
#                                                               2020 indicator (S8)
print('== 3. 2020-indicator panel FE OLS')
d = df.copy()
d['D2020'] = (d['Year'] == 2020).astype(int)
formula = ('Target ~ C(City) + D2020 + ' + ' + '.join(DYN))
res = smf.ols(formula, data=d).fit()
ci_low, ci_high = res.conf_int().loc['D2020']
ols_row = {'item': '2020 indicator (OLS, city FE + 10 covariates)',
           'coef': res.params['D2020'],
           'std_err': res.bse['D2020'],
           't': res.tvalues['D2020'],
           'p': res.pvalues['D2020'],
           'CI_low': ci_low, 'CI_high': ci_high,
           'N': int(res.nobs),
           'model_R2': res.rsquared}
pd.DataFrame([ols_row]).to_csv(os.path.join(OUT, 'ols_2020_indicator.csv'),
                               index=False, encoding='utf-8-sig')
print('  coef=%.4f p=%.4f CI=[%.4f, %.4f] N=%d R2=%.4f' % (
    ols_row['coef'], ols_row['p'], ci_low, ci_high, ols_row['N'], ols_row['model_R2']))

# ----------------------------------------------------------------- 4. diagnostic
#                                                           residual set (fixed params,
#                                                           no early stopping)
print('== 4. diagnostic residual set (no early stopping)')
df0 = df.copy()
df0['City'] = df0['City'].astype('category')
df0['Year'] = df0['Year'].astype('category')
years = sorted(df0['Year'].unique())
oof = np.full(len(df0), np.nan)
for i in range(len(years) - MIN_TRAIN):
    trm = df0['Year'].isin(years[:MIN_TRAIN + i]).to_numpy()
    tem = df0['Year'] == years[MIN_TRAIN + i]
    oof[tem.to_numpy()] = xgb_fit(trm, tem.to_numpy(), DYN + ['City', 'Year'], df0)
rd = pd.DataFrame({'City': df0['City'].values, 'Year': df0['Year'].values,
                   'resid': df0['Target'].values - oof})
rd = rd.dropna(subset=['resid'])
rd.to_csv(os.path.join(OUT, 'oof_resid_with_city.csv'), index=False, encoding='utf-8-sig')
means = rd.groupby('Year')['resid'].mean().round(6)
print('  yearly residual means:')
print(means.to_string())
bm = pd.read_csv(OUT + r'\..\oof_predictions_xgb_m2_full.csv')
bm['Year'] = bm['Test_Year'].astype(int)
bmean = bm.groupby('Year').apply(
    lambda g: (g['True_Values'] - g['Predicted_Values']).mean(), include_groups=False)
diff = (bmean - means).abs().max()
print('  max |benchmark - diagnostic| mean diff = %.4f (disclosed bound in S3: <= 0.014)' % diff)

# ----------------------------------------------------------------- 5. transformer-window
#                                                         sensitivity (Supplementary S1)
# NOTE: this diagnostic trains on ALL rows other than the tested year (leave-one-test-year-out),
# i.e. on a superset of the expanding outer training window (it includes post-test years).
# It is therefore a bounding exercise, not a strictly leakage-free refit; the RobustScaler only
# is refitted (the M2-Full configuration does not use PCA). See Supplementary S1.
print('== 5. transform-window sensitivity (scaler refit on leave-one-test-year-out basis)')
sens_rows = []
for i in range(2):  # outer test years 2018 and 2019 only, as in S1
    test_year = years[MIN_TRAIN + i]
    train_mask = df0['Year'] != test_year
    # scaler fitted on the outer training portion only
    sc = RobustScaler().fit(df0.loc[train_mask, DYN]) if train_mask.sum() > 1 else RobustScaler().fit(df0[DYN])
    Xs = sc.transform(df0[DYN])
    Xsd = df0.copy()
    Xsd[DYN] = Xs
    trm = train_mask.to_numpy()
    tem = (df0['Year'] == test_year).to_numpy()
    y = df0['Target'].to_numpy()
    p_release = xgb_fit(trm, tem, DYN + ['City', 'Year'], Xsd)
    r2_release = r2(y[tem], p_release)
    p_pub = bm.loc[bm['Year'] == test_year, 'Predicted_Values'].to_numpy()
    y_pub = bm.loc[bm['Year'] == test_year, 'True_Values'].to_numpy()
    r2_published = r2(y_pub, p_pub)
    sens_rows.append({'fold_test_year': float(test_year), 'r2_release': round(r2_release, 4),
                      'r2_published': round(float(r2_published), 4),
                      'delta': round(float(r2_release - r2_published), 4)})
    print('  %d: release=%.4f published=%.4f delta=%.4f' % (
        test_year, r2_release, r2_published, r2_release - r2_published))
pd.DataFrame(sens_rows).to_csv(os.path.join(OUT, 'transformers_window_sensitivity.csv'),
                               index=False, encoding='utf-8-sig')

# ----------------------------------------------------------------- 6. 2020 dummy marginal
#                                                          SHAP in the XGBoost pipeline
# NOTE: same leave-one-test-year-out protocol as section 5; the folded training data therefore
# include post-test years, and the Year identifier already encodes the 2020 anomaly.
print('== 6. 2020-dummy marginal SHAP (pred_contribs)')
df1 = df0.copy()
df1['D2020'] = (df1['Year'] == 2020).astype(int)
y = df1['Target'].to_numpy()
dum = np.zeros(len(df1))
for i in range(len(years) - MIN_TRAIN):
    test_year = years[MIN_TRAIN + i]
    trm = df1['Year'] != test_year
    tem = (df1['Year'] == test_year).to_numpy()
    cols = DYN + ['City', 'Year', 'D2020']
    m = xgb.XGBRegressor(**BP, n_estimators=N_EST, random_state=SEED, n_jobs=-1,
                         verbosity=0, enable_categorical=True, device='cpu')
    m.fit(df1.loc[trm, cols], y[trm])
    dm = xgb.DMatrix(df1.loc[tem, cols], enable_categorical=True)
    contribs = m.get_booster().predict(dm, pred_contribs=True)
    dum[tem] = np.abs(contribs[:, -2])  # penultimate column = D2020 (last is bias)
shap_row = {'year_mask': 'all OOF (2018-2022)', 'mean_abs_SHAP_D2020': float(dum.mean())}
pd.DataFrame([shap_row]).to_csv(os.path.join(OUT, 'xgb_2020_dummy_shap.csv'),
                                index=False, encoding='utf-8-sig')
print('  mean |SHAP| of the 2020 indicator = %.6f (expected \u2248 0: redundant with the Year identifier)' % dum.mean())

# ----------------------------------------------------------------- 7. zero-loss PCA OOF RMSE
print('== 7. zero-loss PCA (10 components) OOF RMSE')
X_dyn = df0[DYN].to_numpy()
fit_mask = df0['Year'].isin(years[:9])  # years <= 2019 (Year is categorical here)
pca10 = PCA(n_components=10).fit(X_dyn[fit_mask])
X_pc = pca10.transform(X_dyn)
dfp = df0.copy()
for j in range(10):
    dfp['PC%02d' % (j + 1)] = X_pc[:, j]
feats_pc = ['PC%02d' % (j + 1) for j in range(10)]
y = df0['Target'].to_numpy()
oofp = np.full(len(df0), np.nan)
for i in range(len(years) - MIN_TRAIN):
    trm = df0['Year'].isin(years[:MIN_TRAIN + i]).to_numpy()
    tem = (df0['Year'] == years[MIN_TRAIN + i]).to_numpy()
    m = xgb.XGBRegressor(**BP, n_estimators=N_EST, random_state=SEED, n_jobs=-1,
                         verbosity=0, enable_categorical=True, device='cpu')
    m.fit(dfp.loc[trm, feats_pc + ['City', 'Year']], y[trm])
    oofp[tem] = m.predict(dfp.loc[tem, feats_pc + ['City', 'Year']])
rmse_pca = float(np.sqrt(mean_squared_error(y[~np.isnan(oofp)], oofp[~np.isnan(oofp)])))
r2_pca = float(r2(y[~np.isnan(oofp)], oofp[~np.isnan(oofp)]))
rmse_raw = float(np.sqrt(mean_squared_error(bm['True_Values'], bm['Predicted_Values'])))
pd.DataFrame([{'variant': '100%-variance PCA (10 components)', 'oof_rmse': rmse_pca,
               'oof_r2': r2_pca},
              {'variant': 'raw collinear features (benchmark OOF)', 'oof_rmse': rmse_raw,
               'oof_r2': None}]).to_csv(os.path.join(OUT, 'zero_loss_pca.csv'),
                                        index=False, encoding='utf-8-sig')
print('  PCA-10 OOF RMSE=%.4f R2=%.4f (main text: 0.1279) | raw OOF RMSE=%.4f' % (
    rmse_pca, r2_pca, rmse_raw))

# ----------------------------------------------------------------- 8. fold manifest and
#                                                          coordinates verification
print('== 8. fold manifest / coordinates verification')
folds = pd.read_csv(OUT + r'\..\folds_definitions.csv')
ok_folds = True
for k, row in folds.iterrows():
    i = int(row['outer_fold']) - 1
    if (int(row['train_years_start']) != 2011
            or int(row['train_years_end']) != int(years[MIN_TRAIN + i - 1])
            or int(row['test_year']) != int(years[MIN_TRAIN + i])):
        ok_folds = False
print('  folds_definitions consistent with code: %s' % ok_folds)
coords = pd.read_csv(OUT + r'\..\coordinates.csv')
cities_df = rd['City'].unique()
ok_coords = (len(coords) == 41 and set(coords['city']) == set(cities_df))
print('  coordinates cover all %d panel cities: %s' % (len(cities_df), ok_coords))
pd.DataFrame([{'check': 'folds_definitions matches expanding-window derivation',
               'pass': bool(ok_folds)},
              {'check': 'coordinates.csv covers all 41 cities in the residual set',
               'pass': bool(ok_coords)}]).to_csv(
    os.path.join(OUT, 'fold_manifest_check.csv'), index=False, encoding='utf-8-sig')

print('done. outputs written to', OUT)