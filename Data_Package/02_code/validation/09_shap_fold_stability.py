# -*- coding: utf-8 -*-
"""09_shap_rank_stability.py
Fold/window-level SHAP rank stability (response item 8): the GLOBAL XGBoost M2-Full model
(full-sample fit, see 03_model_training.py) is applied to each outer test year separately,
and mean |SHAP| per dynamic feature is ranked per window.
Run from Data_Package root; outputs to 04_outputs/validation/.
"""
import os, pickle
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, '01_data')
OUT = os.path.join(ROOT, '04_outputs', 'validation')
os.makedirs(OUT, exist_ok=True)

DYN = ['PerCapitaGDP', 'SecondaryIndustryShare', 'TertiaryIndustryShare',
       'FiscalCapacityPerCapita', 'ConstructionIntensity', 'UrbanizationRate',
       'PatentsPerCapita', 'EnergyIntensity', 'PolicyCount', 'PublicAwarenessIndex']

df0 = pd.read_csv(os.path.join(DATA, 'final_dataset_lpc_robust_scaled.csv'))
df0['City'] = df0['City'].astype('category')
df0['Year'] = df0['Year'].astype('category')
years = sorted(df0['Year'].unique())
with open(os.path.join(ROOT, '03_models', 'XGBoost-M2_Full_L1-Native.pkl'), 'rb') as f:
    pipe = pickle.load(f)
model = pipe['model']
dfx = df0[pipe['features']]
dyidx = [pipe['features'].index(c) for c in DYN]

cols = {}
for i in range(len(years) - 7):
    te = df0.index[df0['Year'] == years[7 + i]].to_numpy()
    dm = xgb.DMatrix(dfx.loc[te], enable_categorical=True)
    cc = model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    cols['fold%d' % (i + 1)] = np.abs(cc[:, dyidx]).mean(axis=0)

M = pd.DataFrame(cols, index=DYN).reset_index().rename(columns={'index': 'feature'})
M.to_csv(os.path.join(OUT, 'shap_abs_contrib.csv'), index=False, encoding='utf-8-sig')
rank = M.set_index('feature').rank(ascending=False)
rank.corr().reset_index().rename(columns={'index': 'feature'}).to_csv(
    os.path.join(OUT, 'shap_rank_spearman.csv'), index=False, encoding='utf-8-sig')
print('window-level SHAP stability written to', OUT)
