# Supplementary Materials for WMR-25-1259 (Revision 3)

## S1. Evaluation protocols, transforms and consistency notes

The benchmark tables in the main text use nested cross-validation (expanding-window outer folds testing 2018–2022) with out-of-fold ("OOF") predictions. RobustScaler and PCA are fitted on the first nine years (2011–2019) and applied to the full sample; outer test years 2018–2019 therefore fall inside the transform window. A sensitivity check re-fits the RobustScaler on a leave-one-test-year-out basis — i.e., on all rows other than the tested year, a superset of the expanding outer training windows — and retrains XGBoost M2-Full with fixed hyperparameters and no early stopping (n_estimators = 2000); the M2-Full configuration does not use PCA, so no PCA refit is involved (released script `10_diagnostics_supplement.py`). Because the refit window includes post-test years, the check is a bounding exercise rather than a strictly leakage-free refit:

| fold_test_year | r2_release | r2_published | delta |
|---|---|---|---|
| 2018.0000 | 0.9664 | 0.9428 | 0.0236 |
| 2019.0000 | 0.9571 | 0.9465 | 0.0106 |

Fold-level R² changes (max 0.024) are unidirectional and small: the re-fitted specifications perform equal or slightly better on both tested years, which does not support the concern that the fixed window inflates the reported benchmark or biases the conclusions. Values are those produced by the released script; running under another xgboost version may move them at the third decimal (Reproducibility Note). Fig. 2C presents a complementary component-count scan under repeated 5-fold cross-validation (10 dynamic features; PCA fitted on 2011–2019), a diagnostic protocol distinct from the OOF benchmark; the raw-feature baseline RMSE is 0.208 and PCA configurations range from 0.28 to 0.55. The 95% variance threshold is first exceeded at seven components (0.949 at six components, 0.973 at seven).

**Isolated transform-window check.** The bounding check above changes two things at once (the scaler window *and* the rows used to train the model, which become a superset of the benchmark training windows). To isolate the pure transform-window effect, the released script `11_transform_window_isolated.py` re-fits the RobustScaler only on the outer *training years* of each fold (2011–2017 for the 2018 test year, …, 2011–2021 for 2022) — a strictly leakage-free refit — while keeping the fold definitions, training rows, test rows, hyperparameters and stopping rule byte-identical to the benchmark OOF protocol; only the scaler fitting window changes. Because refitting a RobustScaler on a row subset of the raw features is affine-equivalent to recomputing the median and IQR of the released (already scaled) values on that same subset and re-standardising z′ = (z − med)/iqr per column, the check operates directly on `final_dataset_lpc_robust_scaled.csv` and requires no raw-value reconstruction (columns with zero within-window IQR are left unchanged, mirroring RobustScaler's scale = 1 convention):

| fold_test_year | r2_fixed_window | r2_train_years_only | delta |
|---|---|---|---|
| 2018 | 0.9438 | 0.9440 | 0.0003 |
| 2019 | 0.9488 | 0.9488 | 0.0000 |
| 2020 | 0.8894 | 0.8894 | 0.0000 |
| 2021 | 0.9515 | 0.9515 | 0.0000 |
| 2022 | 0.9659 | 0.9657 | −0.0003 |
| overall | 0.9408 | 0.9408 | 0.0000 |

The absolute change never exceeds 0.0003 in R² (≤ 0.0004 in RMSE), three orders of magnitude below the bounding check above; the signs are mixed (2018 improves, 2022 worsens, intermediate folds unaffected). The fixed reference window therefore neither inflates nor materially perturbs any reported benchmark figure, and the strictly leakage-free and published designs are empirically interchangeable on this panel (values above are environment-internal: both variants are computed identically in one run, so version/device effects cancel; the fixed-window figures as such are not substitutes for the published OOF file, which remains the authority for all performance statistics).

**Missing-value audit.** The three released datasets (`merged_raw_data.csv`, `final_dataset_lpc_robust_scaled.csv`, `final_dataset_M2_PCA.csv`) contain no missing values, and the defensive fill steps in `feature_engineering.py` alter no cell (`imputation_audit.csv`, produced by script 10). Imputation-based information leakage is therefore precluded by construction for the released panel. The isolated transform-window check (see above) is produced by script `11_transform_window_isolated.py`.


## S2. Final model benchmark (OOF, 17 configurations, log-scale)

Six model families were evaluated under three feature configurations (18 conceivable family–configuration combinations), of which 15 were estimated; Table S2 reports 17 rows because several families are listed with multiple encoding variants. EBM-M2-Full was not run (runtime considerations), and the M2-PCA configurations of the linear families (MLR, SVR) were not estimated; these three cells are reported as gaps rather than estimated values. All metrics are reported on the log-transformed target scale.


| Algorithm | Feature set | Encoding | R2 | RMSE | MAE |
|---|---|---|---|---|---|
| DNN | M2-PCA | Native | 0.8578 | 0.1911 | 0.1335 |
| DNN | M1-Baseline | Native | 0.8433 | 0.2006 | 0.1454 |
| DNN | M1-Baseline | OHE | 0.8074 | 0.2224 | 0.1580 |
| XGBoost | M1-Baseline | Native | 0.9014 | 0.1592 | 0.1063 |
| XGBoost | M2-Full | Native | 0.9382 | 0.1260 | 0.0872 |
| XGBoost | M2-PCA | Native | 0.9304 | 0.1338 | 0.0921 |
| DNN | M2-Full | Native | 0.8922 | 0.1664 | 0.1129 |
| DNN | M2-Full | OHE | 0.8441 | 0.2001 | 0.1339 |
| EBM | M1-Baseline | Native | 0.9225 | 0.1411 | 0.0991 |
| SVR | M1-Baseline | OHE | 0.8819 | 0.1742 | 0.1272 |
| MLR | M1-Baseline | OHE | 0.8496 | 0.1965 | 0.1393 |
| EBM | M2-PCA | Native | 0.8299 | 0.2090 | 0.1517 |
| SVR | M2-Full | OHE | 0.6977 | 0.2787 | 0.1816 |
| LGBM | M2-Full | Native | 0.5891 | 0.3249 | 0.2293 |
| MLR | M2-Full | OHE | 0.5268 | 0.3486 | 0.3152 |
| LGBM | M2-PCA | Native | 0.4989 | 0.3588 | 0.2366 |
| LGBM | M1-Baseline | Native | 0.3277 | 0.4156 | 0.2689 |

## S3. Original-scale error metrics (response item 6) and residual diagnostics

OOF predictions were inverse-transformed to the original per-capita scale (expm1) before computing
MAPE/SMAPE; because these are relative errors, the same values apply at the total-volume scale:
MAPE = (1/n)·Σ|yᵢ − ŷᵢ| / yᵢ × 100%,  SMAPE = (1/n)·Σ|yᵢ − ŷᵢ| / ((|yᵢ|+|ŷᵢ|)/2) × 100%.


| year | MAPE_percent | SMAPE_percent | MAE_physical |
|---|---|---|---|
| 2018 | 20.6000 | 17.8500 | 0.2482 |
| 2019 | 21.7400 | 18.6400 | 0.2848 |
| 2020 | 28.7500 | 22.9300 | 0.3365 |
| 2021 | 21.1500 | 16.9100 | 0.2824 |
| 2022 | 15.9400 | 13.2300 | 0.1920 |
| overall | 21.6300 | 17.9100 | 0.2688 |

For the panel diagnostics, residuals were computed with fixed hyperparameters and no early stopping (validation script `10_diagnostics_supplement.py`); their yearly means can differ from the benchmark OOF file by up to ≈ 0.014 (e.g., 2019: −0.031 benchmark vs −0.017 diagnostic), a difference attributable to fitting rules, and is disclosed here explicitly.


## S4. Blocking and baseline tests (response items 2 and 4)

All robustness variants use the XGBoost M2-Full configuration with hyperparameters fixed to the nested-CV optimum and no early stopping. LOOCV trains on all cities except one and predicts that city across all years; the no-identifier model drops City and Year; the lagged-outcome baseline fits OLS of the target on its one-year lag with city fixed effects (in-sample persistence benchmark).


| test | r2 | rmse |
|---|---|---|
| LOOCV (41 cities) | -0.2035 | 0.5551 |
| no City/Year IDs | 0.7748 | 0.2405 |
| lagged-outcome baseline (in-sample OLS) | 0.9650 | — |

The lagged-outcome baseline is an in-sample OLS persistence benchmark; its RMSE is not comparable to the out-of-fold metrics and is therefore not reported (see main text Section 3.3).

## S5. Lag substitution of proxies (response item 10)

One-year lagged versions were substituted (replacing, not adding to) PolicyCount and PublicAwarenessIndex, evaluated on the same 2012–2022 subset (451 rows).


| variant | r2 | rmse |
|---|---|---|
| baseline (451 rows, contemporaneous) | 0.9430 | 0.1215 |
| PolicyCount replaced by lag | 0.9406 | 0.1241 |
| PublicAwarenessIndex replaced by lag | 0.9449 | 0.1196 |

## S6. Residual spatial autocorrelation (response item 11)

Moran’s I on diagnostic OOF residuals, 41 cities, row-standardized 250-km great-circle weights, 999 permutations (seed 42). None of the yearly statistics is significant.


| year | n | morans_I | p_perm | EI |
|---|---|---|---|---|
| 2018.0000 | 41.0000 | -0.0198 | 0.7237 | -0.0250 |
| 2019.0000 | 41.0000 | -0.0239 | 0.6807 | -0.0250 |
| 2020.0000 | 41.0000 | 0.0144 | 0.7768 | -0.0250 |
| 2021.0000 | 41.0000 | -0.0303 | 0.5786 | -0.0250 |
| 2022.0000 | 41.0000 | 0.0596 | 0.2342 | -0.0250 |

## S7. SHAP rank stability across evaluation windows (response item 8)

Hyperparameter importances (Fig. 3C) use Optuna's mean-decrease-impurity evaluator (n_trees = 64, seed = 42) and are deterministic. To assess rank stability, the global XGBoost M2-Full model was applied to each outer test year separately (2018–2022); mean absolute SHAP by feature and window (×10⁻³):

| feature | fold1 | fold2 | fold3 | fold4 | fold5 |
|---|---|---|---|---|---|
| PerCapitaGDP | 12.46 | 13.78 | 13.24 | 13.26 | 14.16 |
| SecondaryIndustryShare | 9.70 | 10.06 | 10.33 | 9.46 | 10.55 |
| TertiaryIndustryShare | 8.77 | 8.87 | 13.42 | 9.36 | 9.26 |
| FiscalCapacityPerCapita | 5.13 | 5.73 | 5.15 | 4.60 | 4.63 |
| ConstructionIntensity | 7.55 | 7.87 | 9.86 | 9.10 | 7.90 |
| UrbanizationRate | 12.94 | 15.43 | 12.84 | 15.00 | 12.98 |
| PatentsPerCapita | 10.04 | 9.83 | 10.15 | 8.05 | 9.72 |
| EnergyIntensity | 20.01 | 17.75 | 24.35 | 23.86 | 24.64 |
| PolicyCount | 14.47 | 15.08 | 19.52 | 14.77 | 13.76 |
| PublicAwarenessIndex | 2.97 | 3.24 | 2.85 | 3.00 | 2.89 |
Rank statistics (1 = highest importance):

| feature | mean_rank | std_rank |
|---|---|---|
| EnergyIntensity | 1.00 | 0.00 |
| PolicyCount | 2.60 | 0.55 |
| UrbanizationRate | 3.20 | 1.30 |
| PerCapitaGDP | 3.60 | 0.89 |
| SecondaryIndustryShare | 5.40 | 0.55 |
| TertiaryIndustryShare | 6.00 | 1.73 |
| PatentsPerCapita | 6.40 | 1.14 |
| ConstructionIntensity | 7.80 | 0.45 |
| FiscalCapacityPerCapita | 9.00 | 0.00 |
| PublicAwarenessIndex | 10.00 | 0.00 |
Pairwise Spearman rank correlations between windows:

| index | feature | fold1 | fold2 | fold3 | fold4 | fold5 |
|---|---|---|---|---|---|---|
| 0 | fold1 | 1.000 | 0.806 | 0.709 | 0.648 | 0.600 |
| 1 | fold2 | 0.806 | 1.000 | 0.745 | 0.770 | 0.709 |
| 2 | fold3 | 0.709 | 0.745 | 1.000 | 0.927 | 0.903 |
| 3 | fold4 | 0.648 | 0.770 | 0.927 | 1.000 | 0.952 |
| 4 | fold5 | 0.600 | 0.709 | 0.903 | 0.952 | 1.000 |
The top tier is stable and consistent with the full-sample ranking (Fig. 5B): Energy Efficiency ranked first in every window, with Policy Count, Urbanization Level, and Economic Development Level following; pairwise Spearman correlations range from 0.60 to 0.95. The per-window mean absolute SHAP values are displayed graphically in Supplementary Figure S1 (file `SI_Fig_SHAP_fold_stability`, provided as PNG and PDF in `SI_Figures/`), using the substantive feature labels consistent with Figures 2/5/6 of the main text.

## S8. Pandemic (2020) indicator test

Panel fixed-effects OLS (city dummies + 2020 indicator + ten dynamic covariates):


| item | coef | std_err | t | p | CI_low | CI_high | N | model_R2 |
|---|---|---|---|---|---|---|---|---|
| 2020 indicator (OLS, city FE + 10 covariates) | -0.0285 | 0.0213 | -1.3400 | 0.1808 | -0.0703 | 0.0133 | 492 | 0.9492 |

In the XGBoost pipeline the marginal SHAP contribution of the 2020 indicator is ≈ 0.000 (about 1.4 × 10⁻⁵, computed with `pred_contribs` by script 10; `xgb_2020_dummy_shap.csv`): in the outer folds with test years 2018–2020 the indicator is constant zero within the training portion, while in the folds with test years 2021–2022 its training portions include 2020 and the indicator is exactly redundant with the Year identifier already in the feature set; no informative split on it can therefore be learned, and the 2020 anomaly is absorbed by the Year identifier; the anomaly is therefore attributed to the unmodeled COVID-19 shock (main text Section 3.3).


## S9. Supplementary analyses: transformer window and transform diagnostics

Q-Q normality fit (squared correlation against theoretical quantiles; produced by script 10):


| var | qq_R2 |
|---|---|
| per-capita raw | 0.6081 |
| log1p(per-capita) | 0.8426 |

Numerical ALE products: 1D (Economic Development Level, with 95% CI) and 2D (Development × Urbanization) matrices are provided in Data_Package/04_outputs/ale/. Fold definitions and the 41-city coordinates used for Moran’s I are in Data_Package/04_outputs/folds_definitions.csv and coordinates.csv.


## S10. Data and code availability (response item 14)

See Data_Package/README.md and Reproducibility_Note_R3.md for environment, run order, seeds, folds, hyperparameters, trained models and outputs. Numerical-sensitivity disclosures are listed there (DNN ±0.01 GPU nondeterminism; redundant 2020 indicator; EBM-M2-Full gap; benchmark vs diagnostic residual sets; the leave-one-test-year-out protocol of the two script-10 diagnostics; xgboost-version effect on the third decimal of tree-based rechecks).



## S11. ALE of Energy Efficiency (top-ranked dynamic driver)

Fig. 5B ranks Energy Efficiency first among the dynamic drivers (mean |SHAP| = 0.020). For completeness, its one-dimensional ALE (30 grid points, 95% CI) is provided in Data_Package/04_outputs/ale/ALE_1D_EnergyIntensity.csv.

The trajectory is broadly increasing across most of the support (effect from ≈ −0.030 at low electricity intensity to ≈ +0.058 near the upper middle of the distribution, crossing zero near the mean, with a high-variance reversal only in the extreme tail). This is consistent with the interpretation in the main text (Section 4) that electricity consumption largely proxies logistical and material-throughput activity.

## S12. Map of supplementary items to review comments

| Request | Where addressed |
|---|---|
| #2/#4 blocking tests | S4; main text 3.3 |
| #6 error metrics | S3; main text 3.2.1 |
| #7 paired t-test | main text 3.2.1 (descriptive) |
| #8 SHAP stability | S7; main text 3.4.1 |
| #8 SHAP interpretation | S7, S11 |
| #10 lagged proxies | S5; main text 2.1 |
| #11 spatial dependence | S6; main text 3.3 |
| #14 data/code | S10 + Data_Package |
