# Reproducibility Note — WMR-25-1259 (R3)

**Scope.** This note describes the locked analysis pipeline behind every statistic reported in the R3
manuscript and its Supplementary Materials.

**Environment.** conda environment `bishe`; Python 3.9.25; xgboost 2.1.4, lightgbm 4.6.0,
interpret 0.7.5 (EBM), tensorflow-gpu 2.10.0 + cudatoolkit 11.2 (NVIDIA RTX 4060), optuna 4.7.0,
shap 0.49.1, statsmodels 0.14.5, PyALE, scikit-learn 1.6.1, pandas 2.3.3, matplotlib 3.7.1;
the full pinned list is in `05_environment/requirements_pinned.txt`.

**Data.** 41 Yangtze River Delta prefecture cities × 12 years (2011–2022); raw series in
`01_data/25_data_raw.zip`; processed files `merged_raw_data.csv`,
`final_dataset_lpc_robust_scaled.csv`, `final_dataset_M2_PCA.csv` (target: log1p per-capita MSW
recycled volume).

**Feature transforms.** Features are RobustScaler-normalized and, for the M2-PCA configuration,
PCA-compressed (95% variance) using statistics from the first nine years (2011–2019) only; the
transformers are then applied to the full sample. Because outer test years 2018–2019 fall inside
this fixed reference window, we additionally report (Supplementary S1) two sensitivity checks:
(i) a bounding check in which the scaler is re-fitted on a leave-one-test-year-out basis and the
model retrained (fold-level R² changes ≤ 0.024; the re-fitted specifications perform equal or
slightly better on both tested years), and (ii) an isolated check (script `11_transform_window_isolated.py`)
in which the scaler is re-fitted only on the outer *training years* of each fold — a strictly
leakage-free refit — with fold definitions, training rows, hyperparameters and stopping rule
identical to the benchmark OOF protocol (fold-level R² changes ≤ 0.0003). The isolated check operates
directly on the released scaled file (refitting a RobustScaler on a row subset is affine-equivalent
to recomputing the within-subset median/IQR of the scaled values), so no raw-value reconstruction is
involved. The released datasets contain no missing values, and the defensive fill steps in
`feature_engineering.py` alter no cell (`04_outputs/validation/imputation_audit.csv`, produced by
script 10).

**Fixed randomness.** numpy/random/tensorflow seed = 42; Optuna TPESampler(seed=42);
model-level random_state = 42; bootstrap/permutation generators seeded with 42.

**Cross-validation.** Expanding-window outer folds testing 2018–2022 (minimum seven training
years); inner expanding window (minimum five years) used only inside `model_training.py`;
early stopping on the inner validation fold (50 rounds); 100 Optuna trials per model per outer
fold (MLR has no tunable hyperparameters; SVR-OHE is tuned by a small grid search).
The validation and diagnostics scripts (07–10) use fixed hyperparameters and **no early
stopping** (n_estimators = 2000).

**Hyperparameters.** Optuna search spaces and final hyperparameters are recorded in
`05_environment/best_params_summary.json`; Fig. 3C importances are computed with Optuna's
mean-decrease-impurity evaluator (n_trees = 64, seed = 42) and are deterministic.

**Run order.** From the Data_Package root:
1. (optional) unzip `01_data/25_data_raw.zip` and rerun preprocessing scripts in `02_code/preprocessing/`;
2. `model_training.py` (needs the project-root layout shown in `README.md`);
3. `visualization_manuscript.py` to regenerate the six figures;
4. validation scripts `02_code/validation/07_*.py` (blocking tests, lag substitution) and `10_*.py`
   (missing-value audit, Q-Q fit, transform-window sensitivity, 2020-indicator OLS and dummy SHAP,
   zero-loss PCA, fold/coordinate verification, diagnostic residual set);
5. `02_code/validation/08_*.py` (Moran's I on the diagnostic residual set produced by script 10);
6. `02_code/validation/09_*.py` (window-level SHAP rank stability);
7. `02_code/validation/11_transform_window_isolated.py` (isolated transformer-window check,
   Supplementary S1; writes `transformers_window_isolated.csv`).
All analysis-output CSVs in `04_outputs/` are produced by one of these steps;
`coordinates.csv` and `folds_definitions.csv` are bundled reference tables (geographic
coordinates; fold design), which script 10 verifies against the code-derived layouts
(`fold_manifest_check.csv`).

**Outputs.** 17 model artifacts (`03_models/`; 12 full pickles; five DNN files are loadable configuration records — Keras weights were serialized as RAM-only bytecode and are not portable across processes, so retraining with the provided script reproduces the DNN rows within ±0.01 R²),
out-of-fold predictions (`04_outputs/oof_predictions_xgb_m2_full.csv`), SHAP and ALE artifacts
(`04_outputs/`), validation tables (`04_outputs/validation/`), coordinates and fold definitions
(`04_outputs/coordinates.csv`, `04_outputs/folds_definitions.csv`).

**Residual sets.** The benchmark out-of-fold file is the authority for all performance statistics
(Table S2, MAPE/SMAPE, Fig. 4). Diagnostic residuals in `04_outputs/validation/oof_resid_with_city.csv`
are recomputed with fixed hyperparameters and no early stopping; yearly means may differ from the
benchmark file by up to ≈0.014 (e.g., 2019: −0.031 vs −0.017), which is disclosed rather than hidden.

**Numerical-sensitivity disclosures.** (a) DNN training has small run-to-run GPU nondeterminism
(±0.01 R²); all other results are deterministic. (b) The 2020-indicator marginal SHAP contribution is
≈ 0 (1.4 × 10⁻⁵; script 10) because the Year identifier already encodes 2020 (redundancy, not a separate
experiment). (c) EBM was not estimated for the M2-Full configuration (runtime); it is reported as a gap,
not a value. (d) Xgboost/library-version differences move the third decimal of tree-based rechecks
(transform-window sensitivity, diagnostic residuals, zero-loss PCA); the CSVs in `04_outputs/validation/`
are the reference-environment values. (e) The transformer-window sensitivity and 2020-indicator
diagnostics in script 10 are computed under a leave-one-test-year-out protocol — training on all rows
other than the tested year, including post-test years — as described in Supplementary S1 and Section 3.3;
they are bounding diagnostics, not strictly leakage-free refits.

**Guarantee.** With the pinned environment and seed, running the released scripts reproduces
Table S2, MAPE/SMAPE, all validation tables, and the six figures (Fig. 3B uses the convergence
series shipped in `04_outputs/validation/convergence_sequences.csv`). The disclosed exceptions are the
GPU-related DNN run-to-run nondeterminism (±0.01 R²), hardware runtime figures, and library-version
effects at the third decimal of tree-based rechecks.