﻿# R3 Revision Package — WMR-25-1259

## Quick start
- Back up your project first; the scripts below expect a **project root layout** identical to the
  folder structure in this package (see `Run order`). An easy path: install the pinned environment
  (`05_environment/requirements_pinned.txt`), then copy `01_data/*` and `02_code/*` into a single
  working directory (the scripts read/write from their working directory).
- Full background: `Reproducibility_Note_R3.md`; headline statistics and all changes are listed in
  `Supplementary_Materials.md` (S1–S11) and in the response letter.

## Environment
- conda env `bishe` (python 3.9.25, xgboost 2.1.4, lightgbm 4.6.0, interpret 0.7.5, tensorflow-gpu
  2.10.0 + cudatoolkit 11.2, optuna 4.7.0, shap 0.49.1, statsmodels 0.14.5, PyALE, scikit-learn 1.6.1);
  complete pins in `05_environment/requirements_pinned.txt`.
- All randomness fixed: seed = 42 (numpy/random/tf; Optuna TPESampler seed 42; xgboost random_state 42;
  bootstrap/permutation generators seeded 42).

## Directory layout
```
01_data/         raw data (25_data_raw.zip) + processed CSVs + selected_features.txt + scaler/PCA pickles
02_code/         data_processing.py, feature_engineering.py, model_training.py, visualization.py,
                 visualization_manuscript.py, load_models.py (DNN loader)
  preprocessing/ raw-data crawling/assembly scripts (incl. Baidu-index keyword rules)
  validation/    07_blocking_spatial_validation.py, 08_moran_coordinates.py, 09_shap_fold_stability.py
03_models/       17 model artifacts (12 full pickles; 5 DNN configuration records — see below)
04_outputs/      oof_predictions_xgb_m2_full.csv (benchmark), shap_values.pkl, X_test_for_shap.pkl,
                 optuna_study_xgb_m2_full.pkl, final_benchmark_values.csv, coordinates.csv,
                 folds_definitions.csv, validation/*.csv, ale/*.csv
05_environment/  requirements_pinned.txt, requirements_full_freeze.txt (complete locked listing), best_params_summary.json
06_figures/      all six figures (PNG 300 dpi + PDF vector) — identical to manuscript
```

## Run order
1. `data_processing.py` — assembles raw series from `25_data/` (or `25_data_raw.zip` in this package).
2. `feature_engineering.py` — constructs the ten proxies, log1p per-capita target, RobustScaler,
   PCA (95% variance, fitted on 2011–2019), writes `final_dataset_*.csv`.
3. `model_training.py` — nested CV benchmark (expanding-window outer folds 2018–2022; 100 Optuna
   trials per model per outer fold; early stopping on the inner validation fold); writes models to
   `03_models/` and metrics to `04_outputs/`. The script reads CSV files from its working directory
   and writes to `output/` (mapped to `03_models/` and `04_outputs/` in this package).
4. `visualization_manuscript.py` — regenerates the six manuscript figures (PNG + PDF) into `06_figures/`.
5. `02_code/validation/07_*.py` — blocking tests (LOOCV, no identifiers, lagged-outcome baseline)
   and lag substitution (fixed hyperparameters, no early stopping).
6. `02_code/validation/10_*.py` — diagnostics supplement: missing-value audit, Q-Q fit, transform-window
   sensitivity, 2020-indicator OLS and dummy SHAP, zero-loss PCA, fold/coordinate verification,
   and the diagnostic residual set consumed by step 7.
7. `02_code/validation/08_*.py` — Moran's I on the diagnostic residuals from step 6
   (row-standardized W, 999 permutations).
8. `02_code/validation/09_*.py` — window-level SHAP rank stability (global model applied to each outer test year).
9. `02_code/validation/11_*.py` — isolated transformer-window check: the scaler re-fitted only on the
   outer training years of each fold (strictly leakage-free), with folds/training rows/hyperparameters
   identical to the benchmark (Supplementary S1; `transformers_window_isolated.csv`).
10. `02_code/validation/12_si_shap_fold_figure.py [output_dir]` — regenerates SI_Fig (SHAP rank
    stability across evaluation windows) from `04_outputs/validation/shap_abs_contrib.csv` with the
    substantive feature labels used in Figures 2/5/6 (the SI copy lives outside the package under
    `R3_Supplementary/SI_Figures/`, passed as the output-dir argument).

## DNN model files
The five DNN `.pkl` files are **configuration records** (loadable everywhere); Keras weights were
serialized as RAM-only bytecode that is not portable across processes, so weights are not shipped to
guarantee loadability. Each artifact stores hyperparameter references and its Table S2 benchmark
metrics; retrain with `model_training.py` (same seed and hyperparameters) to reproduce the DNN
rows within ±0.01 R². The other 12 model objects are full pickles.

## Notes on method choices (audited)
- Transformers (scaler/PCA) are fitted on 2011–2019 and applied to the full sample; outer test years
  2018–2019 fall inside that window. Supplementary S1 reports a leave-one-test-year-out refit sensitivity
  check (fold-level R² changes ≤0.024; the re-fitted specifications perform equal or slightly better on
  both tested years); the manuscript states this design explicitly.
- The released datasets contain no missing values, and the defensive fill steps in `feature_engineering.py`
  alter no cell (`04_outputs/validation/imputation_audit.csv`, produced by script 10).
- Validation and diagnostics scripts (07–10) use fixed hyperparameters and **no early stopping**.
- `visualization_manuscript.py` regenerates the six figures from packaged artifacts; Fig. 3B
  reproduces the submitted panel from `04_outputs/validation/convergence_sequences.csv`
  (XGBoost + EBM series, shipped with the package); if that file is removed, the panel falls back
  to the XGBoost trajectory from the packaged Optuna study.
- Moran's I is computed on diagnostic OOF residuals (41 cities; row-standardized 250-km weights;
  999 permutations); residual means in that set may differ from the benchmark OOF file by up to ≈0.014
  (disclosed in the SI, not silently).
- EBM–M2-Full was not estimated (runtime); it is reported as a gap with a stated reason.
- MLR and SVR were estimated only on their M1-Baseline and M2-Full configurations; their M2-PCA cells are also reported as gaps (Table S2), so the PCA comparison spans the model families for which a PCA configuration was estimated.
- DNN results have ±0.01 run-to-run GPU nondeterminism; all other numbers are deterministic.

## Data availability statement (suggested wording)
The processed dataset, preprocessing scripts, model-training code, Optuna search spaces, seeds, fold
definitions, final hyperparameters, trained models, and complete outputs (out-of-fold predictions,
SHAP and ALE artifacts) are provided in the online repository accompanying this revision; all reported
statistics are reproducible with the provided scripts under the pinned environment versions.

