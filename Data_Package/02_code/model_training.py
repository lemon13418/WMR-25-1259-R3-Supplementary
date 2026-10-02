import pandas as pd
import numpy as np
import pickle
import warnings
import random
import os
import matplotlib.pyplot as plt

# --- 新增: 用于记录时间和硬件信息的库 ---
import time
import platform

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# --- 1. 环境与模型导入 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隐藏 TF 警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 修复 TF 兼容性
# 🚨 已经移除了 'TF_USE_LEGACY_KERAS' = '1'，因为你的环境已经降级回稳定版，不需要这个补丁了。

import tensorflow as tf

# --- 兼容较新版本的 TensorFlow (如 2.16+) 或独立安装的 Keras 环境 ---
try:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, Concatenate, Lambda
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import legacy as legacy_optimizer
except ModuleNotFoundError:
    from keras.models import Model, Sequential
    from keras.layers import Dense, Input, Dropout, Embedding, Concatenate, Lambda
    from keras.callbacks import EarlyStopping
    from keras.regularizers import l2
    import keras.optimizers

    if hasattr(keras.optimizers, 'legacy'):
        legacy_optimizer = keras.optimizers.legacy
    else:
        legacy_optimizer = keras.optimizers

# 🚀 彻底抛弃 scikeras，自己手写一个轻量级 Wrapper，免疫所有的 API 版本冲突！
KERAS_AVAILABLE = True


class SimpleKerasWrapper:
    def __init__(self, build_fn, epochs=300, batch_size=128, verbose=0, **kwargs):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.kwargs = kwargs
        self.model = None

    def get_params(self, deep=True):
        return {'epochs': self.epochs, 'batch_size': self.batch_size, 'verbose': self.verbose, **self.kwargs}

    def set_params(self, **params):
        for k, v in params.items():
            if k == 'epochs':
                self.epochs = v
            elif k == 'batch_size':
                self.batch_size = v
            elif k == 'verbose':
                self.verbose = v
            else:
                self.kwargs[k] = v
        return self

    def fit(self, X, y, **fit_params):
        # 每次 fit 都会重新实例化，避免底层的深拷贝报错
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **fit_params)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练！")
        return self.model.predict(X, verbose=self.verbose).flatten()


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow 确认: 找到 {len(gpus)} 个 GPU, 已设置内存动态增长。")
    except RuntimeError as e:
        print(f"⚠️ 警告: 设置 TF 内存增长时出错: {e}")
else:
    print(f"⚠️ 警告: TensorFlow 未检测到 GPU。模型将运行在 CPU。")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# (V5) VIF 诊断
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    VIF_AVAILABLE = True
except ImportError:
    print("警告: 'statsmodels' 未安装。将跳过 MLR-M2 的 VIF 分析。")
    VIF_AVAILABLE = False

# (V5) EBM
try:
    from interpret.glassbox import ExplainableBoostingRegressor

    EBM_AVAILABLE = True
except ImportError:
    print("警告: 'interpret' (EBM) 未安装。将跳过 EBM。")
    EBM_AVAILABLE = False

# (V5) GBDTs
from xgboost import XGBRegressor

XGB_AVAILABLE = True

from lightgbm import LGBMRegressor
import lightgbm as lgbm

LGBM_AVAILABLE = True
CAT_AVAILABLE = False

# Optuna
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
OPTUNA_AVAILABLE = True

# --- 新增导入: SHAP ---
try:
    import shap

    SHAP_AVAILABLE = True
    print("✅ SHAP 已加载。")
except ImportError:
    print("⚠️ 警告: 'shap' 未安装。将跳过 SHAP 值生成。")
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- 2. (V5-Final) 全局配置 (用于正式运行) ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("== V5-FullTune 模式 == (参数已更新为完整搜索空间)")
print("== (已修复): 彻底弃用 scikeras，自建轻量级 Wrapper，解决所有 Keras 版本冲突")
print("== (已加入): 实时增量保存运行时间，并自动跳过已跑完的模型")
print("=" * 70)

OPTUNA_N_TRIALS = 100
N_ESTIMATORS_LARGE = 2000
MIN_TRAIN_YEARS_OUTER = 7
MIN_TRAIN_YEARS_INNER = 5

N_OUTER_SPLITS_FAST_TEST = None
N_INNER_SPLITS_FAST_TEST = None

DNN_EPOCHS_TUNE = 300
DNN_EPOCHS_FINAL = 300
DNN_ES_PATIENCE = 30

GRIDCV_N_JOBS = -1
EARLY_STOPPING_ROUNDS = 50

DYNAMIC_FEATURES = [
    'PerCapitaGDP', 'SecondaryIndustryShare', 'TertiaryIndustryShare',
    'FiscalCapacityPerCapita', 'ConstructionIntensity', 'UrbanizationRate',
    'PatentsPerCapita', 'EnergyIntensity', 'PolicyCount', 'PublicAwarenessIndex'
]
DYNAMIC_FEATURES_PCA = [f'PC{i + 1}' for i in range(7)]


# --- 获取硬件配置信息的函数 ---
def get_hardware_info():
    """获取当前系统的硬件和操作系统信息"""
    os_info = f"{platform.system()} {platform.release()}"
    cpu_info = platform.processor() or "Unknown CPU"

    if PSUTIL_AVAILABLE:
        ram_gb = f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
        cpu_cores = psutil.cpu_count(logical=True)
    else:
        ram_gb = "Unknown (psutil not installed)"
        cpu_cores = os.cpu_count() or "Unknown"

    gpus = tf.config.list_physical_devices('GPU')
    gpu_info = f"{len(gpus)} GPU(s)" if gpus else "CPU Only"

    return f"OS: {os_info} | CPU: {cpu_info} ({cpu_cores} Cores) | RAM: {ram_gb} | GPU: {gpu_info}"


# --- 3. (V5-Final) 核心辅助函数 ---

def calculate_metrics_log_scale(y_true, y_pred):
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        return np.nan, np.nan, np.nan

    r2 = r2_score(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    return r2, rmse, mae


def create_inner_cv(df_train_outer, min_train_years_inner, n_splits_fast_test=None):
    if 'Year' not in df_train_outer.columns:
        raise ValueError("create_inner_cv 需要一个 'Year' 列。")

    df_year_col = df_train_outer['Year']
    unique_years_inner = sorted(df_year_col.unique())
    n_splits_inner = len(unique_years_inner) - min_train_years_inner

    if n_splits_inner <= 0:
        train_years = unique_years_inner[:-1]
        test_year = unique_years_inner[-1]
        train_indices = df_year_col[df_year_col.isin(train_years)].index.to_numpy()
        test_indices = df_year_col[df_year_col == test_year].index.to_numpy()
        return [(train_indices, test_indices)]

    inner_cv_indices = []
    for i in range(n_splits_inner):
        train_years = unique_years_inner[:min_train_years_inner + i]
        test_year = unique_years_inner[min_train_years_inner + i]
        train_indices = df_year_col[df_year_col.isin(train_years)].index.to_numpy()
        test_indices = df_year_col[df_year_col == test_year].index.to_numpy()
        inner_cv_indices.append((train_indices, test_indices))

    if n_splits_fast_test is not None and len(inner_cv_indices) > n_splits_fast_test:
        print(f"     - (FastTest) 内循环: 仅使用最后 {n_splits_fast_test} / {len(inner_cv_indices)} 折。")
        return inner_cv_indices[-n_splits_fast_test:]

    return inner_cv_indices


def create_dnn_model_v2_ohe(input_dim, nh1=64, nh2=32, learning_rate=0.003, dropout_rate=0.2, l2_reg=0.001):
    model = Sequential(name="DNN_OHE_Model")
    model.add(Input(shape=(input_dim,), name="Input_Layer"))
    model.add(Dense(nh1, activation='relu', kernel_regularizer=l2(l2_reg),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                    name=f"Hidden_1_{nh1}"))
    model.add(Dropout(dropout_rate, seed=SEED, name="Dropout_1"))
    model.add(Dense(nh2, activation='relu', kernel_regularizer=l2(l2_reg),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                    name=f"Hidden_2_{nh2}"))
    model.add(Dropout(dropout_rate, seed=SEED, name="Dropout_2"))
    model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED), name="Output_Layer"))
    optimizer = legacy_optimizer.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def create_dnn_model_v4_native(
        native_input_dim,
        city_vocab_size, year_vocab_size, dynamic_dim,
        city_embed_dim=8, year_embed_dim=4,
        nh1=64, nh2=32, learning_rate=0.003, dropout_rate=0.2, l2_reg=0.001
):
    total_in = Input(shape=(native_input_dim,), name='Combined_Input')

    city_in = Lambda(lambda x: tf.cast(x[:, 0:1], dtype='int32'), name='Slice_City')(total_in)
    with tf.device('/CPU:0'):
        city_emb = Embedding(input_dim=city_vocab_size, output_dim=city_embed_dim,
                             embeddings_initializer=tf.keras.initializers.RandomUniform(seed=SEED),
                             name='City_Embedding')(city_in)
    city_emb = tf.keras.layers.Flatten(name='Flatten_City')(city_emb)

    year_in = Lambda(lambda x: tf.cast(x[:, 1:2], dtype='int32'), name='Slice_Year')(total_in)
    with tf.device('/CPU:0'):
        year_emb = Embedding(input_dim=year_vocab_size, output_dim=year_embed_dim,
                             embeddings_initializer=tf.keras.initializers.RandomUniform(seed=SEED),
                             name='Year_Embedding')(year_in)
    year_emb = tf.keras.layers.Flatten(name='Flatten_Year')(year_emb)

    if dynamic_dim > 0:
        dynamic_in = Lambda(lambda x: x[:, 2:], name='Slice_Dynamic')(total_in)
        all_features = [city_emb, year_emb, dynamic_in]
    else:
        all_features = [city_emb, year_emb]

    x = Concatenate(name='Concatenated_Features')(all_features)
    x = Dense(nh1, activation='relu', kernel_regularizer=l2(l2_reg),
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
              name=f"Hidden_1_{nh1}")(x)
    x = Dropout(dropout_rate, seed=SEED, name="Dropout_1")(x)
    x = Dense(nh2, activation='relu', kernel_regularizer=l2(l2_reg),
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
              name=f"Hidden_2_{nh2}")(x)
    x = Dropout(dropout_rate, seed=SEED, name="Dropout_2")(x)
    output = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED), name="Output_Layer")(x)

    model = Model(inputs=total_in, outputs=output, name=f"DNN_Native_Emb_M{1 if dynamic_dim == 0 else 2}")
    optimizer = legacy_optimizer.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def _prep_dnn_native_input(df, dynamic_features):
    city_codes = df['City'].cat.codes.values.reshape(-1, 1)
    year_codes = df['Year'].cat.codes.values.reshape(-1, 1)
    if len(dynamic_features) > 0:
        dynamic_data = df[dynamic_features].values
        return np.hstack([city_codes, year_codes, dynamic_data])
    else:
        return np.hstack([city_codes, year_codes])


def get_model_instance(
        model_name, params, data_format_type, fixed_ohe_input_dim=None,
        city_vocab_size=None, year_vocab_size=None, dynamic_dim=None
):
    params_copy = params.copy()

    if data_format_type == 'ohe':
        if model_name == 'MLR':
            return LinearRegression()
        if model_name == 'SVR':
            return SVR(**params_copy)
        if model_name == 'DNN':
            # 🚀 使用自建 Wrapper 替换 scikeras 的 KerasRegressor
            model_tuned_params = {k.replace('model__', ''): v for k, v in params.items() if k.startswith('model__')}
            batch_size = params.get('batch_size', 128)
            return SimpleKerasWrapper(
                build_fn=create_dnn_model_v2_ohe,
                input_dim=fixed_ohe_input_dim,
                epochs=DNN_EPOCHS_TUNE,
                batch_size=batch_size,
                verbose=0,
                **model_tuned_params
            )
        raise ValueError(f"未知的 OHE model_name: {model_name}")

    elif data_format_type == 'native':
        if model_name == 'EBM':
            params_copy['random_state'] = SEED
            params_copy['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
            params_copy['feature_types'] = ['nominal', 'nominal'] + ['continuous'] * dynamic_dim
            params_copy['n_jobs'] = -1
            params_copy['validation_size'] = 0.2
            return ExplainableBoostingRegressor(**params_copy)

        if model_name == 'XGBoost':
            params_copy['random_state'] = SEED
            params_copy['n_jobs'] = -1
            params_copy['n_estimators'] = N_ESTIMATORS_LARGE
            params_copy['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
            params_copy['device'] = 'cuda'
            params_copy['verbosity'] = 0
            params_copy['enable_categorical'] = True
            return XGBRegressor(**params_copy)

        if model_name == 'LGBM':
            params_copy['random_state'] = SEED
            params_copy['n_jobs'] = -1
            params_copy['n_estimators'] = N_ESTIMATORS_LARGE
            params_copy['verbose'] = -1
            params_copy['device'] = 'cpu'
            return LGBMRegressor(**params_copy)

        if model_name == 'DNN':
            # 🚀 使用自建 Wrapper 替换 scikeras 的 KerasRegressor
            model_tuned_params = {k.replace('model__', ''): v for k, v in params.items() if k.startswith('model__')}
            batch_size = params.get('batch_size', 128)
            native_input_dim = 2 + dynamic_dim
            return SimpleKerasWrapper(
                build_fn=create_dnn_model_v4_native,
                native_input_dim=native_input_dim,
                city_vocab_size=city_vocab_size,
                year_vocab_size=year_vocab_size,
                dynamic_dim=dynamic_dim,
                epochs=DNN_EPOCHS_TUNE,
                batch_size=batch_size,
                verbose=0,
                **model_tuned_params
            )
        raise ValueError(f"未知的 Native model_name: {model_name}")
    raise ValueError(f"未知的 data_format_type: {data_format_type}")


def run_tuning(
        X_train_outer, y_train_outer, model_name, inner_cv_indices, data_format_type,
        fixed_ohe_input_dim=None, city_vocab_size=None, year_vocab_size=None,
        dynamic_dim=None, native_cat_features=None
):
    if model_name == 'MLR':
        return {}, None

    if model_name == 'SVR' and data_format_type == 'ohe':
        estimator = SVR()
        param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 0.1], 'epsilon': [0.1]}
        current_n_jobs = GRIDCV_N_JOBS
        try:
            X_train_cleaned = X_train_outer.dropna()
            y_train_cleaned = y_train_outer.loc[X_train_cleaned.index]
            X_train_cleaned_input = X_train_cleaned
            cleaned_inner_cv = []
            for train_idx, test_idx in inner_cv_indices:
                valid_train_idx = X_train_cleaned.index.intersection(train_idx)
                valid_test_idx = X_train_cleaned.index.intersection(test_idx)
                loc_train_idx = X_train_cleaned.index.get_indexer(valid_train_idx)
                loc_test_idx = X_train_cleaned.index.get_indexer(valid_test_idx)
                if len(loc_train_idx) > 0 and len(loc_test_idx) > 0:
                    cleaned_inner_cv.append((loc_train_idx, loc_test_idx))

            if not cleaned_inner_cv:
                print(f"     - (内循环) {model_name} GridCV 警告: 没有有效的内循环折, 使用默认参数。")
                return {}, None

            grid_search = GridSearchCV(
                estimator=estimator, param_grid=param_grid, cv=cleaned_inner_cv,
                scoring='neg_root_mean_squared_error', n_jobs=current_n_jobs, verbose=0
            )
            grid_search.fit(X_train_cleaned_input, y_train_cleaned)
            return grid_search.best_params_, None
        except Exception as e:
            print(f"GridSearchCV 调优 {model_name} 失败: {e}")
            return {}, None

    else:
        # 🚀 修复点: 传入当前 optuna 推荐的 params, 并在循环内部直接新建实例
        def _run_optuna_cv(params_for_cv, fit_params_cv):
            rmses = []
            for train_idx, val_idx in inner_cv_indices:
                X_train_fold, X_val_fold = X_train_outer.loc[train_idx], X_train_outer.loc[val_idx]
                y_train_fold, y_val_fold = y_train_outer.loc[train_idx], y_train_outer.loc[val_idx]

                try:
                    # 使用当前参数新建模型实例
                    model_clone = get_model_instance(
                        model_name, params_for_cv, data_format_type,
                        fixed_ohe_input_dim=fixed_ohe_input_dim,
                        city_vocab_size=city_vocab_size,
                        year_vocab_size=year_vocab_size,
                        dynamic_dim=dynamic_dim
                    )

                    if model_name == 'DNN' and data_format_type == 'native':
                        X_train_fold_input = _prep_dnn_native_input(X_train_fold, native_cat_features['dynamic'])
                        X_val_fold_input = _prep_dnn_native_input(X_val_fold, native_cat_features['dynamic'])
                    elif model_name == 'DNN' and data_format_type == 'ohe':
                        X_train_fold_input = X_train_fold.dropna()
                        y_train_fold = y_train_fold.loc[X_train_fold_input.index]
                        X_val_fold_input = X_val_fold.dropna()
                        y_val_fold = y_val_fold.loc[X_val_fold_input.index]
                    else:
                        X_train_fold_input = X_train_fold
                        X_val_fold_input = X_val_fold

                    current_fit_params = {}
                    if 'eval_set' in fit_params_cv:
                        if model_name == 'DNN' and data_format_type == 'ohe':
                            current_fit_params['validation_data'] = (X_val_fold_input, y_val_fold)
                            current_fit_params['callbacks'] = [
                                EarlyStopping(monitor='val_loss', patience=DNN_ES_PATIENCE, restore_best_weights=True)
                            ]
                        elif model_name == 'DNN' and data_format_type == 'native':
                            eval_set_input = _prep_dnn_native_input(X_val_fold, native_cat_features['dynamic'])
                            current_fit_params['validation_data'] = (eval_set_input, y_val_fold)
                            current_fit_params['callbacks'] = [
                                EarlyStopping(monitor='val_loss', patience=DNN_ES_PATIENCE, restore_best_weights=True)
                            ]
                        else:
                            current_fit_params['eval_set'] = [(X_val_fold_input, y_val_fold)]
                            if 'callbacks' in fit_params_cv:
                                current_fit_params['callbacks'] = fit_params_cv['callbacks']
                            if 'verbose' in fit_params_cv:
                                current_fit_params['verbose'] = fit_params_cv['verbose']

                    elif model_name == 'EBM':
                        pass

                    model_clone.fit(X_train_fold_input, y_train_fold, **current_fit_params)
                    y_pred_val_log = model_clone.predict(X_val_fold_input)
                    _, rmse, _ = calculate_metrics_log_scale(y_val_fold, y_pred_val_log)
                    rmses.append(rmse)
                except Exception as e:
                    print(f"     - (内循环) Optuna 折失败 {model_name}: {e}")
                    rmses.append(np.inf)
            return np.mean(rmses)

        def objective_dnn_ohe(trial):
            params = {
                'model__nh1': trial.suggest_categorical('model__nh1', [64, 128, 256]),
                'model__nh2': trial.suggest_categorical('model__nh2', [32, 64, 128]),
                'model__learning_rate': trial.suggest_float('model__learning_rate', 1e-4, 1e-2, log=True),
                'model__dropout_rate': trial.suggest_float('model__dropout_rate', 0.1, 0.4),
                'model__l2_reg': trial.suggest_float('model__l2_reg', 1e-5, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            }
            fit_params_cv = {'eval_set': True}
            return _run_optuna_cv(params, fit_params_cv)

        def objective_dnn_native(trial):
            params = {
                'model__nh1': trial.suggest_categorical('model__nh1', [64, 128, 256]),
                'model__nh2': trial.suggest_categorical('model__nh2', [32, 64, 128]),
                'model__learning_rate': trial.suggest_float('model__learning_rate', 1e-4, 1e-2, log=True),
                'model__dropout_rate': trial.suggest_float('model__dropout_rate', 0.1, 0.4),
                'model__l2_reg': trial.suggest_float('model__l2_reg', 1e-5, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            }
            fit_params_cv = {'eval_set': True}
            return _run_optuna_cv(params, fit_params_cv)

        def objective_xgb(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 50.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            }
            fit_params_cv = {'eval_set': True, 'verbose': 0}
            return _run_optuna_cv(params, fit_params_cv)

        def objective_lgbm(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 31),
                'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 50.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 10.0, 30.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            }
            fit_params_cv = {
                'eval_set': True,
                'callbacks': [lgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                'categorical_feature': native_cat_features['all']
            }
            return _run_optuna_cv(params, fit_params_cv)

        def objective_ebm(trial):
            params = {
                'interactions': 0,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_bins': trial.suggest_categorical('max_bins', [256, 512]),
                'outer_bags': trial.suggest_int('outer_bags', 8, 16),
                'inner_bags': trial.suggest_int('inner_bags', 0, 8),
            }
            fit_params_cv = {}
            return _run_optuna_cv(params, fit_params_cv)

        try:
            if model_name == 'DNN' and data_format_type == 'ohe':
                objective_func = objective_dnn_ohe
            elif model_name == 'DNN' and data_format_type == 'native':
                objective_func = objective_dnn_native
            elif model_name == 'EBM':
                objective_func = objective_ebm
            elif model_name == 'XGBoost':
                objective_func = objective_xgb
            elif model_name == 'LGBM':
                objective_func = objective_lgbm
            else:
                return {}, None

            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
            study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
            return study.best_params, study

        except Exception as e:
            print(f"Optuna 调优 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return {}, None


# --- 4. (V5-Final) 主函数 ---
def main():
    print("=" * 70)
    print(f"V5-FullTune 模型验证 (M1 vs M2_Full vs M2_PCA)")
    print(f"[模型集] MLR, SVR, DNN, EBM, XGBoost, LGBM (已移除 CatBoost)")
    print(f"[新增] 自动跳过已跑完的模型，修复EBM卡死与DNN克隆报错，实时增量保存输出")
    print("=" * 70)

    # --- 新增: 集中管理所有输出的根目录 ---
    BASE_OUTPUT_DIR = "output"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 存放各个算法训练好的模型文件
    output_dir = os.path.join(BASE_OUTPUT_DIR, "models_v5_final")
    os.makedirs(output_dir, exist_ok=True)

    # 存放所有的评估指标表格、预测结果和分析数据
    viz_output_dir = os.path.join(BASE_OUTPUT_DIR, "visualization_data")
    os.makedirs(viz_output_dir, exist_ok=True)

    # 获取并打印全局硬件配置信息
    hardware_info_str = get_hardware_info()
    print(f"当前硬件配置: {hardware_info_str}")
    print(f"总输出根目录已就绪: ./{BASE_OUTPUT_DIR}/\n")

    # --- 🔥 真正的断点续跑：直接从本地 CSV 读取进度 ---
    times_save_path = os.path.join(viz_output_dir, "model_execution_times_V5.csv")
    execution_times_records = []

    # 检测并读取已经增量保存的 CSV
    if os.path.exists(times_save_path):
        try:
            df_existing = pd.read_csv(times_save_path)
            if not df_existing.empty:
                execution_times_records = df_existing.to_dict('records')
                print(f"🔄 [断点恢复] 成功读取本地记录，已恢复 {len(execution_times_records)} 个模型的进度！")
        except Exception as e:
            print(f"⚠️ [断点恢复警告] 读取现有的 CSV 文件失败，将从头开始计算: {e}")

    # 获取包含所有已完成模型的列表，用于跳过
    COMPLETED_MODELS = [record['Model'] for record in execution_times_records]
    if COMPLETED_MODELS:
        print(f"⏭️  本次运行将自动跳过以下已完成的模型:")
        for m in COMPLETED_MODELS:
            print(f"    - {m}")
    else:
        print("▶️ 未检测到已完成的模型记录，将从头开始运行。")
    # ------------------------------------------------------------------

    # --- 步骤 1: (V5) 加载两个数据集 ---
    try:
        df_full_robust = pd.read_csv('final_dataset_lpc_robust_scaled.csv')
        df_full_pca = pd.read_csv('final_dataset_M2_PCA.csv')
    except FileNotFoundError as e:
        print(f"错误: 关键数据文件未找到。{e}")
        print("请确保已成功运行 feature_engineering.py")
        return

    y_full = df_full_robust['Target']
    print("V5 数据准备: 成功加载 'robust' 和 'pca' 数据集。")

    # --- 步骤 2: (V5) 创建三个独立的数据集 ---
    df_ohe_source = df_full_robust.copy()
    df_ohe_source['Year_str'] = df_ohe_source['Year'].astype(str)
    X_ohe_features = pd.get_dummies(df_ohe_source, columns=['City', 'Year_str'], drop_first=True, dtype=np.float32)
    features_ohe_to_keep = DYNAMIC_FEATURES + [col for col in X_ohe_features.columns if
                                               'City_' in col or 'Year_str_' in col]
    X_full_df_ohe = X_ohe_features[features_ohe_to_keep]
    X_full_df_ohe['Year_CV'] = df_full_robust['Year']
    fixed_ohe_input_dim = len(features_ohe_to_keep)
    print(f"V5 数据准备: OHE (数据集 A) 特征数: {fixed_ohe_input_dim}")

    df_native_robust = df_full_robust.copy()
    df_native_robust['City'] = df_native_robust['City'].astype('category')
    df_native_robust['Year'] = df_native_robust['Year'].astype('category')
    features_native_robust_to_keep = DYNAMIC_FEATURES + ['City', 'Year']
    X_full_df_native_robust = df_native_robust[features_native_robust_to_keep]
    X_full_df_native_robust['Year_CV'] = df_full_robust['Year']

    df_native_pca = df_full_pca.copy()
    df_native_pca['City'] = df_native_pca['City'].astype('category')
    df_native_pca['Year'] = df_native_pca['Year'].astype('category')
    features_native_pca_to_keep = DYNAMIC_FEATURES_PCA + ['City', 'Year']
    X_full_df_native_pca = df_native_pca[features_native_pca_to_keep]
    X_full_df_native_pca['Year_CV'] = df_full_pca['Year']

    city_vocab_size = len(df_native_robust['City'].cat.categories) + 1
    year_vocab_size = len(df_native_robust['Year'].cat.categories) + 1
    print(f"V5 Native DNN: City 词汇表 {city_vocab_size}, Year 词汇表 {year_vocab_size}")

    # --- 步骤 3: (V5) 定义特征集 ---
    features_ohe_m1 = [col for col in X_full_df_ohe.columns if 'City_' in col or 'Year_str_' in col]
    features_ohe_m2_full = features_ohe_m1 + DYNAMIC_FEATURES
    FEATURE_SETS_OHE = {
        'M1_Baseline-OHE': features_ohe_m1,
        'M2_Full_L1-OHE': features_ohe_m2_full,
    }

    features_native_m1 = ['City', 'Year']
    features_native_m2_full = features_native_m1 + DYNAMIC_FEATURES
    features_native_m2_pca = features_native_m1 + DYNAMIC_FEATURES_PCA

    FEATURE_SETS_NATIVE = {
        'M1_Baseline-Native': (features_native_m1, 'robust', 0),
        'M2_Full_L1-Native': (features_native_m2_full, 'robust', len(DYNAMIC_FEATURES)),
        'M2_PCA_L1-Native': (features_native_m2_pca, 'pca', len(DYNAMIC_FEATURES_PCA)),
    }

    native_cat_meta = {
        'M1_Baseline-Native': {'all': ['City', 'Year'], 'dynamic': []},
        'M2_Full_L1-Native': {'all': ['City', 'Year'], 'dynamic': DYNAMIC_FEATURES},
        'M2_PCA_L1-Native': {'all': ['City', 'Year'], 'dynamic': DYNAMIC_FEATURES_PCA},
    }

    # --- 步骤 4: 定义外循环 (评估) CV ---
    df_year_col = df_full_robust['Year']
    unique_years_outer = sorted(df_year_col.unique())
    N_SPLITS_OUTER = len(unique_years_outer) - MIN_TRAIN_YEARS_OUTER
    outer_cv_indices_all = []
    for i in range(N_SPLITS_OUTER):
        train_years = unique_years_outer[:MIN_TRAIN_YEARS_OUTER + i]
        test_year = unique_years_outer[MIN_TRAIN_YEARS_OUTER + i]
        train_indices = df_year_col[df_year_col.isin(train_years)].index.to_numpy()
        test_indices = df_year_col[df_year_col == test_year].index.to_numpy()
        outer_cv_indices_all.append((train_indices, test_indices))

    if N_OUTER_SPLITS_FAST_TEST is not None:
        outer_cv_indices = outer_cv_indices_all[-N_OUTER_SPLITS_FAST_TEST:]
    else:
        outer_cv_indices = outer_cv_indices_all

    # --- 步骤 5: 运行嵌套 CV ---
    print("\n" + "=" * 70)
    print(f"V5-FullTune 步骤 5: 正在运行 {len(outer_cv_indices)}-Fold 嵌套交叉验证...")
    print("=" * 70)

    MODELS_TO_TEST_OHE = ['MLR', 'SVR', 'DNN']
    MODELS_TO_TEST_NATIVE_FAST = ['XGBoost', 'LGBM']
    MODELS_TO_TEST_NATIVE_SLOW = ['EBM']
    MODELS_TO_TEST_NATIVE_DNN = ['DNN']

    all_cv_metrics = []
    all_oof_predictions = {}
    all_best_params = {}

    # --- 运行 1: Native 组 (XGBoost, LGBM) ---
    print("\n" + "=" * 70)
    print("V5-FullTune 实验 1: 运行 Native 组 (XGBoost, LGBM)")
    print("=" * 70)

    for model_name in MODELS_TO_TEST_NATIVE_FAST:
        if (model_name == 'XGBoost' and not XGB_AVAILABLE) or \
                (model_name == 'LGBM' and not LGBM_AVAILABLE):
            continue

        for arch_name, (current_feature_set, df_key, current_dynamic_dim) in FEATURE_SETS_NATIVE.items():
            full_model_name = f"{model_name}-{arch_name}"

            # --- 跳转逻辑 ---
            if full_model_name in COMPLETED_MODELS:
                print(f"  - ⏩ 已跳过 (本地记录显示已完成): {full_model_name}")
                continue

            print(f"  - 正在评估: {full_model_name}")
            start_time = time.time()

            oof_preds = np.full(len(y_full), np.nan)
            oof_true = np.full(len(y_full), np.nan)
            oof_years = np.full(len(y_full), np.nan)

            current_cat_features = native_cat_meta[arch_name]
            fold_best_params = {}
            best_study_object = None

            for fold, (train_idx_outer, test_idx_outer) in enumerate(outer_cv_indices):
                if df_key == 'robust':
                    X_full_df_native_source = X_full_df_native_robust
                elif df_key == 'pca':
                    X_full_df_native_source = X_full_df_native_pca

                X_train_outer_df = X_full_df_native_source.iloc[train_idx_outer]
                y_train_outer_series = y_full.iloc[train_idx_outer]
                X_test_outer_df = X_full_df_native_source.iloc[test_idx_outer]
                y_test_outer_series = y_full.iloc[test_idx_outer]

                df_cv_split = X_train_outer_df[['Year_CV']].rename(columns={'Year_CV': 'Year'})
                inner_cv_folds = create_inner_cv(df_cv_split, MIN_TRAIN_YEARS_INNER, N_INNER_SPLITS_FAST_TEST)

                best_params, study = run_tuning(
                    X_train_outer_df[current_feature_set], y_train_outer_series, model_name,
                    inner_cv_folds, data_format_type='native', city_vocab_size=city_vocab_size,
                    year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim,
                    native_cat_features=current_cat_features
                )
                fold_best_params = best_params

                if full_model_name == 'XGBoost-M2_Full_L1-Native' and study is not None:
                    best_study_object = study

                model_fold = get_model_instance(
                    model_name, best_params, 'native', city_vocab_size=city_vocab_size,
                    year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim
                )

                X_train_fit = X_train_outer_df[current_feature_set]
                y_train_fit = y_train_outer_series
                fit_params = {}
                val_train_idx, val_test_idx = inner_cv_folds[-1]
                X_val_fit = X_train_outer_df.loc[val_test_idx][current_feature_set]
                y_val_fit = y_train_outer_series.loc[X_val_fit.index]

                X_train_fit_input = X_train_fit
                X_val_fit_input = X_val_fit
                if model_name == 'XGBoost':
                    fit_params['eval_set'] = [(X_val_fit_input, y_val_fit)]
                    fit_params['verbose'] = 0
                elif model_name == 'LGBM':
                    fit_params['eval_set'] = [(X_val_fit_input, y_val_fit)]
                    fit_params['callbacks'] = [lgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
                    fit_params['categorical_feature'] = current_cat_features['all']

                model_fold.fit(X_train_fit_input, y_train_fit, **fit_params)

                X_test_predict = X_test_outer_df[current_feature_set]
                y_test_predict_true = y_test_outer_series.loc[X_test_predict.index]
                X_test_predict_input = X_test_predict

                if len(X_test_predict) > 0:
                    preds_test = model_fold.predict(X_test_predict_input)
                    oof_preds[X_test_predict.index] = preds_test
                    oof_true[X_test_predict.index] = y_test_predict_true.values
                    oof_years[X_test_predict.index] = X_test_outer_df['Year_CV'].values

            r2_log, rmse_log, mae_log = calculate_metrics_log_scale(oof_true, oof_preds)
            all_cv_metrics.append({
                'Model': full_model_name, 'R2 (log-scale)': r2_log,
                'RMSE (log-scale)': rmse_log, 'MAE (log-scale)': mae_log
            })
            all_oof_predictions[full_model_name] = oof_preds
            all_best_params[full_model_name] = fold_best_params

            if full_model_name == 'XGBoost-M2_Full_L1-Native':
                df_oof_viz = pd.DataFrame(
                    {'True_Values': oof_true, 'Predicted_Values': oof_preds, 'Test_Year': oof_years}).dropna()
                df_oof_viz.to_csv(os.path.join(viz_output_dir, "oof_predictions_xgb_m2_full.csv"), index=False)
                if best_study_object:
                    with open(os.path.join(viz_output_dir, "optuna_study_xgb_m2_full.pkl"), 'wb') as f:
                        pickle.dump(best_study_object, f)

            try:
                if df_key == 'robust':
                    X_full_df_native_source = X_full_df_native_robust
                elif df_key == 'pca':
                    X_full_df_native_source = X_full_df_native_pca

                X_train_final = X_full_df_native_source[current_feature_set]
                y_train_final = y_full.loc[X_train_final.index]
                final_model = get_model_instance(
                    model_name, best_params, 'native', city_vocab_size=city_vocab_size,
                    year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim
                )
                final_fit_params = {}
                X_train_final_input = X_train_final
                if 'n_estimators' in final_model.get_params(): final_model.set_params(n_estimators=N_ESTIMATORS_LARGE)
                if 'iterations' in final_model.get_params(): final_model.set_params(iterations=N_ESTIMATORS_LARGE)
                if 'early_stopping_rounds' in final_model.get_params(): final_model.set_params(
                    early_stopping_rounds=None)
                if model_name == 'LGBM': final_fit_params['categorical_feature'] = current_cat_features['all']

                final_model.fit(X_train_final_input, y_train_final, **final_fit_params)

                final_model_pipeline = {
                    'model': final_model, 'model_name': model_name, 'architecture': arch_name,
                    'features': current_feature_set, 'best_params_from_cv': best_params,
                    'data_format_type': 'native',
                    'native_info': {'city_vocab_size': city_vocab_size, 'year_vocab_size': year_vocab_size,
                                    'dynamic_dim': current_dynamic_dim, 'cat_features': current_cat_features}
                }
                with open(os.path.join(output_dir, f"{full_model_name}.pkl"), 'wb') as f:
                    pickle.dump(final_model_pipeline, f)

                if full_model_name == 'XGBoost-M2_Full_L1-Native' and SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(final_model)
                        shap_values = explainer.shap_values(X_train_final)
                        with open(os.path.join(viz_output_dir, "shap_values.pkl"), 'wb') as f:
                            pickle.dump(shap_values, f)
                        X_train_final.to_pickle(os.path.join(viz_output_dir, "X_test_for_shap.pkl"))
                    except Exception as e:
                        print(f"    - [错误] SHAP 值计算失败: {e}")

            except Exception as e:
                print(f"保存 {full_model_name} 失败: {e} (记录耗时后将继续运行)")

            # --- 计算并记录结束耗时 ---
            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            execution_times_records.append({
                'Model': full_model_name,
                'Hardware_Config': hardware_info_str,
                'Execution_Time_Seconds': round(elapsed_time, 2),
                'Execution_Time_Formatted': formatted_time
            })
            print(f"    -> [耗时统计] {full_model_name} 完成，耗时: {formatted_time}")

            # --- 新增: 实时增量保存运行时间 ---
            df_times_partial = pd.DataFrame(execution_times_records)
            df_times_partial.to_csv(os.path.join(viz_output_dir, "model_execution_times_V5.csv"), index=False,
                                    encoding='utf-8-sig')

    # --- 运行 2: Native 组 (EBM) ---
    print("\n" + "=" * 70)
    print("V5-FullTune 实验 2: 运行 Native 组 (EBM)")
    print("=" * 70)
    for model_name in MODELS_TO_TEST_NATIVE_SLOW:
        if (model_name == 'EBM' and not EBM_AVAILABLE): continue

        for arch_name, (current_feature_set, df_key, current_dynamic_dim) in FEATURE_SETS_NATIVE.items():
            full_model_name = f"{model_name}-{arch_name}"
            if model_name == 'EBM' and 'M2_Full_L1-Native' in arch_name: continue

            if full_model_name in COMPLETED_MODELS:
                print(f"  - ⏩ 已跳过 (本地记录显示已完成): {full_model_name}")
                continue

            print(f"  - 正在评估: {full_model_name}")
            start_time = time.time()

            oof_preds = np.full(len(y_full), np.nan)
            oof_true = np.full(len(y_full), np.nan)
            current_cat_features = native_cat_meta[arch_name]
            fold_best_params = {}

            for fold, (train_idx_outer, test_idx_outer) in enumerate(outer_cv_indices):
                if df_key == 'robust':
                    X_full_df_native_source = X_full_df_native_robust
                elif df_key == 'pca':
                    X_full_df_native_source = X_full_df_native_pca

                X_train_outer_df = X_full_df_native_source.iloc[train_idx_outer]
                y_train_outer_series = y_full.iloc[train_idx_outer]
                X_test_outer_df = X_full_df_native_source.iloc[test_idx_outer]
                y_test_outer_series = y_full.iloc[test_idx_outer]

                df_cv_split = X_train_outer_df[['Year_CV']].rename(columns={'Year_CV': 'Year'})
                inner_cv_folds = create_inner_cv(df_cv_split, MIN_TRAIN_YEARS_INNER, N_INNER_SPLITS_FAST_TEST)

                best_params, _ = run_tuning(
                    X_train_outer_df[current_feature_set], y_train_outer_series, model_name,
                    inner_cv_folds, data_format_type='native', city_vocab_size=city_vocab_size,
                    year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim,
                    native_cat_features=current_cat_features
                )
                fold_best_params = best_params

                model_fold = get_model_instance(model_name, best_params, 'native',
                                                city_vocab_size=city_vocab_size, year_vocab_size=year_vocab_size,
                                                dynamic_dim=current_dynamic_dim)
                model_fold.fit(X_train_outer_df[current_feature_set], y_train_outer_series)

                X_test_predict = X_test_outer_df[current_feature_set]
                if len(X_test_predict) > 0:
                    preds_test = model_fold.predict(X_test_predict)
                    oof_preds[X_test_predict.index] = preds_test
                    oof_true[X_test_predict.index] = y_test_outer_series.loc[X_test_predict.index].values

            all_cv_metrics.append({
                'Model': full_model_name, 'R2 (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[0],
                'RMSE (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[1],
                'MAE (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[2]
            })
            all_oof_predictions[full_model_name] = oof_preds
            all_best_params[full_model_name] = fold_best_params

            try:
                if df_key == 'robust':
                    X_full_df_native_source = X_full_df_native_robust
                elif df_key == 'pca':
                    X_full_df_native_source = X_full_df_native_pca
                X_train_final = X_full_df_native_source[current_feature_set]

                final_model = get_model_instance(model_name, best_params, 'native',
                                                 city_vocab_size=city_vocab_size, year_vocab_size=year_vocab_size,
                                                 dynamic_dim=current_dynamic_dim)

                # --- 🔥 已删除 EBM 的 validation_size=0 覆盖设定，让其保持早停机制 ---

                final_model.fit(X_train_final, y_full.loc[X_train_final.index])

                final_model_pipeline = {
                    'model': final_model, 'model_name': model_name, 'architecture': arch_name,
                    'features': current_feature_set, 'best_params_from_cv': best_params, 'data_format_type': 'native'
                }
                with open(os.path.join(output_dir, f"{full_model_name}.pkl"), 'wb') as f:
                    pickle.dump(final_model_pipeline, f)
            except Exception as e:
                print(f"保存 {full_model_name} 失败: {e} (记录耗时后将继续运行)")

            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            execution_times_records.append({
                'Model': full_model_name, 'Hardware_Config': hardware_info_str,
                'Execution_Time_Seconds': round(elapsed_time, 2), 'Execution_Time_Formatted': formatted_time
            })
            print(f"    -> [耗时统计] {full_model_name} 完成，耗时: {formatted_time}")

            # --- 新增: 实时增量保存 ---
            df_times_partial = pd.DataFrame(execution_times_records)
            df_times_partial.to_csv(os.path.join(viz_output_dir, "model_execution_times_V5.csv"), index=False,
                                    encoding='utf-8-sig')

    # --- 运行 3: OHE 组 (MLR, SVR, DNN-OHE) ---
    print("\n" + "=" * 70)
    print("V5-FullTune 实验 3: 运行 OHE 组 (MLR, SVR, DNN-OHE)")
    print("=" * 70)
    for model_name in MODELS_TO_TEST_OHE:
        if (model_name == 'DNN' and not KERAS_AVAILABLE): continue

        for arch_name, feature_names in FEATURE_SETS_OHE.items():
            full_model_name = f"{model_name}-{arch_name}"

            if full_model_name in COMPLETED_MODELS:
                print(f"  - ⏩ 已跳过 (本地记录显示已完成): {full_model_name}")
                continue

            print(f"  - 正在评估: {full_model_name}")
            start_time = time.time()

            oof_preds = np.full(len(y_full), np.nan)
            oof_true = np.full(len(y_full), np.nan)
            current_feature_set = [f for f in feature_names if f in X_full_df_ohe.columns]
            current_input_dim = len(current_feature_set)
            fold_best_params = {}

            for fold, (train_idx_outer, test_idx_outer) in enumerate(outer_cv_indices):
                X_train_outer_df = X_full_df_ohe.iloc[train_idx_outer]
                y_train_outer_series = y_full.iloc[train_idx_outer]
                X_test_outer_df = X_full_df_ohe.iloc[test_idx_outer]
                y_test_outer_series = y_full.iloc[test_idx_outer]

                inner_cv_folds = create_inner_cv(X_train_outer_df.rename(columns={'Year_CV': 'Year'}),
                                                 MIN_TRAIN_YEARS_INNER, N_INNER_SPLITS_FAST_TEST)
                best_params, _ = run_tuning(X_train_outer_df[current_feature_set], y_train_outer_series, model_name,
                                            inner_cv_folds, data_format_type='ohe',
                                            fixed_ohe_input_dim=current_input_dim)
                fold_best_params = best_params

                model_fold = get_model_instance(model_name, best_params, 'ohe', fixed_ohe_input_dim=current_input_dim)

                X_train_fit = X_train_outer_df[current_feature_set].dropna()
                y_train_fit = y_train_outer_series.loc[X_train_fit.index]
                fit_params = {}
                if model_name == 'DNN':
                    val_train_idx, val_test_idx = inner_cv_folds[-1]
                    X_val_fit = X_train_outer_df.loc[val_test_idx][current_feature_set].dropna()
                    fit_params['validation_data'] = (X_val_fit, y_train_outer_series.loc[X_val_fit.index])
                    fit_params['callbacks'] = [
                        EarlyStopping(monitor='val_loss', patience=DNN_ES_PATIENCE, restore_best_weights=True)]

                model_fold.fit(X_train_fit, y_train_fit, **fit_params)

                X_test_predict = X_test_outer_df[current_feature_set]
                if len(X_test_predict) > 0:
                    preds_test = model_fold.predict(X_test_predict)
                    oof_preds[X_test_predict.index] = preds_test
                    oof_true[X_test_predict.index] = y_test_outer_series.loc[X_test_predict.index].values

            all_cv_metrics.append({
                'Model': full_model_name, 'R2 (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[0],
                'RMSE (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[1],
                'MAE (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[2]
            })
            all_oof_predictions[full_model_name] = oof_preds
            all_best_params[full_model_name] = fold_best_params

            try:
                X_train_final = X_full_df_ohe[current_feature_set].dropna()
                final_model = get_model_instance(model_name, best_params, 'ohe', fixed_ohe_input_dim=current_input_dim)
                final_fit_params = {}
                if model_name == 'DNN': final_model.set_params(epochs=DNN_EPOCHS_FINAL); final_fit_params[
                    'callbacks'] = []
                final_model.fit(X_train_final, y_full.loc[X_train_final.index], **final_fit_params)

                final_model_pipeline = {
                    'model': final_model, 'model_name': model_name, 'architecture': arch_name,
                    'features': current_feature_set, 'best_params_from_cv': best_params, 'data_format_type': 'ohe'
                }
                with open(os.path.join(output_dir, f"{full_model_name}.pkl"), 'wb') as f:
                    pickle.dump(final_model_pipeline, f)
            except Exception as e:
                print(f"保存 {full_model_name} 失败: {e} (记录耗时后将继续运行)")

            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            execution_times_records.append({
                'Model': full_model_name, 'Hardware_Config': hardware_info_str,
                'Execution_Time_Seconds': round(elapsed_time, 2), 'Execution_Time_Formatted': formatted_time
            })
            print(f"    -> [耗时统计] {full_model_name} 完成，耗时: {formatted_time}")

            # --- 新增: 实时增量保存 ---
            df_times_partial = pd.DataFrame(execution_times_records)
            df_times_partial.to_csv(os.path.join(viz_output_dir, "model_execution_times_V5.csv"), index=False,
                                    encoding='utf-8-sig')

    # --- 运行 4: Native 组 (DNN-Native) ---
    print("\n" + "=" * 70)
    print("V5-FullTune 实验 4: 运行 Native 组 (DNN-Native)")
    print("=" * 70)
    for model_name in MODELS_TO_TEST_NATIVE_DNN:
        if (model_name == 'DNN' and not KERAS_AVAILABLE): continue

        for arch_name, (current_feature_set, df_key, current_dynamic_dim) in FEATURE_SETS_NATIVE.items():
            full_model_name = f"{model_name}-{arch_name}"

            if full_model_name in COMPLETED_MODELS:
                print(f"  - ⏩ 已跳过 (本地记录显示已完成): {full_model_name}")
                continue

            print(f"  - 正在评估: {full_model_name}")
            start_time = time.time()

            oof_preds = np.full(len(y_full), np.nan)
            oof_true = np.full(len(y_full), np.nan)
            current_cat_features = native_cat_meta[arch_name]
            fold_best_params = {}

            for fold, (train_idx_outer, test_idx_outer) in enumerate(outer_cv_indices):
                if df_key == 'robust':
                    X_full_df_native_source = X_full_df_native_robust
                elif df_key == 'pca':
                    X_full_df_native_source = X_full_df_native_pca

                X_train_outer_df = X_full_df_native_source.iloc[train_idx_outer]
                y_train_outer_series = y_full.iloc[train_idx_outer]
                X_test_outer_df = X_full_df_native_source.iloc[test_idx_outer]
                y_test_outer_series = y_full.iloc[test_idx_outer]

                df_cv_split = X_train_outer_df[['Year_CV']].rename(columns={'Year_CV': 'Year'})
                inner_cv_folds = create_inner_cv(df_cv_split, MIN_TRAIN_YEARS_INNER, N_INNER_SPLITS_FAST_TEST)

                best_params, _ = run_tuning(X_train_outer_df[current_feature_set], y_train_outer_series, model_name,
                                            inner_cv_folds, data_format_type='native', city_vocab_size=city_vocab_size,
                                            year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim,
                                            native_cat_features=current_cat_features)
                fold_best_params = best_params

                model_fold = get_model_instance(model_name, best_params, 'native', city_vocab_size=city_vocab_size,
                                                year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim)

                X_train_fit = X_train_outer_df[current_feature_set]
                val_train_idx, val_test_idx = inner_cv_folds[-1]
                X_val_fit = X_train_outer_df.loc[val_test_idx][current_feature_set]

                X_train_fit_input = _prep_dnn_native_input(X_train_fit, current_cat_features['dynamic'])
                X_val_fit_input = _prep_dnn_native_input(X_val_fit, current_cat_features['dynamic'])
                fit_params = {'validation_data': (X_val_fit_input, y_train_outer_series.loc[X_val_fit.index]),
                              'callbacks': [EarlyStopping(monitor='val_loss', patience=DNN_ES_PATIENCE,
                                                          restore_best_weights=True)]}

                model_fold.fit(X_train_fit_input, y_train_outer_series, **fit_params)

                X_test_predict = X_test_outer_df[current_feature_set]
                if len(X_test_predict) > 0:
                    preds_test = model_fold.predict(
                        _prep_dnn_native_input(X_test_predict, current_cat_features['dynamic']))
                    oof_preds[X_test_predict.index] = preds_test
                    oof_true[X_test_predict.index] = y_test_outer_series.loc[X_test_predict.index].values

            all_cv_metrics.append({
                'Model': full_model_name, 'R2 (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[0],
                'RMSE (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[1],
                'MAE (log-scale)': calculate_metrics_log_scale(oof_true, oof_preds)[2]
            })
            all_oof_predictions[full_model_name] = oof_preds
            all_best_params[full_model_name] = fold_best_params

            try:
                if df_key == 'robust':
                    X_full_df_native_source = X_full_df_native_robust
                elif df_key == 'pca':
                    X_full_df_native_source = X_full_df_native_pca
                X_train_final = X_full_df_native_source[current_feature_set]

                final_model = get_model_instance(model_name, best_params, 'native', city_vocab_size=city_vocab_size,
                                                 year_vocab_size=year_vocab_size, dynamic_dim=current_dynamic_dim)
                final_model.set_params(epochs=DNN_EPOCHS_FINAL)
                final_model.fit(_prep_dnn_native_input(X_train_final, current_cat_features['dynamic']),
                                y_full.loc[X_train_final.index], callbacks=[])

                final_model_pipeline = {
                    'model': final_model, 'model_name': model_name, 'architecture': arch_name,
                    'features': current_feature_set, 'best_params_from_cv': best_params, 'data_format_type': 'native'
                }
                with open(os.path.join(output_dir, f"{full_model_name}.pkl"), 'wb') as f:
                    pickle.dump(final_model_pipeline, f)
            except Exception as e:
                print(f"保存 {full_model_name} 失败: {e} (记录耗时后将继续运行)")

            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            execution_times_records.append({
                'Model': full_model_name, 'Hardware_Config': hardware_info_str,
                'Execution_Time_Seconds': round(elapsed_time, 2), 'Execution_Time_Formatted': formatted_time
            })
            print(f"    -> [耗时统计] {full_model_name} 完成，耗时: {formatted_time}")

            # --- 新增: 实时增量保存 ---
            df_times_partial = pd.DataFrame(execution_times_records)
            df_times_partial.to_csv(os.path.join(viz_output_dir, "model_execution_times_V5.csv"), index=False,
                                    encoding='utf-8-sig')

    # --- 步骤 6: 报告 V5-Final 结果 ---
    print("\n" + "=" * 70)
    print(f"步骤 6: V5-FullTune 嵌套 CV 评估结果")
    print("=" * 70)
    if not all_cv_metrics: return
    df_results = pd.DataFrame(all_cv_metrics).sort_values(by='R2 (log-scale)', ascending=False)
    print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))
    df_results.to_csv(os.path.join(viz_output_dir, "V5_FullTune_model_evaluation.csv"), index=False,
                      encoding='utf-8-sig')

    # --- 步骤 7: VIF 分析 ---
    print("\n" + "=" * 70)
    print("步骤 7: 运行诊断分析 (重现 VIF)")
    print("=" * 70)
    if VIF_AVAILABLE and 'MLR-M2_Full_L1-OHE' in all_best_params:
        try:
            X_train_final_vif = X_full_df_ohe.drop(columns=['Year_CV'])[FEATURE_SETS_OHE['M2_Full_L1-OHE']].dropna()
            dynamic_features_in_m2 = [f for f in DYNAMIC_FEATURES if f in X_train_final_vif.columns]
            if dynamic_features_in_m2:
                vif_data = pd.DataFrame({"feature": dynamic_features_in_m2})
                vif_data["VIF"] = [
                    variance_inflation_factor(X_train_final_vif.values, list(X_train_final_vif.columns).index(f)) for f
                    in dynamic_features_in_m2]
                vif_data = vif_data.sort_values(by="VIF", ascending=False)
                vif_data.to_csv(os.path.join(viz_output_dir, "VIF_MLR_M2_dynamic_features_V5_FullTune.csv"),
                                index=False, encoding='utf-8-sig')
                print(tabulate(vif_data, headers='keys', tablefmt='grid', showindex=False))
        except Exception as e:
            print(f"  - VIF 分析失败: {e}")

    # --- 步骤 8: 输出并保存硬件配置与模型运行时间 ---
    print("\n" + "=" * 70)
    print("步骤 8: 汇总并保存模型运行时间与硬件配置信息")
    print("=" * 70)
    if execution_times_records:
        df_times = pd.DataFrame(execution_times_records)
        times_save_path = os.path.join(viz_output_dir, "model_execution_times_V5.csv")
        df_times.to_csv(times_save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 模型运行时间与硬件配置已成功保存至: {times_save_path}\n")

        # 仅打印精简版便于在终端查看
        print(tabulate(df_times[['Model', 'Execution_Time_Formatted', 'Execution_Time_Seconds']], headers='keys',
                       tablefmt='grid', showindex=False))
    else:
        print("⚠️ 未记录到任何模型的运行时间。")

    print("\n" + "=" * 70)
    print(f"V5-FullTune 验证与模型保存执行完毕。")
    print(f"所有输出均已归档至目录: ./{BASE_OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()