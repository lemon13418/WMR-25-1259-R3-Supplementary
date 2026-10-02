# -*- coding: utf-8 -*-
"""
visualization_manuscript.py  (自足版: 手稿6图生成器 - 全部数据来自数据包内产物)

可从 Data_Package 根目录直接运行:
    python 02_code/visualization_manuscript.py [output_dir]

数据来源(包内固定路径, 无外部/绝对路径依赖):
  - 基准表:       04_outputs/final_benchmark_values.csv
  - Optuna study: 04_outputs/optuna_study_xgb_m2_full.pkl (XGBoost M2-Full, 末折)
  - OOF 预测:     04_outputs/oof_predictions_xgb_m2_full.csv
  - SHAP 产物:    04_outputs/shap_values.pkl, 04_outputs/X_test_for_shap.pkl
  - 全局模型:     03_models/XGBoost-M2_Full_L1-Native.pkl
  - (可选)收敛序列: 04_outputs/validation/convergence_sequences.csv
     若不存在, Fig.3B 退化为仅绘制包内 Optuna study 的 XGBoost 轨迹
     (提交版 Fig.3B 由作者从其完整运行日志绘制, 含 EBM 曲线与 trial-60 阈值线)。

输出(默认写入参数指定目录, 缺省为数据包根下 '06_figures'):
  Fig1..Fig6 的 PNG (600dpi; Fig2/Fig4 为 300dpi) + 矢量 PDF,
  以及 ALE 数值产物 04_outputs/ale/ALE_1D_PerCapitaGDP.csv,
  ALE_1D_EnergyIntensity.csv, ALE_2D_Dev_x_Urb.csv。

注意: 不同 matplotlib/PyALE 版本的渲染细节可能与提交版图件存在像素级差异;
面板内容、坐标范围与标注数值应与提交版一致。
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

BISHE = os.path.dirname(os.path.abspath(__file__))
if BISHE not in sys.path:
    sys.path.insert(0, BISHE)
ROOT = os.path.abspath(os.path.join(BISHE, '..'))
OUT = os.path.join(ROOT, '06_figures')
if len(sys.argv) > 1:
    OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
})

ALGO_COLOR_TAB10 = {
    'XGBoost': '#1f77b4', 'DNN': '#ff7f0e', 'MLR': '#2ca02c',
    'LGBM': '#d62728', 'SVR': '#9467bd', 'EBM': '#8c564b',
}

DYN = ['PerCapitaGDP', 'SecondaryIndustryShare', 'TertiaryIndustryShare',
       'FiscalCapacityPerCapita', 'ConstructionIntensity', 'UrbanizationRate',
       'PatentsPerCapita', 'EnergyIntensity', 'PolicyCount', 'PublicAwarenessIndex']

OOF_CSV = os.path.join(ROOT, '04_outputs', 'oof_predictions_xgb_m2_full.csv')
BENCH_CSV = os.path.join(ROOT, '04_outputs', 'final_benchmark_values.csv')
STUDY_PKL = os.path.join(ROOT, '04_outputs', 'optuna_study_xgb_m2_full.pkl')
CONV_CSV = os.path.join(ROOT, '04_outputs', 'validation', 'convergence_sequences.csv')
SHAP_PKL = os.path.join(ROOT, '04_outputs', 'shap_values.pkl')
SHAP_X_PKL = os.path.join(ROOT, '04_outputs', 'X_test_for_shap.pkl')
MODEL_PKL = os.path.join(ROOT, '03_models', 'XGBoost-M2_Full_L1-Native.pkl')
DATA_CSV = os.path.join(ROOT, '01_data', 'final_dataset_lpc_robust_scaled.csv')
ALE_DIR = os.path.join(ROOT, '04_outputs', 'ale')


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'))
    fig.savefig(os.path.join(OUT, name + '.pdf'))


# ================= Fig.1 性能基准 =================
def fig1():
    if not os.path.exists(BENCH_CSV):
        raise FileNotFoundError('benchmark table not found: %s' % BENCH_CSV)
    dfr = pd.read_csv(BENCH_CSV)
    saved = dict(plt.rcParams)
    mpl_rc = dict(plt.rcParams)
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['DejaVu Sans', 'Arial']})
    vals = {r['Model']: (r['R2 (log-scale)'], r['RMSE (log-scale)'], r['MAE (log-scale)'])
            for r in dfr.to_dict('records') if not pd.isna(r['R2 (log-scale)'])}

    items = []
    for key, (r2, rm, mae) in vals.items():
        parts = key.split('-')
        algo = parts[0]
        enc = parts[-1]
        combo = 'M1-Baseline' if 'M1_Baseline' in key else 'M2-PCA' if 'M2_PCA' in key else 'M2-Full'
        items.append((f'{algo} ({combo}) - {enc}', ALGO_COLOR_TAB10.get(algo, '#888888'), r2, rm, mae))
    items.sort(key=lambda t: -t[2])
    n = len(items)
    labels = [t[0] for t in items]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for ax in axes:
        ax.set_ylim(n - 0.5, -0.5)
    cols = ['R$^2$ Score', 'RMSE', 'MAE']
    better = ['(Higher is better)', '(Lower is better)', '(Lower is better)']
    pans = [
        (0.34, 0.96, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        (0.00, 0.43, [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]),
        (0.00, 0.32, [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
    ]
    for ax, ci, (xmin, xmax, xticks) in zip(axes, range(3), pans):
        ax.set_facecolor('white')
        for j, (lab, color, v0, v1, v2) in enumerate(items):
            val = (v0, v1, v2)[ci]
            if not np.isnan(val):
                ax.barh(j, val, color=color, edgecolor='none', height=0.82)
                ax.text(max(val, xmin) + (xmax - xmin) * 0.008, j, f'{val:.4f}',
                        ha='left', va='center', fontsize=6.8, color='black')
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{t:.2f}' if t < 0.1 or t >= 1 else (f'{t:.1f}' if t >= 0.1 else f'{t:.2f}')
                            for t in xticks], fontsize=7.2)
        ax.tick_params(axis='x', direction='out', length=2.5, width=0.6, color='#555555')
        ax.set_xlabel(f'{cols[ci]} {better[ci]}', fontsize=9, fontweight='bold', labelpad=4)
        ax.grid(axis='x', linestyle='-', alpha=0.35, color='#c8c8c8', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax is axes[0]:
            ax.set_yticks([])
            for j, lab in enumerate(labels):
                ax.text(-0.015, j, lab, ha='right', va='center', fontsize=6.6, color='black',
                        transform=ax.get_yaxis_transform())
        else:
            ax.set_yticks([])
            ax.tick_params(axis='y', length=0)

    handles = [Line2D([0], [0], color=c, lw=0, marker='s', markersize=5.5,
                      label=a, markerfacecolor=c) for a, c in ALGO_COLOR_TAB10.items()]
    leg = axes[0].legend(handles=handles, title='Model Type', title_fontsize=7,
                         loc='lower right', ncol=1, fontsize=6.2, frameon=True,
                         facecolor='white', edgecolor='#888888', handlelength=0.7,
                         handletextpad=0.4, borderaxespad=0.4, handleheight=0.7)
    leg.get_frame().set_linewidth(0.5)

    for ax in axes:
        sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.subplots_adjust(left=0.20)
    fig.text(0.030, 0.5, 'Model Configuration', rotation=90, va='center', ha='center',
             fontsize=9, fontweight='bold')
    save_fig(fig, 'Fig1_Performance_Benchmark')
    plt.close(fig)
    plt.rcParams.update(saved)
    print('Fig1 done | rows: %d / 17' % n)


# ================= Fig.2 共线性结构与 PCA 敏感性 =================
def fig2():
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, KFold
    import xgboost as xgb

    saved = dict(plt.rcParams)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.dpi': 300, 'savefig.dpi': 300, 'axes.unicode_minus': False,
        'axes.facecolor': '#f9f9f9', 'savefig.bbox': 'tight',
    })

    def get_model_performance(X, y, model_type='xgb'):
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1) \
            if model_type == 'xgb' else LinearRegression()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        return -scores.mean()

    df2 = pd.read_csv(DATA_CSV)
    dyn_cols = DYN
    X_dyn = df2[dyn_cols].select_dtypes(include=[np.number])
    y = df2['Target']

    n_viz = min(5, X_dyn.shape[1])
    tr_mask = (df2['Year'].astype(int) <= 2019).to_numpy()
    pca_viz = PCA(n_components=n_viz).fit(X_dyn.loc[tr_mask])
    X_pca_df = pd.DataFrame(pca_viz.transform(X_dyn), columns=['PC%d' % (i + 1) for i in range(n_viz)])

    baseline_rmse = get_model_performance(X_dyn, y, model_type='xgb')
    n_components_list = range(1, X_dyn.shape[1] + 1)
    pca_rmses, explained_variances = [], []
    for n in n_components_list:
        pca = PCA(n_components=n).fit(X_dyn.loc[tr_mask])
        X_pca_current = pca.transform(X_dyn)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))
        pca_rmses.append(get_model_performance(X_pca_current, y, model_type='xgb'))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # A. 原始动态特征的相关性 (10 个变量, 与主文 Fig.2A 一致)
    # 轴标签: 与正文 (Section 2.1) 及 Fig.5/Fig.6 一致的名义标签
    FEATURE_LABELS = {
        'PerCapitaGDP': 'Economic Development Level',
        'SecondaryIndustryShare': 'Industrial Structure',
        'TertiaryIndustryShare': 'Economic Modernization',
        'FiscalCapacityPerCapita': 'Public Financial Resources',
        'ConstructionIntensity': 'Infrastructure Expansion',
        'UrbanizationRate': 'Urbanization Level',
        'PatentsPerCapita': 'Technological Innovation',
        'EnergyIntensity': 'Energy Efficiency',
        'PolicyCount': 'Government Attitude',
        'PublicAwarenessIndex': 'Public Environmental Awareness',
    }
    X_dyn_labelled = X_dyn.rename(columns=FEATURE_LABELS)
    sns.heatmap(X_dyn_labelled.corr(), mask=np.triu(np.ones_like(X_dyn_labelled.corr(), dtype=bool)),
                annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('A. Collinearity of the Ten Dynamic Drivers', fontweight='bold')
    # B. 前 5 个主成分的正交性
    sns.heatmap(X_pca_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('B. First 5 PCs: Near-Zero Correlation (Full Sample)', fontweight='bold')

    ax_sens = axes[2]
    color_rmse = '#E63946'
    color_var = '#457B9D'
    ax_sens.set_xlabel('Number of PCA Components')
    ax_sens.set_ylabel('Model RMSE (Lower is Better)', color=color_rmse, fontweight='bold')
    line1 = ax_sens.plot(n_components_list, pca_rmses, color=color_rmse, marker='o',
                         label='PCA-XGBoost RMSE')
    ax_sens.tick_params(axis='y', labelcolor=color_rmse)
    line2 = ax_sens.axhline(y=baseline_rmse, color='black', linestyle='--', linewidth=2,
                            label='Baseline (Raw Features)')
    ax_var = ax_sens.twinx()
    ax_var.set_ylabel('Cumulative Explained Variance', color=color_var, fontweight='bold')
    line3 = ax_var.plot(n_components_list, explained_variances, color=color_var,
                        linestyle=':', marker='x', label='Explained Variance')
    ax_var.tick_params(axis='y', labelcolor=color_var)
    ax_var.set_ylim(0, 1.05)
    try:
        idx_95 = next(x[0] for x in enumerate(explained_variances) if x[1] >= 0.95)
        ax_var.axvline(x=n_components_list[idx_95], color='gray', linestyle='-.', alpha=0.5)
        ax_var.text(n_components_list[idx_95], 0.5, ' 95% Variance', rotation=90,
                    verticalalignment='center', color='gray')
    except StopIteration:
        pass
    ax_sens.legend(line1 + [line2] + line3,
                   [l.get_label() for l in line1] + ['Baseline (Raw Features)'] +
                   ['Explained Variance'], loc='center right')
    axes[2].set_title('C. Impact of PCA Transformation on XGBoost Performance', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'Fig2_Multicollinearity_Paradox')
    plt.close(fig)
    plt.rcParams.update(saved)
    print('Fig2 done | raw baseline RMSE = %.4f | PCA-10 RMSE = %.4f' % (baseline_rmse, pca_rmses[-1]))


# ================= Fig.3 Optuna 三面板 =================
def fig3():
    import optuna
    if not os.path.exists(STUDY_PKL):
        raise FileNotFoundError('optuna study not found: %s' % STUDY_PKL)
    with open(STUDY_PKL, 'rb') as f:
        study = pickle.load(f)
    vals = np.array([t.value for t in study.trials if t.value is not None])
    best = np.minimum.accumulate(vals)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # A 优化历史
    axes[0].scatter(np.arange(1, len(vals) + 1), vals, s=18, color='#1f77b4', alpha=0.65,
                    label='Objective Value')
    axes[0].plot(np.arange(1, len(vals) + 1), best, color='#d62728', lw=2,
                 drawstyle='steps-post', label='Best Value')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_xticks(range(0, 101, 20))
    axes[0].set_title('A. Optimisation History')
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', frameon=False)
    # B 收敛轨迹 (优先读取作者固化数据; 否则由包内 study 推导 XGBoost 轨迹)
    if os.path.exists(CONV_CSV):
        c = pd.read_csv(CONV_CSV)
        axes[1].plot(c['trial'], c['XGBoost'], color='#1f77b4', lw=2.2, ls='-',
                     label='Complex Model (XGBoost)')
        if 'EBM' in c and c['EBM'].notna().any():
            axes[1].plot(c['trial'], c['EBM'], color='#ff7f0e', lw=2.2, ls='--',
                         label='Simple Model (EBM)')
        axes[1].axvline(60, color='grey', ls=':', lw=1.8, label='Plateau Threshold')
    else:
        axes[1].plot(np.arange(1, len(vals) + 1), vals, color='#1f77b4', lw=2.2,
                     label='XGBoost (M2-Full, last outer fold)')
        axes[1].annotate('EBM series not shipped; add convergence_sequences.csv to\n'
                         '04_outputs/validation/ to reproduce the full panel',
                         xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
                         fontsize=8, color='grey')
    axes[1].set_ylabel('Objective Value (RMSE)')
    axes[1].set_xlabel('Number of Trials')
    axes[1].set_title('B. Convergence Trajectory')
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=9)
    # C 超参重要性 (Optuna mean-decrease-impurity evaluator, n_trees=64, seed=42 -> 确定性, 与投稿图一致)
    from optuna.importance import get_param_importances, MeanDecreaseImpurityImportanceEvaluator
    evaluator = MeanDecreaseImpurityImportanceEvaluator(n_trees=64, seed=42)
    imp = get_param_importances(study, evaluator=evaluator)
    names = list(imp.keys())
    scores = list(imp.values())
    order = np.argsort(scores)
    bars = axes[2].barh(np.arange(len(names)), [scores[i] for i in order],
                        color='#6A5ACD', alpha=0.9, edgecolor='black', linewidth=0.3)
    axes[2].set_yticks(np.arange(len(names)))
    axes[2].set_yticklabels([names[i] for i in order], fontsize=9)
    axes[2].set_xlabel('Hyperparameter Importance')
    for b in bars:
        w = b.get_width()
        axes[2].text(w + 0.005, b.get_y() + b.get_height() / 2, f'{w:.2f}',
                     ha='left', va='center', fontsize=8)
    axes[2].set_title('C. Hyperparameter Importance')
    axes[2].grid(axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, 'Fig3_Optuna_History_Convergence_Importance')
    plt.close(fig)
    print('Fig3 done')


# ================= Fig.4 预测保真度与残差 (内联三面板) =================
def fig4():
    if not os.path.exists(OOF_CSV):
        raise FileNotFoundError('OOF predictions not found: %s' % OOF_CSV)
    oof = pd.read_csv(OOF_CSV)
    oof['Test_Year'] = oof['Test_Year'].astype(int)
    true = oof['True_Values'].to_numpy()
    pred = oof['Predicted_Values'].to_numpy()
    resid = true - pred
    years = sorted(oof['Test_Year'].unique())

    r2_val = r2_score(true, pred)
    rmse_val = np.sqrt(mean_squared_error(true, pred))
    mae_val = mean_absolute_error(true, pred)
    text_content = (
        "Overall Metrics\n"
        f"R^2 = {r2_val:.4f}\n"
        f"RMSE = {rmse_val:.4f}\n"
        f"MAE = {mae_val:.4f}\n"
        f"(N = {len(oof)})"
        "\nYearly Breakdown (2018-2022)")
    for year in years:
        m = oof['Test_Year'] == year
        yt, yp = true[m], pred[m]
        text_content += ("\n%d: R^2=%.3f | RMSE=%.3f | MAE=%.3f"
                         % (year, r2_score(yt, yp),
                            np.sqrt(mean_squared_error(yt, yp)),
                            mean_absolute_error(yt, yp)))

    fig = plt.figure(figsize=(18.3, 8.9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.9, 1.1], height_ratios=[1, 1],
                           left=0.06, right=0.97, bottom=0.10, top=0.93,
                           wspace=0.35, hspace=0.5)
    # A 面板: 主区 + 上/右边缘直方图 (与投稿图一致的 jointplot 结构)
    axA = fig.add_subplot(gs[:, 0])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axA)
    axH = divider.append_axes('top', size='18%', pad=0.08)
    axR = divider.append_axes('right', size='18%', pad=0.08)
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    axis_lim = [min(true.min(), pred.min()) * 0.95, max(true.max(), pred.max()) * 1.05]
    hb = axA.hexbin(true, pred, gridsize=30, cmap='viridis', edgecolor='none', mincnt=1)
    fig.colorbar(hb, ax=axA, label='Count')
    bins = np.linspace(axis_lim[0], axis_lim[1], 40)
    axH.hist(pred, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='white', linewidth=0.2)
    axR.hist(true, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='white', linewidth=0.2,
             orientation='horizontal')
    axH.set_xlim(axis_lim)
    axR.set_ylim(axis_lim)
    for axm in (axH, axR):
        axm.set_xticks([])
        axm.set_yticks([])
    axH.set_ylabel('Count', fontsize=8)
    axR.set_xlabel('Count', fontsize=8)
    lr = LinearRegression().fit(true.reshape(-1, 1), pred)
    x_fit = np.array(axis_lim).reshape(-1, 1)
    axA.plot(x_fit, lr.predict(x_fit), '#0072B2', linewidth=2, label='OLS Fit')
    axA.set_xlim(axis_lim)
    axA.set_ylim(axis_lim)
    axA.set_xlabel('True Values (log-scale)')
    axA.set_ylabel('Predicted Values (log-scale)')
    axA.text(0.05, 0.95, text_content, transform=axA.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8), fontsize=9)
    axA.legend(loc='lower right')
    axA.set_title('A. Prediction Fidelity', fontweight='bold')

    # B. 残差 vs 预测值 (±2σ)
    axB.scatter(pred, resid, s=20, alpha=0.7, color='#1f77b4', edgecolor='white', linewidth=0.3)
    xb = np.linspace(pred.min(), pred.max(), 100)
    lrB = LinearRegression().fit(pred.reshape(-1, 1), resid)
    axB.plot(xb, lrB.predict(xb.reshape(-1, 1)), color='#E63946', lw=2, label='OLS Fit')
    axB.axhline(0, color='black', lw=1.2, label='Zero Error')
    sd = resid.std()
    axB.axhline(2 * sd, color='grey', ls='--', lw=1.2, label='\u00b12 Std Dev')
    axB.axhline(-2 * sd, color='grey', ls='--', lw=1.2)
    axB.set_xlabel('Predicted Values (log-scale)')
    axB.set_ylabel('Residuals')
    axB.set_title('B. Residuals vs. Predicted Values\n(Check for Homoscedasticity)', fontweight='bold')
    axB.legend(loc='upper right', fontsize=8)

    # C. 逐年残差箱线图 (viridis, 对应主文 Fig.4C)
    box_data = [resid[oof['Test_Year'].to_numpy() == y] for y in years]
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(years)))
    bp = axC.boxplot(box_data, labels=[str(y) for y in years], patch_artist=True, widths=0.55)
    for patch, cc in zip(bp['boxes'], colors):
        patch.set_facecolor(cc)
        patch.set_alpha(0.75)
    axC.axhline(0, color='black', lw=1.0, ls='-', alpha=0.6)
    axC.set_xlabel('Test Year (Fold)')
    axC.set_ylabel('Residual Distribution')
    axC.set_title('C. Residual Stability Over Time\n(Check for Temporal Bias)', fontweight='bold')

    save_fig(fig, 'Fig4_Prediction_Fidelity_Residuals')
    plt.close(fig)
    print('Fig4 done')


# ================= Fig.5 SHAP (visualization.py 式布局) =================
def fig5():
    import visualization as viz
    for p in [SHAP_PKL, SHAP_X_PKL]:
        if not os.path.exists(p):
            raise FileNotFoundError('SHAP artifacts not found: %s' % p)
    shap_values = pickle.load(open(SHAP_PKL, 'rb'))
    X_test = pd.read_pickle(SHAP_X_PKL)
    viz.OUTPUT_DIR = OUT
    viz.plot_figure_3(shap_values, X_test, filename='Fig5_SHAP_Global.png')
    print('Fig5 done')


# ================= Fig.6 ALE (1D 主效应 + 2D 交互; 写入数值产物) =================
def fig6(compute=True, write_ale=True):
    from PyALE import ale
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError('model not found: %s' % MODEL_PKL)
    with open(MODEL_PKL, 'rb') as f:
        pipe = pickle.load(f)
    model = pipe['model']
    df = pd.read_csv(DATA_CSV)
    df['City'] = df['City'].astype('category')
    df['Year'] = df['Year'].astype('category')
    Xf = df[pipe['features']]

    a1 = ale(X=Xf, model=model, feature=['PerCapitaGDP'], grid_size=50, include_CI=True, plot=False)
    aE = ale(X=Xf, model=model, feature=['EnergyIntensity'], grid_size=30, include_CI=True, plot=False)
    a2 = ale(X=Xf, model=model, feature=['PerCapitaGDP', 'UrbanizationRate'], grid_size=5, plot=False)

    if write_ale:
        os.makedirs(ALE_DIR, exist_ok=True)
        a1.index.name = 'PerCapitaGDP'
        a1.to_csv(os.path.join(ALE_DIR, 'ALE_1D_PerCapitaGDP.csv'), encoding='utf-8-sig')
        aE.index.name = 'EnergyIntensity'
        aE.to_csv(os.path.join(ALE_DIR, 'ALE_1D_EnergyIntensity.csv'), encoding='utf-8-sig')
        a2.index.name = 'PerCapitaGDP'
        a2.to_csv(os.path.join(ALE_DIR, 'ALE_2D_Dev_x_Urb.csv'), encoding='utf-8-sig')
        print('ALE CSVs written to', ALE_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    x = a1.index.values.astype(float)
    eff = a1['eff'].values
    axes[0].step(x, eff, where='post', color='#1f77b4', lw=2.5)
    axes[0].fill_between(x, a1['lowerCI_95%'].values, a1['upperCI_95%'].values,
                         color='#1f77b4', alpha=0.15)
    axes[0].axhline(0, color='black', lw=1.2)
    axes[0].plot(Xf['PerCapitaGDP'], np.full(len(Xf), -0.017), '|', color='black',
                 alpha=0.25, ms=6)
    axes[0].set_xlabel('Economic Development Level')
    axes[0].set_ylabel('Effect on prediction (centered)')
    axes[0].set_xlim(-1.03, 2.76)
    axes[0].set_ylim(-0.04, 0.04)
    axes[0].set_title('A. Main Effect of Economic Development Level')
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    arr = a2.values.astype(float)
    im = axes[1].imshow(arr, cmap='viridis', aspect='auto',
                        extent=[a2.columns.values.min(), a2.columns.values.max(),
                                a2.index.values.max(), a2.index.values.min()])
    axes[1].set_yticks(np.linspace(a2.index.values.min(), a2.index.values.max(), 6))
    axes[1].set_xticks(np.linspace(a2.columns.values.min(), a2.columns.values.max(), 6))
    cb = fig.colorbar(im, ax=axes[1])
    cb.set_label('Effect on prediction (centered)')
    axes[1].set_xlabel('Urbanization Level')
    axes[1].set_ylabel('Economic Development Level')
    axes[1].set_title('B. Interaction Effect: Development and Urbanization')
    fig.tight_layout()
    save_fig(fig, 'Fig6_ALE_Main_Interaction')
    plt.close(fig)
    print('Fig6 done')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print('ALL DONE ->', OUT)