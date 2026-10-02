import pandas as pd
import numpy as np
import pickle
import os
import warnings

# --- (新) 导入 matplotlib 并设置后端 ---
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde
# --- (新) 导入 Fig 4 所需的平滑工具 ---
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# --- 依赖库检查 ---
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    print("警告: 'optuna' 库或其 matplotlib 后端未安装。图 2 将无法绘制。")
    OPTUNA_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("警告: 'shap' 库未安装。图 3 和 图 4 将无法绘制。")
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- (新) 全局配置 & 学术样式 (Times New Roman, 高 DPI, 浅灰背景) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#f8f9fa',  # 轻灰色背景
    'figure.facecolor': 'white'
})

# --- 路径配置 (与旧代码一致) ---
VIZ_DATA_DIR = 'visualization_data'
MODEL_EVAL_FILE = 'V5_Final_model_evaluation.csv'
OOF_PRED_FILE = os.path.join(VIZ_DATA_DIR, 'oof_predictions_with_folds.pkl')
OPTUNA_STUDY_FILE = os.path.join(VIZ_DATA_DIR, 'optuna_study_xgb_m2_full.pkl')
SHAP_VALUES_FILE = os.path.join(VIZ_DATA_DIR, 'shap_values.pkl')
SHAP_X_TEST_FILE = os.path.join(VIZ_DATA_DIR, 'X_test_for_shap.pkl')
OUTPUT_DIR = 'visualization_figures'

# --- (新) 和谐的调色板 ---
COLORS = {
    'primary': '#2E86AB',  # 深蓝
    'secondary': '#A23B72',  # 深粉
    'accent1': '#F18F01',  # 橙
    'accent2': '#C73E1D',  # 红
    'neutral1': '#6B7280',  # 灰
    'neutral2': '#374151',  # 深灰
    'positive': '#059669',  # 绿
    'negative': '#DC2626',  # 红
    'background': '#f8f9fa'  # 轻灰
}

# --- (新) 更新的颜色与形状映射 ---
color_map_algorithm = {
    'XGBoost': COLORS['primary'],  # 深蓝
    'LGBM': COLORS['positive'],  # 绿
    'DNN': COLORS['accent1'],  # 橙
    'MLR': COLORS['neutral1'],  # 灰
    'SVR': COLORS['secondary'],  # 深粉
    'EBM': COLORS['accent2']  # 红
}
shape_map_encoding = {
    'Native': 'o',  # 圆
    'OHE': 's'  # 方
}
color_map_featureset = {
    'M1_Baseline': COLORS['neutral1'],  # 灰
    'M2_PCA_L1': COLORS['accent1'],  # 橙
    'M2_Full_L1': COLORS['primary']  # 深蓝
}

# --- (新) 更新的英文特征名称映射 ---
FEATURE_NAME_MAP = {
    'PerCapitaGDP': 'Economic Development Level',
    'SecondaryIndustryShare': 'Industrial Structure',
    'TertiaryIndustryShare': 'Economic Modernization ',
    'PolicyCount': 'Government Attitude',
    'FiscalCapacityPerCapita': 'Public Financial Resources',
    'PatentsPerCapita': 'Technological Innovation',
    'EnergyIntensity': 'Energy Efficiency',
    'UrbanizationRate': 'Urbanization Level',
    'PublicAwarenessIndex': 'Public Environmental Awareness',
    'ConstructionIntensity': 'Infrastructure Expansion',
    'City': 'City',
    'Year': 'Year'
}


# --- 1. 绘图函数 (已更新样式) ---

def plot_figure_1(df_results, filename="figure_1_performance_ablation.png"):
    """
    (修改 1) 图 1：保持 V5 的 GridSpec 布局
    """
    if df_results.empty:
        print("     - 警告: df_results 为空, 跳过图 1。")
        return

    # --- 数据预处理 (与旧代码一致) ---
    def parse_model_name(name):
        parts = name.split('-')
        algo = parts[0]
        encoding = parts[-1]
        feature_set = 'M1_Baseline' if 'M1_Baseline' in name else \
            'M2_PCA_L1' if 'M2_PCA_L1' in name else \
                'M2_Full_L1' if 'M2_Full_L1' in name else 'Unknown'
        return algo, feature_set, encoding

    df_plot = df_results.copy()
    df_plot[['Algorithm', 'Feature Set', 'Encoding']] = df_plot['Model'].apply(lambda x: pd.Series(parse_model_name(x)))
    df_plot = df_plot.sort_values(by='R2 (log-scale)', ascending=False).reset_index(drop=True)

    # --- (修改 1) 保持 V5 的 GridSpec 布局 ---
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2, 1.5, 1], height_ratios=[4, 1])

    # --- 子图 A: 综合性能天梯图 (Cleveland Dot Plot) ---
    axA = fig.add_subplot(gs[:, 0])
    y_range = range(len(df_plot))

    for algo, color in color_map_algorithm.items():
        sub_df = df_plot[df_plot['Algorithm'] == algo]
        for encoding, shape in shape_map_encoding.items():
            sub_df_enc = sub_df[sub_df['Encoding'] == encoding]
            if not sub_df_enc.empty:
                axA.scatter(
                    sub_df_enc['R2 (log-scale)'],
                    sub_df_enc.index,
                    color=color,
                    marker=shape,
                    s=150,
                    label=f"Algorithm: {algo}" if shape == 'o' else "",
                    edgecolor='black',
                    linewidth=0.5
                )

    axA.set_yticks(y_range)
    axA.set_yticklabels(df_plot['Model'])
    axA.set_xlabel("$R^2$ (log-scale) on OOF Predictions")
    axA.set_ylabel("Model Configuration")
    axA.set_xlim(0.3, 1.0)
    axA.grid(axis='x', linestyle='--', alpha=0.7)
    axA.invert_yaxis()

    from matplotlib.lines import Line2D
    legend_elements = []
    for algo, color in color_map_algorithm.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color=color, label=f"Algorithm: {algo}", markersize=10, linestyle='None'))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='gray', label='Encoding: Native', markersize=10, linestyle='None'))
    legend_elements.append(
        Line2D([0], [0], marker='s', color='gray', label='Encoding: OHE', markersize=10, linestyle='None'))
    axA.legend(handles=legend_elements, loc='lower right')
    axA.set_title("A. Overall Model Performance Ranking", loc='left', fontsize=14, fontweight='bold')

    # --- 子图 B: 特征集消融对比 (Grouped Dot Plot) ---
    axB = fig.add_subplot(gs[0, 1])
    algo_for_ablation = ['XGBoost', 'LGBM', 'DNN', 'EBM']
    df_ablation = df_plot[df_plot['Algorithm'].isin(algo_for_ablation)]

    df_pivot = df_ablation.pivot_table(
        index='Algorithm',
        columns='Feature Set',
        values='R2 (log-scale)'
    ).reindex(algo_for_ablation)

    y_pos = np.arange(len(df_pivot))

    for i, algo in enumerate(df_pivot.index):
        vals = df_pivot.loc[algo].dropna()
        if not vals.empty:
            axB.plot([vals.min(), vals.max()], [i, i], color=COLORS['neutral1'], linestyle='-', linewidth=1, zorder=1)

    for fs_name, color in color_map_featureset.items():
        if fs_name in df_pivot.columns:
            axB.scatter(
                df_pivot[fs_name],
                y_pos,
                color=color,
                s=150,
                label=fs_name,
                zorder=2,
                edgecolor='black',
                linewidth=0.5
            )

    axB.set_yticks(y_pos)
    axB.set_yticklabels(df_pivot.index)
    axB.set_xlabel("$R^2$ (log-scale)")
    axB.legend(title="Feature Set", loc='lower right')
    axB.grid(axis='x', linestyle='--', alpha=0.7)
    axB.set_title("B. Feature Set Effectiveness", loc='left', fontsize=14, fontweight='bold')

    # --- 子图 C: 最佳模型性能指标 ---
    axC = fig.add_subplot(gs[1, 1])
    best_model_metrics = df_plot.iloc[0]
    metrics_data = {
        'R2': best_model_metrics['R2 (log-scale)'],
        'RMSE': best_model_metrics['RMSE (log-scale)'],
        'MAE': best_model_metrics['MAE (log-scale)']
    }

    axC.bar('R2', metrics_data['R2'], color=color_map_featureset['M2_Full_L1'], label='$R^2$')
    axC_twin = axC.twinx()
    axC_twin.bar('RMSE', metrics_data['RMSE'], color=color_map_algorithm['EBM'], label='RMSE')
    axC_twin.bar('MAE', metrics_data['MAE'], color=color_map_algorithm['DNN'], label='MAE')

    axC.set_ylabel("$R^2$ (log-scale)")
    axC.set_ylim(0.0, 1.0)

    axC_twin.set_ylabel("Error (log-scale)")
    axC_twin.set_ylim(0.0, 0.20)

    lines, labels = axC.get_legend_handles_labels()
    lines2, labels2 = axC_twin.get_legend_handles_labels()
    axC_twin.legend(lines + lines2, labels + labels2, loc='upper right')

    axC.set_xlabel("Metric")
    axC.text('R2', metrics_data['R2'] - 0.05, f"{metrics_data['R2']:.4f}", ha='center', va='top', fontsize=10,
             color='white', weight='bold')
    axC_twin.text('RMSE', metrics_data['RMSE'] + 0.005, f"{metrics_data['RMSE']:.4f}", ha='center', va='bottom',
                  fontsize=10, color=COLORS['neutral2'])
    axC_twin.text('MAE', metrics_data['MAE'] + 0.005, f"{metrics_data['MAE']:.4f}", ha='center', va='bottom',
                  fontsize=10, color=COLORS['neutral2'])
    axC.set_title("C. Best Model Metrics", loc='left', fontsize=14, fontweight='bold')

    plt.tight_layout(pad=2.0)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)  # RCParams 负责 dpi 和 bbox_inches
    print(f"     - 图 1 已保存至: {save_path}")
    plt.close(fig)


def plot_figure_2(study, filename_base="figure_2_optuna"):
    """
    (新修改) 图 2：移除子图 C，并将 A 和 B 横向合并为一张图。
    (新) 为兼容低版本，采用 "先保存A,B，再用PIL读取合并" 的策略。
    """
    if not (OPTUNA_AVAILABLE and study):
        print("     - 警告: Optuna study 不可用, 跳过图 2。")
        return

    # (新) 检查 PIL 是否成功导入
    if 'Image' not in globals() or Image.__class__ == type:
        try:
            from PIL import Image
        except ImportError:
            print("[错误] 无法合并图像，因为 'Pillow' 库未安装或无法导入。")
            print("     请运行: pip install Pillow")
            return

    try:
        print("     - 正在生成 Optuna 图像 (A 和 B)...")

        # --- 图 A: 优化历史图 ---
        fig_hist = optuna.visualization.plot_optimization_history(study)
        fig_hist.update_layout(title="A. Optimization History")
        # 暂时保存 A
        save_path_a = os.path.join(OUTPUT_DIR, f"{filename_base}_temp_A_history.png")
        fig_hist.write_image(save_path_a, scale=3)
        # print(f"     - 图 2(A) 临时保存至: {save_path_a}") # 调试用

        # --- 图 B: 参数重要性图 ---
        fig_imp = optuna.visualization.plot_param_importances(study)
        fig_imp.update_layout(title="B. Hyperparameter Importance")
        # 暂时保存 B
        save_path_b = os.path.join(OUTPUT_DIR, f"{filename_base}_temp_B_importance.png")
        fig_imp.write_image(save_path_b, scale=3)
        # print(f"     - 图 2(B) 临时保存至: {save_path_b}") # 调试用

        # --- (新) 步骤 3: 使用 PIL 合并 A 和 B ---
        print("     - 正在合并图 2(A) 和 2(B)...")

        # 使用 PIL 读取已保存的图像
        img_a = Image.open(save_path_a)
        img_b = Image.open(save_path_b)

        # 获取尺寸
        width_a, height_a = img_a.size
        width_b, height_b = img_b.size

        # 计算合并后的图像尺寸
        # (使用最高的高度，总的宽度)
        total_width = width_a + width_b
        max_height = max(height_a, height_b)

        # 创建一个新的空白图像 (使用 'RGB' 和 'white' 背景)
        combined_img = Image.new('RGB', (total_width, max_height), 'white')

        # 粘贴图像 (A 在左，B 在右)
        combined_img.paste(img_a, (0, 0))
        combined_img.paste(img_b, (width_a, 0))  # B 粘贴在 A 的右侧

        # 保存合并后的图像
        save_path_combined = os.path.join(OUTPUT_DIR, f"{filename_base}_AB_combined.png")
        combined_img.save(save_path_combined)
        print(f"     - 图 2 (A+B 合并) 已保存至: {save_path_combined}")

        # --- (新) 步骤 4: 清理临时文件 ---
        try:
            os.remove(save_path_a)
            os.remove(save_path_b)
            # print("     - 已清理临时文件 A 和 B。") # 调试用
        except Exception as e_clean:
            print(f"     - 警告: 清理临时 A/B 图像失败: {e_clean}")

    except Exception as e:
        print(f"[错误] 绘制图 2 失败: {e}")


def plot_figure_3(shap_values, X_test, filename="figure_3_shap_beeswarm_bar.png"):
    """
    (新) 图 3：严格按照用户提供的参考代码逻辑重写
    """
    if not SHAP_AVAILABLE:
        print("     - 警告: SHAP 库不可用, 跳过图 3。")
        return
    if shap_values is None or X_test is None:
        print("     - 警告: SHAP 数据无效, 跳过图 3。")
        return

    try:
        # 1. 找到动态特征并过滤 SHAP 数据
        all_features = X_test.columns.tolist()
        dynamic_features = [col for col in all_features if col not in ['City', 'Year']]
        dynamic_indices = [all_features.index(col) for col in dynamic_features]

        shap_values_dynamic = shap_values[:, dynamic_indices]
        X_test_dynamic = X_test[dynamic_features].copy()
        dynamic_feature_names = [FEATURE_NAME_MAP.get(name, name) for name in X_test_dynamic.columns]

        # 2. 转换类别为 .cat.codes
        cat_cols = X_test_dynamic.select_dtypes(include=['category']).columns
        if not cat_cols.empty:
            for col in cat_cols:
                X_test_dynamic[col] = X_test_dynamic[col].cat.codes

        # 3. 创建 shap.Explanation 对象 (用于 Beeswarm)
        explanation_dynamic = shap.Explanation(
            values=shap_values_dynamic,
            data=X_test_dynamic.values,
            feature_names=dynamic_feature_names
        )

        # --- (新) 严格按照参考代码的逻辑 ---
        fig = plt.figure(figsize=(16, 8))
        # (新) facecolor='white' 已由 RCParams 全局设置

        # --- 子图 A: SHAP Beeswarm Summary Plot ---
        ax1 = plt.subplot(1, 2, 1)
        # (新) RCParams 负责背景色

        # (新) 不传递 ax=, 让 SHAP 自动绘制到 ax1
        shap.plots.beeswarm(
            explanation_dynamic,
            max_display=len(dynamic_feature_names),
            show=False
        )

        ax1.set_title('A. SHAP Beeswarm Summary Plot', fontweight='bold', pad=15, color=COLORS['neutral2'])
        # (新) 修复 X 轴标签
        if not ax1.get_xlabel():
            ax1.set_xlabel("SHAP Value (Impact on model output [log-scale])", color=COLORS['neutral2'])
        else:
            ax1.set_xlabel(ax1.get_xlabel(), color=COLORS['neutral2'])

        # --- 子图 B: SHAP Global Importance Bar Plot (手动绘制) ---
        ax2 = plt.subplot(1, 2, 2)
        # (新) RCParams 负责背景色

        # (新) 计算平均绝对 SHAP 值
        mean_abs_shap = np.abs(shap_values_dynamic).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': dynamic_feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=True)  # 升序

        # (新) 创建水平条形图
        bars = ax2.barh(range(len(feature_importance_df)),
                        feature_importance_df['Importance'],
                        color=COLORS['primary'], alpha=0.8,
                        edgecolor=COLORS['neutral2'], linewidth=0.8)

        # (新) 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, feature_importance_df['Importance'])):
            ax2.text(importance + max(feature_importance_df['Importance']) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{importance:.3f}',
                     va='center', ha='left', fontsize=10, color=COLORS['neutral2'])

        # (新) 手动设置 Y 轴刻度和标签
        ax2.set_yticks(range(len(feature_importance_df)))
        ax2.set_yticklabels(feature_importance_df['Feature'], color=COLORS['neutral2'])

        ax2.set_xlabel('Mean |SHAP Value|', color=COLORS['neutral2'])
        ax2.set_title('B. Global Feature Importance', fontweight='bold', pad=15, color=COLORS['neutral2'])
        ax2.grid(axis='x', alpha=0.3)  # 恢复网格

        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(save_path)
        fig.savefig(save_path.replace('.png', '.pdf'))
        print(f"     - 图 3 已保存至: {save_path}")
        plt.close(fig)

    except Exception as e:
        print(f"[错误] 绘制图 3 失败: {e}")
        import traceback
        traceback.print_exc()


def plot_figure_4(shap_values, X_test, filename="figure_4_shap_dependence.png"):
    """
    图 4：趋近线为实线，说明在标题
    """
    if not SHAP_AVAILABLE:
        print("     - 警告: SHAP 库不可用, 跳过图 4。")
        return
    if shap_values is None or X_test is None:
        print("     - 警告: SHAP 数据无效, 跳过图 4。")
        return

    # --- 1. 计算 SHAP 特征重要性 (mean|SHAP|) ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': mean_abs_shap
    })

    # --- 2. 筛选 Top 4 动态特征 ---
    dynamic_features = [col for col in X_test.columns if col not in ['City', 'Year']]
    top_4_dynamic = shap_importance[shap_importance['feature'].isin(dynamic_features)] \
        .sort_values(by='importance', ascending=False) \
        .head(4)['feature'].tolist()

    if len(top_4_dynamic) == 0:
        print("     - 警告: 未找到动态特征, 跳过图 4。")
        return

    # --- 3. 绘制 2x2 面板 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axs_flat = axes.flatten()

    for i, feature in enumerate(top_4_dynamic):
        ax = axs_flat[i]
        feature_display_name = FEATURE_NAME_MAP.get(feature, feature)

        # --- (新) 手动重绘 SHAP 依赖图 ---
        feat_idx = X_test.columns.get_loc(feature)
        shap_values_feat = shap_values[:, feat_idx]
        X_unscaled_feat = X_test[feature]

        # --- (新) 寻找最佳交互特征 (借鉴自您上传的文件) ---
        correlations = []
        for j in range(shap_values.shape[1]):
            if j != feat_idx:
                try:
                    # 计算 SHAP 值之间的相关性
                    corr = np.corrcoef(shap_values[:, j], shap_values_feat)[0, 1]
                    if not np.isnan(corr):
                        correlations.append((j, corr))
                except:
                    continue

        if correlations:
            interaction_idx = max(correlations, key=lambda x: x[1])[0]
            interaction_feat_name = X_test.columns[interaction_idx]
            interaction_values_unscaled = X_test[interaction_feat_name]
            interaction_display_name = FEATURE_NAME_MAP.get(interaction_feat_name, interaction_feat_name)

            # 绘制散点图 (带交互着色)
            scatter = ax.scatter(X_unscaled_feat, shap_values_feat,
                                 c=interaction_values_unscaled, cmap='viridis',
                                 alpha=0.7, s=35, edgecolor='white', linewidth=0.5, zorder=5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(interaction_display_name, rotation=270, labelpad=15)
        else:
            # 绘制散点图 (无交互)
            ax.scatter(X_unscaled_feat, shap_values_feat,
                       alpha=0.7, s=35, color=COLORS['primary'],
                       edgecolor='white', linewidth=0.5, zorder=5)

        # --- (新) 添加平滑趋近线 (借鉴自您上传的文件) ---
        try:
            sort_indices = np.argsort(X_unscaled_feat)
            x_sorted = np.array(X_unscaled_feat.iloc[sort_indices])
            y_sorted = np.array(shap_values_feat[sort_indices])

            if len(x_sorted) > 10:
                x_uniform = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                y_interp = np.interp(x_uniform, x_sorted, y_sorted)
                sigma = max(1, len(y_interp) / 50)
                y_smooth = gaussian_filter1d(y_interp, sigma=sigma)

                # (修改 4) 绘制趋近线 (实线)
                ax.plot(x_uniform, y_smooth, color=COLORS['secondary'],  # 使用醒目的次色
                        alpha=0.9, linewidth=3, label='Smooth Trend',
                        linestyle='-', zorder=10)  # <-- 改为实线
        except Exception as e_smooth:
            print(f"    - 警告: 无法为 {feature} 绘制趋近线: {e_smooth}")
        # --- 手动重绘结束 ---

        ax.set_xlabel(f"Feature Value ({feature_display_name})")
        ax.set_ylabel(f"SHAP Value (for {feature_display_name})")
        # (修改 4) 添加子图标题
        ax.set_title(f'{chr(65 + i)}. Dependence on {feature_display_name}', loc='left', fontsize=14, fontweight='bold')

    for i in range(len(top_4_dynamic), 4):
        fig.delaxes(axs_flat[i])

    plt.tight_layout(pad=1.5)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)  # RCParams 负责 dpi 和 bbox_inches
    print(f"     - 图 4 已保存至: {save_path}")
    plt.close(fig)


def plot_figure_5(df_oof, filename="figure_5_prediction_fidelity.png"):

    if df_oof.empty:
        print("     - 警告: OOF 数据为空, 跳过图 5。")
        return

    # --- 1. 计算 *整体* 指标 ---
    r2_val = r2_score(df_oof['True_Values'], df_oof['Predicted_Values'])
    rmse_val = np.sqrt(mean_squared_error(df_oof['True_Values'], df_oof['Predicted_Values']))
    mae_val = mean_absolute_error(df_oof['True_Values'], df_oof['Predicted_Values'])

    text_content_overall = (
        f"Overall Metrics\n"
        f"R^2 = {r2_val:.4f}\n"
        f"RMSE = {rmse_val:.4f}\n"
        f"MAE = {mae_val:.4f}\n"
        f"(N = {len(df_oof)})"
    )

    # --- (新) 2. 计算 *按年份* 分解的指标 ---
    yearly_metrics_text = ["\nYearly Breakdown (2018-2022)"]

    # 确保 'Test_Year' 存在
    if 'Test_Year' not in df_oof.columns:
        yearly_metrics_text.append(" ('Test_Year' column not found in OOF data)")
    else:
        # 确保年份是整数，以便正确排序和匹配
        try:
            df_oof['Test_Year'] = df_oof['Test_Year'].astype(int)
        except ValueError:
            yearly_metrics_text.append(" (Could not convert 'Test_Year' to int)")

        target_years = [2018, 2019, 2020, 2021, 2022]

        # 检查 OOF 数据中实际存在的年份
        available_years = sorted([y for y in target_years if y in df_oof['Test_Year'].unique()])

        if not available_years:
            yearly_metrics_text.append(" (No 2018-2022 data found in 'Test_Year')")

        for year in available_years:
            df_year = df_oof[df_oof['Test_Year'] == year]

            if len(df_year) > 0:
                y_true = df_year['True_Values']
                y_pred = df_year['Predicted_Values']

                r2_y = r2_score(y_true, y_pred)
                rmse_y = np.sqrt(mean_squared_error(y_true, y_pred))
                mae_y = mean_absolute_error(y_true, y_pred)

                # (新) 格式化为更紧凑的单行，以节省空间
                yearly_metrics_text.append(
                    f"{year}: R^2={r2_y:.3f} | RMSE={rmse_y:.3f} | MAE={mae_y:.3f}"
                )
            else:
                yearly_metrics_text.append(f"{year}: (No data)")

    # --- (新) 3. 合并所有文本 ---
    text_content = text_content_overall + "\n" + "\n".join(yearly_metrics_text)

    # --- 4. 绘制 Hexbin 密度图 (使用 Seaborn JointGrid) ---
    g = sns.jointplot(
        data=df_oof,
        x='True_Values',
        y='Predicted_Values',
        kind='hex',
        cmap='viridis',
        height=8,
        joint_kws={'gridsize': 30, 'edgecolor': 'none'},
        marginal_kws={'bins': 40}
    )

    g.ax_joint.set_xlabel("True Values (log-scale)")
    g.ax_joint.set_ylabel("Predicted Values (log-scale)")

    # --- 5. 添加辅助线 ---
    min_val = min(df_oof['True_Values'].min(), df_oof['Predicted_Values'].min())
    max_val = max(df_oof['True_Values'].max(), df_oof['Predicted_Values'].max())
    axis_lim = [min_val * 0.95, max_val * 1.05]

    model_fit = LinearRegression().fit(df_oof[['True_Values']], df_oof['Predicted_Values'])
    x_fit = np.array(axis_lim).reshape(-1, 1)
    y_fit = model_fit.predict(x_fit)
    # (修改 5) 恢复 V5 的 OLS 颜色
    g.ax_joint.plot(x_fit, y_fit, '#0072B2', linewidth=2, label='OLS Fit')

    g.ax_joint.set_xlim(axis_lim)
    g.ax_joint.set_ylim(axis_lim)

    # --- 6. 添加(更新后的)文本框和图例 ---
    g.ax_joint.text(
        0.05, 0.95, text_content,
        transform=g.ax_joint.transAxes,
        verticalalignment='top',
        # (修改 5) 恢复 V5 的文本框背景
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
        fontsize=9  # (新) 缩小字体以适应更多内容
    )
    g.ax_joint.legend(loc='lower right')

    plt.tight_layout()

    # (新) 调整布局以适应总标题
    try:
        g.fig.subplots_adjust(top=0.95)
    except:
        pass  # 避免在某些后端出错

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)  # RCParams 负责 dpi 和 bbox_inches
    print(f"     - 图 5 (含年份分解) 已保存至: {save_path}")

    # (新) 确保关闭 jointplot 的 figure
    plt.close(g.fig)


def plot_figure_6(df_oof, filename="figure_6_temporal_stability.png"):
    """
    (修改 6) 图 6：上移图例
    """
    if df_oof.empty or 'Fold' not in df_oof.columns or 'Test_Year' not in df_oof.columns:
        print("     - 警告: OOF 数据不包含 Fold/Test_Year 信息, 跳过图 6。")
        return

    # --- 1. 按 Fold 分组计算指标 ---
    fold_metrics = []
    for fold, group in df_oof.groupby('Fold'):
        test_year = int(group['Test_Year'].iloc[0])
        r2 = r2_score(group['True_Values'], group['Predicted_Values'])
        rmse = np.sqrt(mean_squared_error(group['True_Values'], group['Predicted_Values']))
        mae = mean_absolute_error(group['True_Values'], group['Predicted_Values'])
        fold_metrics.append({
            'Fold': fold,
            'Test_Year': test_year,
            'Fold_Label': f"Fold {fold + 1}\n(Test {test_year})",
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })
    fold_metrics = pd.DataFrame(fold_metrics).sort_values(by='Test_Year')

    # --- 2. 计算平均指标 ---
    avg_r2 = df_oof.attrs.get('Overall_R2', fold_metrics['R2'].mean())
    avg_rmse = df_oof.attrs.get('Overall_RMSE', fold_metrics['RMSE'].mean())
    avg_mae = df_oof.attrs.get('Overall_MAE', fold_metrics['MAE'].mean())

    # --- 3. 绘制组合图 (Bar + Line) ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- R2 (左轴 - 条形图) ---
    ax1.bar(
        fold_metrics['Fold_Label'],
        fold_metrics['R2'],
        color=color_map_featureset['M2_Full_L1'],
        label='$R^2$'
    )
    ax1.axhline(
        avg_r2,
        color=color_map_featureset['M2_Full_L1'],
        linestyle='--',
        label=f"Average $R^2$ ({avg_r2:.4f})"
    )

    ax1.set_ylabel("$R^2$ (log-scale)")
    ax1.set_ylim(min(0.8, fold_metrics['R2'].min() * 0.95), 1.0)
    ax1.set_xlabel("Outer CV Fold (Test Year)")

    # --- RMSE & MAE (右轴 - 线图) ---
    ax2 = ax1.twinx()

    ax2.plot(
        fold_metrics['Fold_Label'],
        fold_metrics['RMSE'],
        color=color_map_algorithm['EBM'],
        marker='o',
        markersize=8,
        linestyle='-',
        linewidth=2,
        label='RMSE (Right Axis)'
    )
    ax2.plot(
        fold_metrics['Fold_Label'],
        fold_metrics['MAE'],
        color=color_map_algorithm['DNN'],
        marker='s',
        markersize=8,
        linestyle='-',
        linewidth=2,
        label='MAE (Right Axis)'
    )

    ax2.axhline(
        avg_rmse,
        color=color_map_algorithm['EBM'],
        linestyle='--',
        label=f"Average RMSE ({avg_rmse:.4f})"
    )
    ax2.axhline(
        avg_mae,
        color=color_map_algorithm['DNN'],
        linestyle='--',
        label=f"Average MAE ({avg_mae:.4f})"
    )

    ax2.set_ylabel("Error (log-scale)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # (修改 6) 上移图例，防止重叠
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.0, 1.15))

    plt.tight_layout(pad=1.0)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)  # RCParams 负责 dpi 和 bbox_inches
    print(f"     - 图 6 已保存至: {save_path}")
    plt.close(fig)


# --- 2. 数据加载函数 (无需修改) ---

def load_data_fig1():
    try:
        df = pd.read_csv(MODEL_EVAL_FILE)
        return df
    except FileNotFoundError:
        print(f"[错误] 图 1 数据文件未找到: {MODEL_EVAL_FILE}")
        return pd.DataFrame()


def load_data_fig2():
    if not OPTUNA_AVAILABLE:
        return None
    try:
        with open(OPTUNA_STUDY_FILE, 'rb') as f:
            study = pickle.load(f)
        return study
    except FileNotFoundError:
        print(f"[错误] 图 2 数据文件未找到: {OPTUNA_STUDY_FILE}")
        return None


def load_data_fig3_4():
    if not SHAP_AVAILABLE:
        return None, None
    try:
        with open(SHAP_VALUES_FILE, 'rb') as f:
            shap_values = pickle.load(f)
        X_test = pd.read_pickle(SHAP_X_TEST_FILE)
        return shap_values, X_test
    except FileNotFoundError:
        print(f"[错误] 图 3/4 SHAP 文件未找到 (SHAP or X_test)。")
        return None, None
    except Exception as e:
        print(f"[错误] 加载 SHAP 数据时出错: {e}")
        return None, None


def load_data_fig5_6():
    try:
        df = pd.read_pickle(OOF_PRED_FILE)
        return df
    except FileNotFoundError:
        print(f"[错误] 图 5/6 OOF 数据文件未找到: {OOF_PRED_FILE}")
        return pd.DataFrame()


# --- 3. 主执行函数 (无需修改) ---

def main():
    print("=" * 50)
    print("--- 开始执行 V5 可视化流程 (加载真实数据) ---")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 图 1 ---
    print("... (Real Data) 正在加载图1数据 (V5_Final_model_evaluation.csv)...")
    df_results = load_data_fig1()
    print("--- 正在绘制图 1 (保持 V5 布局)...")
    plot_figure_1(df_results)

    # --- 图 2 ---
    print("... (Real Data) 正在加载图2数据 (Optuna Study)...")
    study = load_data_fig2()
    print("--- 正在绘制图 2 (3 张独立图)...")
    plot_figure_2(study)

    # --- 图 3 & 4 ---
    print("... (Real Data) 正在加载图3/4数据 (SHAP 值)...")
    shap_values, X_test = load_data_fig3_4()

    print("--- 正在绘制图 3 (参考代码逻辑)...")
    plot_figure_3(shap_values, X_test)


    # --- 图 5 & 6 ---
    print("... (Real Data) 正在加载图5/6数据 (OOF 预测)...")
    df_oof = load_data_fig5_6()

    print("--- 正在绘制图 5 (V5 样式)...")
    plot_figure_5(df_oof)

    print("--- 正在绘制图 6 (图例位置修复)...")
    plot_figure_6(df_oof)

    print("\n" + "=" * 50)
    print("--- V5 可视化流程执行完毕 ---")
    print(f"--- 图像已保存至: ./{OUTPUT_DIR}/ ---")
    print("=" * 50)


if __name__ == "__main__":
    main()