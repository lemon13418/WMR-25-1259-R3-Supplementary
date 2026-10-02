import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler  # <--- V-LPC-Robust: 仍然使用 RobustScaler
from sklearn.decomposition import PCA  # <--- (V-PCA) 审稿人建议: 导入 PCA
import seaborn as sns
import matplotlib.pyplot as plt
# from data_processing import merge_all_data # 请确保您已替换此函数
import warnings
import joblib
import re

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 模拟的 merge_all_data ---
def merge_all_data():
    """
    占位函数：请确保您有原始的 merge_all_data 函数或 pd.read_csv(...)
    """

    try:
        # 假设原始数据包含:
        # ..., Policy_count (小写), ...
        df = pd.read_csv("merged_raw_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("请替换 merge_all_data() 函数，或将您的原始数据命名为 'all_city_data_raw.csv'")
    return df


# -----------------------------


# --- 核心函数 ---

def handle_missing_values(df):
    """
    处理缺失值
    (V-LPC-Robust) 策略: 目标变量为人均值 (I/P)，并进行 Log 变换
    """
    print("处理缺失值 (V-LPC: Log-Per-Capita 策略)...")  # <--- (V-PCA) 保留 V-LPC 逻辑
    if 'Target' not in df.columns:
        raise ValueError("错误：数据中未找到 'Target' (固废回收总量) 列。")
    if '城市' not in df.columns:
        raise ValueError("错误：数据中未找到 '城市' 列。")
    if 'Total_population' not in df.columns:
        raise ValueError("错误：数据中未找到 'Total_population' 列。")

    df.rename(columns={'城市': 'City'}, inplace=True)

    # (V-LPC) 步骤 1: 优先填充 Total_population
    pop_col = 'Total_population'
    df[pop_col] = df.groupby('City')[pop_col].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    df[pop_col] = df.groupby('City')[pop_col].transform(
        lambda x: x.fillna(x.mean())
    )
    if df[pop_col].isnull().any():
        global_mean_pop = df[pop_col].mean()
        df[pop_col].fillna(global_mean_pop, inplace=True)
        print(f"  - 警告: '{pop_col}' 在分组填充后仍有缺失，已使用全局均值填充。")

    # (V-LPC) 步骤 2: 转换 Target 为 Per-Capita (人均)
    df = df.dropna(subset=['Target'])  # 移除原始 Target 为空的行
    df['Target_per_capita'] = df['Target'] / (df[pop_col] + 1e-6)  # 防止除以0
    df = df.drop(columns=['Target'])
    df = df.rename(columns={'Target_per_capita': 'Target'})
    print(f"  - (V-LPC) 'Target' (固废回收总量) 已转换为 'Target' (人均)。")

    # (V-LPC) 步骤 3: 对 *人均* Target 应用 log1p 变换
    df['Target'] = np.log1p(df['Target'])
    print(f"  - 已对 'Target' (人均) 列应用 log1p 变换。")

    # (V-LPC) 步骤 4: 填充所有 *其他* 数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(
        ['Year', 'Target', 'Total_population'], errors='ignore'
    )

    for col in numeric_columns:
        df[col] = df.groupby('City')[col].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        df[col] = df.groupby('City')[col].transform(
            lambda x: x.fillna(x.mean())
        )

    for col in numeric_columns:
        if df[col].isnull().any():
            global_mean = df[col].mean()
            df[col].fillna(global_mean, inplace=True)
            print(f"  - 警告: 特征 '{col}' 在分组填充后仍有缺失，已使用全局均值填充。")

    print("  - 缺失值处理完成。")
    return df


def create_derived_features(df):
    """
    (受STIRPAT启发)
    构建 10 个核心驱动因素特征。
    (V-EKC-Removed) <--- (V-PCA) 保留 V-EKC-Removed 逻辑
    """
    print("\n创建衍生特征 (受 STIRPAT 框架启发 - 移除 EKC 项)...")

    # --- 检查所需原始列 ---
    required_raw_cols = [
        'GDP', 'Total_population', 'Secondary_industry', 'Tertiary_industry',
        'Fiscal_revenue', 'Gross_construction_output', 'Urban_population',
        'Patent_grants',
        'Policy_count',  # <--- 确保检查小写的 'Policy_count'
        'Public_awareness', 'Total_electricity_consumption'
    ]
    missing_cols = [col for col in required_raw_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"错误: 原始数据中缺少以下所需列: {missing_cols}")

    # --- 3.1. (A) 经济驱动因素 (Economic Drivers) ---
    print("  - 正在构建 (A) 经济驱动因素 (Economic Drivers)...")
    df['PerCapitaGDP'] = df['GDP'] / (df['Total_population'] + 1e-6)
    df['SecondaryIndustryShare'] = df['Secondary_industry'] / (df['GDP'] + 1e-6)
    df['TertiaryIndustryShare'] = df['Tertiary_industry'] / (df['GDP'] + 1e-6)
    df['FiscalCapacityPerCapita'] = df['Fiscal_revenue'] / (df['Total_population'] + 1e-6)
    df['ConstructionIntensity'] = df['Gross_construction_output'] / (df['GDP'] + 1e-6)
    # --- (V-EKC-Removed) 移除:
    # df['PerCapitaGDP_squared'] = df['PerCapitaGDP']**2

    # --- 3.2. (P_struct) 人口结构因素 (Population Structure) ---
    print("  - 正在构建 (P_struct) 人口结构因素 (Population Structure)...")
    df['UrbanizationRate'] = df['Urban_population'] / (df['Total_population'] + 1e-6)
    df['UrbanizationRate'] = df['UrbanizationRate'].clip(0, 1)

    # --- 3.3. (T_ext) 社会-技术与制度因素 (Socio-technical & Institutional) ---
    print("  - 正在构建 (T_ext) 社会-技术与制度因素...")
    df['PatentsPerCapita'] = df['Patent_grants'] / (df['Total_population'] + 1e-6)
    if 'Policy_count' in df.columns:
        df.rename(columns={'Policy_count': 'PolicyCount'}, inplace=True)
        print("  - (修复) 已将 'Policy_count' 重命名为 'PolicyCount'。")
    elif 'PolicyCount' not in df.columns:
        raise ValueError("错误: 原始数据中既未找到 'Policy_count' 也未找到 'PolicyCount'。")
    df['PublicAwarenessIndex'] = df['Public_awareness'] / (df['Urban_population'] + 1e-6)
    df['EnergyIntensity'] = df['Total_electricity_consumption'] / (df['GDP'] + 1e-6)

    # --- 清理无限值 ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 再次处理因除法产生的缺失值
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            global_mean = df[col].mean()
            if pd.isna(global_mean):
                global_mean = 0
            df[col].fillna(global_mean, inplace=True)

    print(f"衍生特征创建完成，当前特征数: {len(df.columns)}")
    return df


def add_contextual_features(df):
    """
    移除了 COVID 虚拟变量的创建
    """
    print("\n创建上下文特征 (最终清理版)...")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    print("  - COVID 虚拟变量 (is_2020/21/22) 已被永久移除。")
    print("  - 'City' 列已保留。")
    return df


def prepare_final_dataset(df, final_core_features):
    """
    (V-LPC-PCA) <--- 修改点
    步骤 1: (V-LPC-Robust) 使用 RobustScaler (Scale 策略)。
    步骤 2: (V-PCA) 对缩放后的特征应用 PCA 降维。
    """
    print("\n准备最终数据集 (V-LPC-PCA 策略)...")  # <--- 修改点

    if "City" not in df.columns:
        raise ValueError("错误: 'City' 列在传入 prepare_final_dataset 之前丢失。")
    if "Total_population" not in df.columns:
        raise ValueError(
            "错误: 'Total_population' 列在传入 prepare_final_dataset 之前丢失。请检查 handle_missing_values。")

    # M1 特征 (基线)
    M1_FEATURES = ['City', 'Year', 'Target', 'Total_population']
    # M2 特征 (10个动态特征)
    M2_DYNAMIC_FEATURES = final_core_features

    # 最终列：M1 + M2
    final_cols = M1_FEATURES + M2_DYNAMIC_FEATURES
    final_cols = list(dict.fromkeys(final_cols))  # 保持顺序并去重

    missing_in_df = [col for col in M2_DYNAMIC_FEATURES if col not in df.columns]
    if missing_in_df:
        raise ValueError(f"错误: 以下核心特征在 df 中不存在: {missing_in_df}。")

    final_df = df[final_cols].copy()

    final_df['City'] = final_df['City'].astype('category')
    print("  - 'City' 列已保留并设置为 category 类型。")
    print("  - 'Total_population' 列已保留用于评估。")

    features_to_scale = M2_DYNAMIC_FEATURES

    # --- (V-PCA) 步骤 1: (V-LPC-Robust) 鲁棒缩放 ---
    scaler = RobustScaler()
    print("  - (V-PCA 步骤 1) 已切换到 RobustScaler (鲁棒缩放策略)。")

    final_df_scaled = final_df.copy()

    train_data_mask = final_df_scaled['Year'] <= 2019
    if train_data_mask.sum() == 0:
        print("  - 警告: 训练集 (<= 2019) 为空。缩放器 (Scaler) 将在所有数据上训练。")
        train_data_mask = slice(None)

    if features_to_scale:
        # 确保只缩放数值型
        numeric_features_to_scale = final_df.loc[train_data_mask, features_to_scale].select_dtypes(
            include=np.number).columns

        if list(numeric_features_to_scale) != features_to_scale:
            print(f"  - 警告: 以下非数值特征将不会被缩放: {set(features_to_scale) - set(numeric_features_to_scale)}")
            features_to_scale = list(numeric_features_to_scale)

        if 'PolicyCount' in features_to_scale and not pd.api.types.is_numeric_dtype(final_df['PolicyCount']):
            print("  - 'PolicyCount' 不是数值型, 将从缩放中移除。")
            features_to_scale.remove('PolicyCount')

        # 确保 M2 特征列表在缩放后保持一致
        M2_DYNAMIC_FEATURES = features_to_scale

        if M2_DYNAMIC_FEATURES:
            scaler.fit(final_df.loc[train_data_mask, M2_DYNAMIC_FEATURES])
            final_df_scaled[M2_DYNAMIC_FEATURES] = scaler.transform(final_df[M2_DYNAMIC_FEATURES])
        else:
            print("  - 没有找到可缩放的数值特征。")

    scaler_filename = 'scaler_lpc_robust_final.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"  - 鲁棒缩放器 (RobustScaler) 已保存至: {scaler_filename}")

    # (V-PCA) 保存 V4/V5-M1 使用的原始缩放数据集 (M1+M2)
    output_csv_name = 'final_dataset_lpc_robust_scaled.csv'
    final_df_scaled.to_csv(output_csv_name, index=False, encoding='utf-8-sig')
    print(f"  - 原始缩放数据集 (M1+M2, V4使用) 形状: {final_df_scaled.shape}")
    print(f"  - 原始缩放数据集已保存至: {output_csv_name}")

    with open('selected_features.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(M2_DYNAMIC_FEATURES))
    print(f"  - 核心特征列表 (M2) 已保存至: selected_features.txt")

    # --- (V-PCA) 步骤 2: (审稿人建议) PCA 降维 ---
    print("\n  - (V-PCA 步骤 2) 正在对缩放后的 10 个动态特征 (M2) 执行 PCA 降维...")

    if not M2_DYNAMIC_FEATURES:
        print("  - 错误: M2 动态特征列表为空，无法执行 PCA。")
        return final_df_scaled, scaler

    X_train_dynamic_scaled = final_df_scaled.loc[train_data_mask, M2_DYNAMIC_FEATURES]

    # 审稿人建议 n_components=2-3。我们使用 0.95 (保留95%方差) 来自动确定
    pca = PCA(n_components=0.95, random_state=SEED)

    # 在训练集上拟合 PCA
    pca.fit(X_train_dynamic_scaled)

    n_pcs = pca.n_components_
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"  - (V-PCA) PCA 拟合完成。")
    print(f"  - (V-PCA) 自动选择 {n_pcs} 个主成分 (PC), 共解释 {explained_variance:.2f}% 的方差。")

    # 保存 PCA 模型
    pca_filename = 'pca_model_on_scaled_features.pkl'
    joblib.dump(pca, pca_filename)
    print(f"  - (V-PCA) PCA 模型已保存至: {pca_filename}")

    # 转换完整数据集 (训练集+测试集)
    X_full_dynamic_scaled = final_df_scaled[M2_DYNAMIC_FEATURES]
    X_full_pca = pca.transform(X_full_dynamic_scaled)

    # 创建 PCA 特征的 DataFrame
    pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]
    df_pca_features = pd.DataFrame(X_full_pca, columns=pc_cols, index=final_df_scaled.index)

    # 创建新的 M2_PCA 数据集 (M1 特征 + PCA 特征)
    df_m2_pca = pd.concat([final_df_scaled[M1_FEATURES], df_pca_features], axis=1)

    # 保存 M2_PCA 数据集
    output_pca_csv_name = 'final_dataset_M2_PCA.csv'
    df_m2_pca.to_csv(output_pca_csv_name, index=False, encoding='utf-8-sig')
    print(f"  - (V-PCA) M2 (PCA 降维后) 数据集形状: {df_m2_pca.shape}")
    print(f"  - (V-PCA) M2 (PCA) 数据集已保存至: {output_pca_csv_name}")

    return final_df_scaled, scaler


def main():
    """
    主函数 (V-LPC-PCA)
    """
    print("=" * 70)
    print("特征工程开始 (V-LPC-PCA 策略)")  # <--- 修改点
    print("=" * 70)

    global SEED
    SEED = 42  # 确保 PCA 随机状态一致

    # --- 步骤 1: 准备基础数据 ---
    df_raw = merge_all_data()
    df = handle_missing_values(df_raw)
    df = create_derived_features(df)
    df = add_contextual_features(df)

    # --- (V-EKC-Removed / 理论框架修正) ---
    # 核心驱动因素 (Core Drivers) - 受 STIRPAT 框架启发
    # 这 10 个特征是 VIF > 70 的来源，将被 PCA 处理
    final_core_features = [
        # (A) 经济驱动因素 (Economic Drivers) - 对应 STIRPAT 的 "Affluence"
        'PerCapitaGDP',  # A1: 经济发展水平
        'SecondaryIndustryShare',  # A2: 经济结构 (二产)
        'TertiaryIndustryShare',  # A3: 经济结构 (三产)
        'FiscalCapacityPerCapita',  # A4: 政府财政能力
        'ConstructionIntensity',  # A5: 建筑业活动强度
        # 'PerCapitaGDP_squared',     # (V-EKC-Removed) A-EKC: 已移除

        # (P_struct) 人口结构因素 (Population Structure)
        'UrbanizationRate',  # P_struct: 城镇化水平 (人口特征)

        # (T_ext) 社会-技术与制度因素 (Socio-technical & Institutional)
        'PatentsPerCapita',  # T1: 技术创新水平 (技术)
        'EnergyIntensity',  # T2: 能源消耗效率 (技术效率)
        'PolicyCount',  # T3: 政策驱动强度 (制度因素)
        'PublicAwarenessIndex',  # T4: 公众环保意识 (社会因素)
    ]
    # --- (修改结束) ---

    print(f"\n已定义 {len(final_core_features)} 个核心驱动因素 (M2 特征)。")
    print(f"  - (V-PCA) 这 {len(final_core_features)} 个特征将被 RobustScaler 缩放，然后由 PCA 进行降维。")

    # --- 步骤 3: 准备并保存最终数据集 (缩放与 PCA) ---
    prepare_final_dataset(df, final_core_features)

    print("\n" + "=" * 70)
    print("特征工程（V-LPC-PCA 版）完成！")  # <--- 修改点
    print("=" * 70)

    # --- (V-PCA) 新增步骤: 分析 PCA 载荷 (响应用户提问) ---
    print("\n" + "=" * 70)
    print("(V-PCA) 步骤 3: 分析主成分 (PC) 载荷")
    print("=" * 70)
    try:
        # 加载刚刚保存的 PCA 模型和特征列表
        pca: PCA = joblib.load('pca_model_on_scaled_features.pkl')

        with open('selected_features.txt', 'r', encoding='utf-8') as f:
            original_features = [line.strip() for line in f.readlines()]

        if not original_features:
            print("  - 错误: 'selected_features.txt' 为空或未找到。")
            return

        print(f"  - 成功加载 PCA 模型和 {len(original_features)} 个原始特征。")

        # pca.components_ 存储了载荷矩阵
        # 形状是 (n_components, n_features)
        loadings = pca.components_

        # 创建 DataFrame 以便查看
        pc_names = [f'PC{i + 1}' for i in range(pca.n_components_)]
        loadings_df = pd.DataFrame(loadings, columns=original_features, index=pc_names)

        # .T (转置) 使其更易于阅读: 每列是一个 PC，每行是一个原始特征
        loadings_df_transposed = loadings_df.T

        print("\n  --- PCA 载荷矩阵 (转置) ---")
        print("  - (含义: 每个 PC 由哪些原始特征构成)")
        print(loadings_df_transposed)

        # 保存为 CSV 以便在 Excel 中详细分析
        loadings_csv_path = 'pca_loadings_matrix.csv'
        loadings_df_transposed.to_csv(loadings_csv_path, encoding='utf-8-sig')
        print(f"\n  - 完整的 PCA 载荷矩阵已保存至: {loadings_csv_path}")

        print("\n  --- 如何解读此表: ---")
        print("  - 1. 查看 'PC1' 列。")
        print("  - 2. 找到绝对值*最大*的几个特征 (例如 'PerCapitaGDP' 可能是 0.4, 'EnergyIntensity' 可能是 -0.3)。")
        print("  - 3. 这意味着 'PC1' 主要代表了 'PerCapitaGDP' (正相关) 和 'EnergyIntensity' (负相关) 的组合。")
        print("  - 4. 您可以据此为 'PC1' 命名, 例如 '经济发展-能源效率因子'。")
        print("  - 5. 在模型训练后, 如果 SHAP 显示 'PC1' 很重要, 您就可以*推断*是 '经济发展-能源效率因子' 很重要。")

    except FileNotFoundError:
        print("  - 错误: 未能找到 'pca_model_on_scaled_features.pkl' 或 'selected_features.txt'。")
    except Exception as e:
        print(f"  - 分析 PCA 载荷时发生错误: {e}")


if __name__ == "__main__":
    main()

