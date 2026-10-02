"""
data_processing.py
数据处理模块 - 加载和整合所有数据源
"""

import pandas as pd
import numpy as np
import os
import re
import warnings

warnings.filterwarnings('ignore')


def extract_year(x):
    """从字符串中提取年份"""
    if pd.isna(x):
        return None
    year_match = re.search(r'(\d{4})', str(x))
    return int(year_match.group(1)) if year_match else None


def safe_to_numeric(series):
    """安全地将系列转换为数值类型"""
    return pd.to_numeric(series, errors='coerce')


def find_column_by_keywords(columns, keywords, default_index=None):
    """根据关键词查找列名"""
    for col in columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    if default_index is not None and default_index < len(columns):
        return columns[default_index]
    return None


def load_target_data(folder_path):
    """加载目标变量数据"""
    all_data = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_excel(file_path)
                    year_col = find_column_by_keywords(df.columns, ('year', '年份', '年度'), default_index=0)
                    target_col = find_column_by_keywords(df.columns,
                                                         ('固体废物综合利用量', '回收量', 'target'),
                                                         default_index=1)
                    if year_col and target_col:
                        out = pd.DataFrame()
                        out['Year'] = df[year_col].apply(extract_year)
                        out['Target'] = safe_to_numeric(df[target_col])
                        out['城市'] = city
                        out = out.dropna(subset=['Year', 'Target'])
                        all_data.append(out)
                except Exception as e:
                    print(f"[目标变量] 无法读取文件 {file}，跳过。原因：{e}")

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"目标变量数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Target'])


def load_policy_data(policy_path):
    """加载政策文件数据（政策发布次数）"""
    policy_all = []
    if os.path.exists(policy_path):
        for file in os.listdir(policy_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(policy_path, file)
                try:
                    df = pd.read_excel(file_path)
                    if len(df.columns) >= 2:
                        out = pd.DataFrame()
                        out['Year'] = df.iloc[:, 0].apply(extract_year)
                        out['Policy_count'] = safe_to_numeric(df.iloc[:, 1])
                        out['城市'] = city
                        out = out.dropna(subset=['Year', 'Policy_count'])
                        policy_all.append(out)
                except Exception as e:
                    print(f"[政策文件] 无法读取文件 {file}，跳过。原因：{e}")

    if policy_all:
        result_df = pd.concat(policy_all, ignore_index=True)
        print(f"政策文件数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Policy_count'])


def load_awareness_data(awareness_path):
    """加载公众意识数据（百度搜索指数）"""
    awareness_all = []
    if os.path.exists(awareness_path):
        for file in os.listdir(awareness_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(awareness_path, file)
                try:
                    df = pd.read_excel(file_path)
                    if len(df.columns) >= 2:
                        out = pd.DataFrame()
                        out['Year'] = df.iloc[:, 0].apply(extract_year)
                        out['Public_awareness'] = safe_to_numeric(df.iloc[:, 1])
                        out['城市'] = city
                        out = out.dropna(subset=['Year', 'Public_awareness'])
                        awareness_all.append(out)
                except Exception as e:
                    print(f"[公众意识] 无法读取文件 {file}，跳过。原因：{e}")

    if awareness_all:
        result_df = pd.concat(awareness_all, ignore_index=True)
        print(f"公众意识数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Public_awareness'])


def load_population_data(pop_path):
    """加载人口数据"""
    pop_all = []
    if os.path.exists(pop_path):
        for file in os.listdir(pop_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(pop_path, file)
                try:
                    df = pd.read_excel(file_path)
                    out = pd.DataFrame()

                    # 特殊处理上海（直辖市）
                    if city == '上海':
                        if len(df.columns) >= 2:
                            out['Year'] = df.iloc[:, 0].apply(extract_year)
                            out['Total_population'] = safe_to_numeric(df.iloc[:, 1])
                            out['Urban_population'] = safe_to_numeric(df.iloc[:, 1])  # 上海全部为城镇人口
                    else:
                        year_col = find_column_by_keywords(df.columns, ('year', '年份', '年度'), default_index=0)
                        total_col = find_column_by_keywords(df.columns, ('总人口', '人口总', '总计'), default_index=1)
                        urban_col = find_column_by_keywords(df.columns, ('城镇', '城镇人口'), default_index=2)

                        out['Year'] = df[year_col].apply(extract_year)
                        out['Total_population'] = safe_to_numeric(df[total_col]) if total_col else np.nan
                        out['Urban_population'] = safe_to_numeric(df[urban_col]) if urban_col else np.nan

                    out['城市'] = city
                    out = out.dropna(subset=['Year'])
                    pop_all.append(out)
                except Exception as e:
                    print(f"[人口] 无法读取文件 {file}，跳过。原因：{e}")

    if pop_all:
        result_df = pd.concat(pop_all, ignore_index=True)
        print(f"人口数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Total_population', 'Urban_population'])


def load_economics_data(econ_path):
    """加载经济社会变量数据"""
    econ_all = []
    if os.path.exists(econ_path):
        for file in os.listdir(econ_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(econ_path, file)
                try:
                    df = pd.read_excel(file_path)
                    year_col = find_column_by_keywords(df.columns, ('year', '年份', '年度'), default_index=0)
                    rev_col = find_column_by_keywords(df.columns, ('财政收入', '收入'), default_index=1)
                    gdp_col = find_column_by_keywords(df.columns, ('gdp', '地区生产总值', '生产总值'), default_index=2)

                    out = pd.DataFrame()
                    out['Year'] = df[year_col].apply(extract_year)
                    out['Fiscal_revenue'] = safe_to_numeric(df[rev_col]) if rev_col else np.nan
                    out['GDP'] = safe_to_numeric(df[gdp_col]) if gdp_col else np.nan
                    out['城市'] = city
                    out = out.dropna(subset=['Year'])
                    econ_all.append(out)
                except Exception as e:
                    print(f"[经济社会变量] 无法读取文件 {file}，跳过。原因：{e}")

    if econ_all:
        result_df = pd.concat(econ_all, ignore_index=True)
        print(f"经济社会变量数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Fiscal_revenue', 'GDP'])


def load_industry_data(industry_path):
    """加载产业结构数据"""
    ind_all = []
    if os.path.exists(industry_path):
        for file in os.listdir(industry_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(industry_path, file)
                try:
                    df = pd.read_excel(file_path)
                    year_col = find_column_by_keywords(df.columns, ('year', '年份', '年度'), default_index=0)
                    second_col = find_column_by_keywords(df.columns, ('第二产业', '二产业'), default_index=1)
                    third_col = find_column_by_keywords(df.columns, ('第三产业', '三产业'), default_index=2)

                    out = pd.DataFrame()
                    out['Year'] = df[year_col].apply(extract_year)
                    out['Secondary_industry'] = safe_to_numeric(df[second_col]) if second_col else np.nan
                    out['Tertiary_industry'] = safe_to_numeric(df[third_col]) if third_col else np.nan
                    out['城市'] = city
                    out = out.dropna(subset=['Year'])
                    ind_all.append(out)
                except Exception as e:
                    print(f"[产业结构] 无法读取文件 {file}，跳过。原因：{e}")

    if ind_all:
        result_df = pd.concat(ind_all, ignore_index=True)
        print(f"产业结构数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Secondary_industry', 'Tertiary_industry'])


def load_innovation_data(innovation_path):
    """加载技术创新数据（专利授权量）"""
    innovation_all = []
    if os.path.exists(innovation_path):
        for file in os.listdir(innovation_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(innovation_path, file)
                try:
                    df = pd.read_excel(file_path)
                    if len(df.columns) >= 2:
                        out = pd.DataFrame()
                        out['Year'] = df.iloc[:, 0].apply(extract_year)
                        out['Patent_grants'] = safe_to_numeric(df.iloc[:, 1])
                        out['城市'] = city
                        out = out.dropna(subset=['Year', 'Patent_grants'])
                        innovation_all.append(out)
                except Exception as e:
                    print(f"[技术创新] 无法读取文件 {file}，跳过。原因：{e}")

    if innovation_all:
        result_df = pd.concat(innovation_all, ignore_index=True)
        print(f"技术创新数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Patent_grants'])


def load_construction_data(construction_path):
    """加载建筑产业数据（企业数、总产值）"""
    construction_all = []
    if os.path.exists(construction_path):
        for file in os.listdir(construction_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(construction_path, file)
                try:
                    df = pd.read_excel(file_path)
                    if len(df.columns) >= 3:
                        out = pd.DataFrame()
                        out['Year'] = df.iloc[:, 0].apply(extract_year)
                        out['Num_construction_enterprises'] = safe_to_numeric(df.iloc[:, 1])
                        out['Gross_construction_output'] = safe_to_numeric(df.iloc[:, 2])
                        out['城市'] = city
                        out = out.dropna(subset=['Year', 'Num_construction_enterprises', 'Gross_construction_output'])
                        construction_all.append(out)
                except Exception as e:
                    print(f"[建筑产业] 无法读取文件 {file}，跳过。原因：{e}")

    if construction_all:
        result_df = pd.concat(construction_all, ignore_index=True)
        print(f"建筑产业数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Num_construction_enterprises', 'Gross_construction_output'])


def load_electricity_data(electricity_path):
    """加载用电量数据"""
    all_data = []
    if os.path.exists(electricity_path):
        for file in os.listdir(electricity_path):
            if file.endswith((".xlsx", ".xls")):
                city = os.path.splitext(file)[0]
                file_path = os.path.join(electricity_path, file)
                try:
                    df = pd.read_excel(file_path)
                    if len(df.columns) >= 2:
                        out = pd.DataFrame()
                        out['Year'] = df.iloc[:, 0].apply(extract_year)
                        out['Total_electricity_consumption'] = safe_to_numeric(df.iloc[:, 1])
                        out['城市'] = city
                        out = out.dropna(subset=['Year', 'Total_electricity_consumption'])
                        all_data.append(out)
                except Exception as e:
                    print(f"[用电量] 无法读取文件 {file}，跳过。原因：{e}")

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"用电量数据加载完成: {len(result_df)} 条记录")
        return result_df
    return pd.DataFrame(columns=['城市', 'Year', 'Total_electricity_consumption'])


def merge_all_data(base_path="25数据"):
    """整合所有数据源"""
    print("=" * 50)
    print("开始加载所有数据源...")
    print("=" * 50)

    # 加载各个数据源 (路径根据您的实际情况调整)
    target_df = load_target_data(os.path.join(base_path, "目标变量"))
    pop_df = load_population_data(os.path.join(base_path, "人口"))
    econ_df = load_economics_data(os.path.join(base_path, "经济社会变量"))
    industry_df = load_industry_data(os.path.join(base_path, "产业结构"))
    innovation_df = load_innovation_data(os.path.join(base_path, "技术创新"))
    policy_df = load_policy_data(os.path.join(base_path, "政策文件"))
    awareness_df = load_awareness_data(os.path.join(base_path, "公众意识"))
    construction_df = load_construction_data(os.path.join(base_path, "建筑产业"))
    electricity_df = load_electricity_data(os.path.join(base_path, "用电量"))

    # 合并所有数据
    print("\n开始合并数据...")
    merged_df = target_df.copy()

    # 依次合并各个数据源
    data_frames_to_merge = [
        (pop_df, "人口"), (econ_df, "经济"), (industry_df, "产业"),
        (innovation_df, "创新"),
        (policy_df, "政策"), (awareness_df, "意识"),  # --- 新增 ---
        (construction_df, "建筑"), (electricity_df, "用电")
    ]

    for df, name in data_frames_to_merge:
        if not df.empty:
            merged_df = pd.merge(merged_df, df, on=['城市', 'Year'], how='left')
            print(f"已合并{name}数据")

    # 数据清洗 (不变)
    print("\n数据清洗...")
    merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce')
    merged_df = merged_df.dropna(subset=['Year'])
    merged_df['Year'] = merged_df['Year'].astype(int)
    merged_df = merged_df[(merged_df['Year'] >= 2011) & (merged_df['Year'] <= 2022)]

    print(f"合并后数据形状: {merged_df.shape}")
    print(f"缺失值统计:\n{merged_df.isnull().sum()}")

    merged_df.to_csv("merged_raw_data.csv", index=False, encoding='utf-8-sig')
    print(f"\n原始合并数据已保存至: merged_raw_data.csv")

    return merged_df


if __name__ == "__main__":
    # 测试数据加载
    df = merge_all_data()
    print(f"\n最终数据形状: {df.shape}")
    print(f"城市数量: {df['城市'].nunique()}")
    print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")
    print(f"\n数据预览:\n{df.head()}")

