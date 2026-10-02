import pandas as pd
import os
import warnings

# 忽略读取Excel时的一些样式警告
warnings.filterwarnings('ignore', category=UserWarning)

# ================= 配置区域 =================

# 文件夹名称
FOLDER_PATH = 'finance'

# 年份范围
YEARS = [str(y) for y in range(2011, 2024)]

# 城市名所在列 (Excel中 A=0, B=1)
COL_INDEX_CITY = 0

# 目标列索引配置 (A=0, B=1, C=2...)
# C列=财政收入, D列=财政支出, E列=财政科技支出
TARGET_COLS = {
    "income": {"index": 2, "desc": "C列 (财政收入)", "filename": "财政收入提取结果.xlsx"},
    "expense": {"index": 3, "desc": "D列 (财政支出)", "filename": "财政支出提取结果.xlsx"},
    "tech": {"index": 4, "desc": "E列 (财政科技支出)", "filename": "财政科技支出提取结果.xlsx"}
}

# 113个城市列表 (保持顺序)
CITY_LIST = [
    "北京", "天津", "上海", "重庆", "石家庄", "太原", "呼和浩特", "沈阳", "长春", "哈尔滨",
    "南京", "杭州", "合肥", "福州", "南昌", "济南", "郑州", "武汉", "长沙", "广州",
    "南宁", "海口", "成都", "贵阳", "昆明", "拉萨", "西安", "兰州", "西宁", "银川",
    "乌鲁木齐", "大连", "青岛", "宁波", "厦门", "深圳", "秦皇岛", "唐山", "保定", "邯郸",
    "长治", "临汾", "阳泉", "大同", "包头", "赤峰", "鞍山", "抚顺", "本溪", "锦州",
    "吉林", "牡丹江", "齐齐哈尔", "大庆", "苏州", "南通", "连云港", "无锡", "常州", "扬州",
    "徐州", "温州", "嘉兴", "绍兴", "台州", "湖州", "马鞍山", "芜湖", "泉州", "九江",
    "烟台", "淄博", "泰安", "威海", "枣庄", "济宁", "潍坊", "日照", "洛阳", "安阳",
    "焦作", "开封", "平顶山", "荆州", "宜昌", "岳阳", "湘潭", "张家界", "株洲", "常德",
    "湛江", "珠海", "汕头", "佛山", "中山", "韶关", "桂林", "北海", "三亚", "柳州",
    "绵阳", "攀枝花", "泸州", "宜宾", "遵义", "曲靖", "咸阳", "延安", "宝鸡", "铜川",
    "金昌", "石嘴山", "克拉玛依"
]


# ================= 主逻辑 =================

def extract_finance_data():
    # 初始化三个结果DataFrame：行是城市，列是年份
    dfs = {
        "income": pd.DataFrame(index=CITY_LIST, columns=YEARS),
        "expense": pd.DataFrame(index=CITY_LIST, columns=YEARS),
        "tech": pd.DataFrame(index=CITY_LIST, columns=YEARS)
    }

    print(f"开始处理 '{FOLDER_PATH}' 文件夹，共 {len(YEARS)} 个年份文件...")
    print(f"目标提取: 财政收入(C列), 财政支出(D列), 财政科技支出(E列)")

    for year in YEARS:
        file_name = f"{year}.xlsx"
        file_path = os.path.join(FOLDER_PATH, file_name)

        print(f"正在处理: {file_name}")

        if not os.path.exists(file_path):
            print(f"[警告] 文件 {file_path} 不存在，跳过。")
            continue

        try:
            # === 核心逻辑保持不变：针对2011年特殊处理多sheet ===
            if year == "2011":
                print(f"  - 检测到2011年文件，正在合并所有Sheet的数据...")
                sheets_dict = pd.read_excel(file_path, header=None, sheet_name=None)
                df_source = pd.concat(sheets_dict.values(), ignore_index=True)
            else:
                df_source = pd.read_excel(file_path, header=None)

            # 预处理城市列：转字符串并去除所有空白字符
            city_col_data = df_source.iloc[:, COL_INDEX_CITY].astype(str).str.replace(r'\s+', '', regex=True)

            # 遍历每一个目标城市
            for city in CITY_LIST:
                # 1. 基础匹配
                mask = city_col_data.str.contains(city, na=False)

                # 2. 冲突解决逻辑 (保持不变)
                if city == "吉林":
                    mask = mask & (~city_col_data.str.contains("省", na=False))
                if city == "鞍山":
                    mask = mask & (~city_col_data.str.contains("马", na=False))
                if city == "中山":
                    mask = mask & (~city_col_data.str.contains("区", na=False))

                # 获取过滤后的匹配行
                matched_rows = df_source[mask]

                if not matched_rows.empty:
                    # === 核心逻辑保持不变：针对特定城市选择第二行 ===
                    if city in ["吉林", "重庆"] and len(matched_rows) > 1:
                        target_row = matched_rows.iloc[1]
                        # 仅在调试时取消注释，避免刷屏
                        # print(f"  [特殊规则] {year}年 '{city}' 匹配到 {len(matched_rows)} 行，已选取第2行数据。")
                    else:
                        target_row = matched_rows.iloc[0]

                    # === 提取三个指标 ===
                    for key, config in TARGET_COLS.items():
                        col_idx = config["index"]
                        # 确保列索引没有超出Excel范围
                        if col_idx < target_row.shape[0]:
                            val = target_row.iloc[col_idx]
                            dfs[key].at[city, year] = val
                        else:
                            print(f"[错误] {year}年 {city} 数据列索引 {col_idx} 超出范围")
                else:
                    pass  # 未找到匹配项

        except Exception as e:
            print(f"[错误] 处理文件 {file_name} 时出错: {e}")

    # ================= 保存结果 =================

    print("\n正在保存文件...")

    for key, config in TARGET_COLS.items():
        try:
            output_file = config["filename"]
            df_final = dfs[key]
            df_final.to_excel(output_file)
            print(f"成功生成: {output_file}")
        except Exception as e:
            print(f"[错误] 保存 {config['desc']} 文件时出错: {e}")

    print("\n所有任务处理完成！")


if __name__ == "__main__":
    extract_finance_data()