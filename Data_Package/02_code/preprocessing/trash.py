import pandas as pd
import os
import warnings

# 忽略读取Excel时的一些样式警告
warnings.filterwarnings('ignore', category=UserWarning)

# ================= 配置区域 =================

# 文件夹名称
FOLDER_PATH = 'trash'

# 年份范围
YEARS = [str(y) for y in range(2011, 2024)]

# 城市名所在列 (Excel中 A=0, B=1)
COL_INDEX_CITY = 0

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

def extract_trash_data():
    # 初始化三个结果DataFrame：行是城市，列是年份
    df_vol = pd.DataFrame(index=CITY_LIST, columns=YEARS)  # 清运量
    df_cap = pd.DataFrame(index=CITY_LIST, columns=YEARS)  # 无害化处理能力
    df_veh = pd.DataFrame(index=CITY_LIST, columns=YEARS)  # 市容环卫车辆数

    print(f"开始处理 '{FOLDER_PATH}' 文件夹，共 {len(YEARS)} 个年份文件...")

    for year in YEARS:
        file_name = f"{year}.xlsx"
        file_path = os.path.join(FOLDER_PATH, file_name)

        # === 核心修改1：根据年份设置三类数据的列索引 ===
        year_int = int(year)

        # 1. 清运量 (始终D列/索引3)
        col_vol = 3

        # 2. 无害化处理能力 & 3. 车辆数
        if 2011 <= year_int <= 2016:
            col_cap = 10  # K列 (无害化)
            col_veh = 22  # W列 (车辆数)
            desc = "2011-2016规则 (清运量:D, 无害化:K, 车辆:W)"
        else:
            col_cap = 9  # J列 (无害化)
            col_veh = 19  # T列 (车辆数)
            desc = "2017-2023规则 (清运量:D, 无害化:J, 车辆:T)"

        print(f"正在处理: {file_name} [{desc}]")

        if not os.path.exists(file_path):
            print(f"[警告] 文件 {file_path} 不存在，跳过。")
            continue

        try:
            # === 核心修改2：针对2011年特殊处理多sheet (逻辑保持不变) ===
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
                    # === 核心修改3：针对特定城市选择第二行 (逻辑保持不变) ===
                    if city in ["吉林", "重庆"] and len(matched_rows) > 1:
                        target_row = matched_rows.iloc[1]
                        print(f"  [特殊规则] {year}年 '{city}' 匹配到 {len(matched_rows)} 行，已选取第2行数据。")
                    else:
                        if len(matched_rows) > 1:
                            print(f"  [提示] {year}年 '{city}' 匹配到 {len(matched_rows)} 行，默认使用第1行。")
                        target_row = matched_rows.iloc[0]

                    # 提取数据并存入对应的DataFrame
                    df_vol.at[city, year] = target_row.iloc[col_vol]  # 清运量
                    df_cap.at[city, year] = target_row.iloc[col_cap]  # 无害化
                    df_veh.at[city, year] = target_row.iloc[col_veh]  # 车辆数
                else:
                    pass

        except Exception as e:
            print(f"[错误] 处理文件 {file_name} 时出错: {e}")

    # ================= 保存结果 =================

    print("正在保存文件...")

    try:
        # 生成三个文件
        output_vol = "清运量.xlsx"
        output_cap = "无害化处理能力.xlsx"
        output_veh = "市容环卫车辆数.xlsx"

        df_vol.to_excel(output_vol)
        df_cap.to_excel(output_cap)
        df_veh.to_excel(output_veh)

        print(f"处理完成！生成以下文件：")
        print(f"1. {output_vol}")
        print(f"2. {output_cap}")
        print(f"3. {output_veh}")

    except Exception as e:
        print(f"[错误] 保存文件时出错: {e}")


if __name__ == "__main__":
    extract_trash_data()