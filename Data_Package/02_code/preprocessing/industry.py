import pandas as pd
import os
import warnings

# 忽略读取Excel时的一些样式警告
warnings.filterwarnings('ignore', category=UserWarning)

# ================= 配置区域 =================

# 文件夹名称
FOLDER_PATH = '1'

# 年份范围
YEARS = [str(y) for y in range(2011, 2024)]

# 目标列索引 (Excel中 A=0, B=1, C=2, D=3, E=4, F=5, G=6)
# 注意：pandas读取时如果不指定header，索引是严格对应的
COL_INDEX_CITY = 0  # 假设城市名在 A 列，如果在 B 列请改为 1

# 默认列索引 (适用于除2017外的年份)
COL_INDEX_SEC_DEFAULT = 4  # E列：第二产业
COL_INDEX_TER_DEFAULT = 6  # G列：第三产业

# 2017年特殊列索引
COL_INDEX_SEC_2017 = 3  # D列：第二产业
COL_INDEX_TER_2017 = 4  # E列：第三产业

# 113个城市列表 (严格按照要求的顺序)
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

def extract_data():
    # 初始化两个DataFrame
    # 行索引为城市(CITY_LIST)，列名为年份(YEARS)
    df_sec_result = pd.DataFrame(index=CITY_LIST, columns=YEARS)
    df_ter_result = pd.DataFrame(index=CITY_LIST, columns=YEARS)

    print(f"开始处理，共 {len(YEARS)} 个年份文件...")

    for year in YEARS:
        file_name = f"{year}.xlsx"
        file_path = os.path.join(FOLDER_PATH, file_name)

        # 判断并设置当前年份使用的列索引
        if year == "2017":
            current_col_sec = COL_INDEX_SEC_2017
            current_col_ter = COL_INDEX_TER_2017
            print(f"正在处理: {file_name} (使用特殊列索引: 第二产业D列, 第三产业E列)")
        else:
            current_col_sec = COL_INDEX_SEC_DEFAULT
            current_col_ter = COL_INDEX_TER_DEFAULT
            print(f"正在处理: {file_name} (使用默认列索引: 第二产业E列, 第三产业G列)")

        if not os.path.exists(file_path):
            print(f"[警告] 文件 {file_path} 不存在，跳过。")
            continue

        try:
            # 读取Excel，不使用表头(header=None)
            df_source = pd.read_excel(file_path, header=None)

            # 获取城市列的所有数据并转为字符串，去除前后空格，方便后续处理
            # 增加 .strip() 处理，防止 Excel 单元格里有看不见的空格导致匹配失败
            city_col_data = df_source.iloc[:, COL_INDEX_CITY].astype(str).str.strip()

            # 遍历每一个目标城市
            for city in CITY_LIST:
                # 1. 基础匹配：查找包含城市名的行
                mask = city_col_data.str.contains(city, na=False)

                # 2. 冲突解决逻辑 (精细化过滤)

                # 针对 "吉林"：排除 "吉林省"
                if city == "吉林":
                    mask = mask & (~city_col_data.str.contains("省", na=False))

                # 针对 "鞍山"：排除 "马鞍山" (防止搜 '鞍山' 时匹配到 '马鞍山')
                if city == "鞍山":
                    mask = mask & (~city_col_data.str.contains("马", na=False))

                # 针对 "中山"：排除 "中山区" (防止搜 '中山' 时匹配到大连的 '中山区')
                # 同时也排除可能存在的街道办等，如果不放心可以只匹配 "中山市" 或 "中山"
                if city == "中山":
                    mask = mask & (~city_col_data.str.contains("区", na=False))

                # 获取过滤后的匹配行
                matched_rows = df_source[mask]

                if not matched_rows.empty:
                    # 多重匹配预警：如果匹配到多于1行，打印警告
                    if len(matched_rows) > 1:
                        print(
                            f"  [提示] {year}年 '{city}' 匹配到 {len(matched_rows)} 行，默认使用第1行。请检查数据是否准确。")
                        # print(f"       匹配内容: {matched_rows.iloc[:, COL_INDEX_CITY].tolist()}")

                    target_row = matched_rows.iloc[0]

                    # 提取数据 (使用动态确定的列索引)
                    val_sec = target_row.iloc[current_col_sec]
                    val_ter = target_row.iloc[current_col_ter]

                    # 存入结果表 [城市, 年份]
                    df_sec_result.at[city, year] = val_sec
                    df_ter_result.at[city, year] = val_ter
                else:
                    # 如果需要调试，可以取消下面注释查看哪些城市没匹配到
                    # print(f"  - {year} 未匹配到: {city}")
                    pass

        except Exception as e:
            print(f"[错误] 处理文件 {file_name} 时出错: {e}")

    # ================= 保存结果 =================

    print("正在保存文件...")

    try:
        output_sec = "第二产业产值占比.xlsx"
        output_ter = "第三产业产值占比.xlsx"

        df_sec_result.to_excel(output_sec)
        df_ter_result.to_excel(output_ter)

        print(f"处理完成！")
        print(f"生成的第二产业文件: {output_sec}")
        print(f"生成的第三产业文件: {output_ter}")

    except Exception as e:
        print(f"[错误] 保存文件时出错: {e}")


if __name__ == "__main__":
    extract_data()