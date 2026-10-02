import pandas as pd
import os


def process_consume_data():
    # 1. 定义目标城市顺序列表 (严格按照用户提供的顺序)
    target_cities = [
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

    # 定义文件夹路径
    folder_path = 'consume'

    # 定义需要的年份
    years = list(range(2011, 2024))  # 2011 到 2023

    # 初始化结果 DataFrame，行索引为城市
    final_df = pd.DataFrame(index=target_cities)

    print(f"开始处理 {len(years)} 个年份的文件...")

    for year in years:
        file_name = f"{year}.xlsx"
        file_path = os.path.join(folder_path, file_name)

        # 1. 判断目标列
        # 2011-2016: E列 (索引4)
        # 2017-2023: C列 (索引2)
        # 假设 A列 (索引0) 永远是城市名称
        if 2011 <= year <= 2016:
            target_col_index = 4  # E列
            col_letter = "E"
        else:
            target_col_index = 2  # C列
            col_letter = "C"

        print(f"正在处理: {year}年 (目标: {col_letter}列)...")

        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，该年份数据将为空。")
            final_df[year] = None
            continue

        try:
            # === 使用 ExcelFile 并逐个 Sheet 读取 ===
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names

            # 创建一个临时的字典用于存储当前年份提取的数据 {城市: 数值}
            current_year_data = {city: None for city in target_cities}

            print(f"  - 发现 {len(sheet_names)} 个 Sheet，开始扫描...")

            # 遍历每一个 Sheet
            for sheet_name in sheet_names:
                try:
                    # 读取整个 Sheet
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)

                    # 检查列数是否足够
                    if df.empty or df.shape[1] <= target_col_index:
                        continue

                    # 提取需要的两列
                    df_extracted = df.iloc[:, [0, target_col_index]].copy()

                    # 重命名列
                    df_extracted.columns = ['City_Raw', 'Value']

                    # 数据清洗
                    df_extracted['City_Raw'] = df_extracted['City_Raw'].astype(str).str.strip()

                    # 遍历目标城市列表，去当前 Sheet 中查找
                    for target_city in target_cities:
                        # 如果该城市在这个年份已经找到数据了，跳过
                        if current_year_data[target_city] is not None:
                            continue

                        # === 修改核心逻辑：增加冲突检测 ===

                        # 1. 基础匹配：包含目标城市名
                        mask = df_extracted['City_Raw'].str.contains(target_city, na=False)

                        # 2. 冲突排除：鞍山 vs 马鞍山
                        # 如果我们要找 "鞍山"，必须排除包含 "马鞍山" 的行
                        if target_city == "鞍山":
                            mask = mask & ~df_extracted['City_Raw'].str.contains("马鞍山", na=False)

                        # 3. 冲突排除：吉林 vs 吉林省
                        # 如果我们要找 "吉林" (默认指吉林市)，必须排除包含 "省" 的行
                        if target_city == "吉林":
                            mask = mask & ~df_extracted['City_Raw'].str.contains("省", na=False)

                        match = df_extracted[mask]

                        if not match.empty:
                            # 如果找到匹配行，取第一个匹配项的 Value
                            val = pd.to_numeric(match.iloc[0]['Value'], errors='coerce')
                            current_year_data[target_city] = val
                            # print(f"    > 找到 {target_city}: {val} (Sheet: {sheet_name})") # 调试用

                except Exception as sheet_e:
                    print(f"    警告: 读取 Sheet '{sheet_name}' 时出错: {sheet_e}")
                    continue

            # 遍历完所有 Sheet 后，将整理好的该年数据映射到 final_df
            final_df[year] = final_df.index.map(current_year_data)

            # 关闭 Excel 文件句柄
            xls.close()

        except Exception as e:
            print(f"错误: 处理 {file_name} 时发生异常: {e}")
            final_df[year] = None

    # 4. 保存文件
    output_filename = '社会消费品零售总额.xlsx'
    print("正在保存结果...")

    try:
        final_df.to_excel(output_filename, index_label='城市')
        print(f"处理完成！文件已保存为: {output_filename}")
    except PermissionError:
        print(f"错误: 无法写入文件 {output_filename}。请确保该文件未被打开，然后重试。")


if __name__ == "__main__":
    process_consume_data()