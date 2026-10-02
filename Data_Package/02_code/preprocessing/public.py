import pandas as pd
import os
import glob


def process_city_waste_data():
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
    folder_path = 'public'

    # 定义需要的年份列
    years = list(range(2011, 2024))  # 2011 到 2023

    # 用于存储所有城市处理后的数据
    all_data = []

    print(f"开始处理 {len(target_cities)} 个城市的数据...")

    for city in target_cities:
        # 假设文件名就是 "城市名.csv"
        file_path = os.path.join(folder_path, f"{city}.csv")

        if not os.path.exists(file_path):
            print(f"警告: 未找到文件 {file_path}，该城市行将为空。")
            all_data.append(pd.Series(name=city, dtype='float64'))
            continue

        try:
            # === 修复1：增强的读取逻辑 (自动尝试 GBK 和 UTF-8) ===
            df = None
            # 优先尝试 gbk (中文CSV常见编码)，然后尝试 utf-8
            encodings_to_try = ['gbk', 'utf-8', 'gb18030']

            for encoding in encodings_to_try:
                try:
                    temp_df = pd.read_csv(file_path, encoding=encoding)
                    temp_df.columns = temp_df.columns.str.strip()  # 去除列名空格
                    if '日期' in temp_df.columns and '固废' in temp_df.columns:
                        df = temp_df
                        # print(f"成功读取 {city} (编码: {encoding})") # 调试用
                        break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue

            if df is None:
                print(f"错误: {city}.csv 无法读取或关键列名('日期', '固废')缺失。请检查文件编码或列名。")
                all_data.append(pd.Series(name=city, dtype='float64'))
                continue

            # === 修复2：更灵活的日期解析 ===
            # 不指定 format，让 pandas 自动推断 (兼容 2014/7/9, 2014-07-09 等)
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

            # 提取年份
            df['Year'] = df['日期'].dt.year

            # 筛选 2011-2023 年的数据
            mask = (df['Year'] >= 2011) & (df['Year'] <= 2023)
            df_filtered = df.loc[mask]

            if df_filtered.empty:
                print(f"提示: {city} 在 2011-2023 期间无有效数据 (日期解析后为空)。")
                all_data.append(pd.Series(name=city, dtype='float64'))
                continue

            # === 修改3：按年统计改为求和 (sum) ===
            yearly_stats = df_filtered.groupby('Year')['固废'].sum()

            # 将该城市的Series重命名为城市名
            yearly_stats.name = city

            all_data.append(yearly_stats)

        except Exception as e:
            print(f"处理 {city} 时发生未知错误: {e}")
            all_data.append(pd.Series(name=city, dtype='float64'))

    # 2. 合并数据
    print("正在合并数据并生成 Excel...")

    # 将列表转换为 DataFrame (行是城市，列是年份)
    result_df = pd.DataFrame(all_data)

    # 3. 数据整理
    # 确保列只包含 2011-2023，且按顺序排列
    # reindex 会自动处理缺失的年份，填入 NaN
    result_df = result_df.reindex(columns=years)

    # 确保行按照目标城市列表顺序排列
    result_df = result_df.reindex(target_cities)

    # 4. 保存文件
    output_filename = '城市固废统计_2011-2023.xlsx'
    try:
        result_df.to_excel(output_filename, index_label='城市')
        print(f"处理完成！文件已保存为: {output_filename}")
    except PermissionError:
        print(f"错误: 无法写入文件 {output_filename}。请确保该文件未被打开，然后重试。")


if __name__ == "__main__":
    process_city_waste_data()