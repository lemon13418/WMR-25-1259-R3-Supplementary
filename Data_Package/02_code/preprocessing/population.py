import pandas as pd
import os
import glob
import re


def clean_text(text):
    """
    清理文本：去除所有空白字符（包括空格、制表符、换行符、全角空格等）
    """
    if pd.isna(text):
        return ""
    # 将内容转为字符串
    text = str(text)
    # 去除两端空白
    text = text.strip()
    # 去除内部所有空白字符 (包括 \t, \n, \r, \f, \v 和空格)
    # \s 匹配任意空白字符
    text = re.sub(r'\s+', '', text)
    return text


def convert_xls_to_xlsx(xls_path, xlsx_path):
    """
    将 .xls 文件转换为 .xlsx 文件
    尝试使用 xlrd 读取，如果失败则尝试 read_html (针对伪装成xls的html文件)
    """
    try:
        # 尝试标准读取 (需要安装 xlrd)
        # engine='xlrd' 专门处理 .xls
        df = pd.read_excel(xls_path, header=None, engine='xlrd')
        df.to_excel(xlsx_path, index=False, header=None)
        return True
    except Exception as e:
        print(f"  普通模式转换失败 ({e})，尝试 HTML 兼容模式...")
        try:
            # 针对国内常见的将 HTML 表格直接改名为 .xls 的情况
            dfs = pd.read_html(xls_path, header=None)
            if dfs:
                dfs[0].to_excel(xlsx_path, index=False, header=None)
                return True
        except Exception as e2:
            print(f"  转换失败: {e2}")
            return False
    return False


def extract_population_data():
    # 1. 定义目标城市列表（严格按照用户提供的顺序）
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

    # 2. 定义年份和文件夹路径
    years = [str(y) for y in range(2011, 2024)]
    folder_path = "population"

    # 初始化用于存储结果的 DataFrame
    # 行名为城市，列名为年份
    df_total_pop = pd.DataFrame(index=target_cities, columns=years)
    df_urban_pop = pd.DataFrame(index=target_cities, columns=years)

    print(f"开始处理 {folder_path} 文件夹下的数据...")

    # 3. 遍历每一年
    for year in years:
        xlsx_path = os.path.join(folder_path, f"{year}.xlsx")
        xls_path = os.path.join(folder_path, f"{year}.xls")

        target_file_path = xlsx_path

        # 检查是否需要转换格式
        if not os.path.exists(xlsx_path):
            if os.path.exists(xls_path):
                print(f"发现 {year}.xls，正在转换为 .xlsx 格式...")
                success = convert_xls_to_xlsx(xls_path, xlsx_path)
                if success:
                    print(f"  已保存为 {year}.xlsx")
                else:
                    print(f"错误: 无法转换 {year}.xls，跳过该年份。")
                    continue
            else:
                print(f"警告: {year} 年的文件 (.xlsx 或 .xls) 均不存在，跳过。")
                continue

        # 此时应该已经有 .xlsx 文件了
        try:
            # 读取 Excel 文件
            # header=None 确保我们可以通过索引访问 C列(2) 和 D列(3)
            df = pd.read_excel(target_file_path, header=None)

            # 数据预处理：清理前两列的数据（转字符串，去所有空格）
            df[0] = df[0].apply(clean_text)
            df[1] = df[1].apply(clean_text)

            print(f"正在处理: {year}.xlsx")

            # 遍历目标城市，寻找数据
            for city in target_cities:
                matched = False
                found_value_total = None
                found_value_urban = None

                # 构建严格匹配的集合
                # 只有单元格内容完全等于 "城市名" 或 "城市名+市" 才算匹配
                valid_names = {city, city + "市"}

                # 遍历 Excel 的每一行寻找匹配的城市
                for index, row in df.iterrows():
                    val_a = row[0]  # 已清洗过的A列
                    val_b = row[1]  # 已清洗过的B列

                    # 检查是否完全匹配
                    if val_a in valid_names or val_b in valid_names:
                        try:
                            found_value_total = row[2]  # C列
                            found_value_urban = row[3]  # D列
                            matched = True
                            break
                        except KeyError:
                            print(f"  错误: 在 {year} 年找到 {city}，但列索引越界。")

                if matched:
                    # 赋值时，行是 city，列是 year
                    df_total_pop.at[city, year] = found_value_total
                    df_urban_pop.at[city, year] = found_value_urban
                else:
                    pass

        except Exception as e:
            print(f"处理 {year}.xlsx 时发生错误: {e}")

    # 4. 保存结果到文件
    output_total_file = "extracted_total_population.xlsx"
    output_urban_file = "extracted_urban_population.xlsx"

    try:
        # 保存时，DataFrame 的 index 就是城市名，columns 就是年份
        df_total_pop.to_excel(output_total_file, index_label="城市")
        df_urban_pop.to_excel(output_urban_file, index_label="城市")
        print("\n" + "=" * 30)
        print("提取完成！")
        print(f"总人口数据已保存至: {output_total_file} (行:城市, 列:年份)")
        print(f"城市人口数据已保存至: {output_urban_file} (行:城市, 列:年份)")
        print("=" * 30)
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


if __name__ == "__main__":
    extract_population_data()