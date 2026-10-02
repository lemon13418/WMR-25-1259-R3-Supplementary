import time
import random
import pandas as pd
from playwright.sync_api import sync_playwright

# --- 配置区域 ---
YEAR_START = 2011
YEAR_END = 2024

# 目标城市列表
TARGET_CITIES = [
    "北京", "天津", "上海", "重庆",
    "石家庄", "太原", "呼和浩特", "沈阳", "长春", "哈尔滨",
    "南京", "杭州", "合肥", "福州", "南昌", "济南",
    "郑州", "武汉", "长沙", "广州", "南宁", "海口",
    "成都", "贵阳", "昆明", "拉萨", "西安", "兰州",
    "西宁", "银川", "乌鲁木齐",
    "大连", "青岛", "宁波", "厦门", "深圳",
    "秦皇岛", "唐山", "保定", "邯郸", "长治", "临汾", "阳泉", "大同",
    "包头", "赤峰", "鞍山", "抚顺", "本溪", "锦州", "吉林", "牡丹江",
    "齐齐哈尔", "大庆", "苏州", "南通", "连云港", "无锡", "常州", "扬州",
    "徐州", "温州", "嘉兴", "绍兴", "台州", "湖州", "马鞍山", "芜湖",
    "泉州", "九江", "烟台", "淄博", "泰安", "威海", "枣庄", "济宁",
    "潍坊", "日照", "洛阳", "安阳", "焦作", "开封", "平顶山", "荆州",
    "宜昌", "岳阳", "湘潭", "张家界", "株洲", "常德", "湛江", "珠海",
    "汕头", "佛山", "中山", "韶关", "桂林", "北海", "三亚", "柳州",
    "绵阳", "攀枝花", "泸州", "宜宾", "遵义", "曲靖", "咸阳", "延安",
    "宝鸡", "铜川", "金昌", "石嘴山", "克拉玛依"
]


def perform_search(page, query):
    """
    执行单次搜索并返回结果列表
    """
    search_url = f"https://cn.bing.com/search?q={query}"

    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
        # 这里的等待非常重要，有时候 Bing 会先加载框架再填入内容
        page.wait_for_selector("#b_results", timeout=8000)
    except Exception as e:
        print(f"    [警告] 页面加载超时或失败: {e}")
        return []

    # --- 修正点 1: 放宽选择器 ---
    # 不要限定 li.b_algo，因为 PDF 结果或置顶结果可能没有这个类
    # 直接找 #b_results 下所有的 h2 标题链接
    search_results = page.locator("#b_results h2 a").all()

    print(f"    [调试] 本页共抓取到 {len(search_results)} 个标题 (前5个将被分析)")  # 调试信息

    found_items = []
    # 遍历前 5 条
    for i, link in enumerate(search_results[:5]):
        try:
            title = link.inner_text()
            url = link.get_attribute("href")

            print(f"      [{i + 1}] 分析标题: {title}")  # 调试信息：让你看到它读到了什么

            # --- 修正点 2: 极简过滤 ---
            # 只要包含 "固" 或者 "环境" 或者是 PDF，且看似是公告，就先留着
            # 宁滥勿缺，后续在 Excel 里删总比抓不到强
            is_relevant = False

            if "固" in title: is_relevant = True
            if "废" in title: is_relevant = True
            if "环境" in title and "公" in title: is_relevant = True  # 匹配 "环境状况公报" 这种可能包含固废的

            if is_relevant:
                found_items.append({
                    "title": title,
                    "url": url,
                })
        except Exception as e:
            print(f"      [跳过] 解析错误: {e}")
            continue

    return found_items


def run_scraper():
    results_data = []

    with sync_playwright() as p:
        # headless=False 方便你盯着浏览器看
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        print(f"开始抓取任务：共 {len(TARGET_CITIES)} 个城市")

        for city in TARGET_CITIES:
            print(f"--- 处理城市: {city} ---")
            for year in range(YEAR_START, YEAR_END + 1):
                found_for_year = False

                # 策略列表
                strategies = [
                    f"site:gov.cn {city} 生态环境局 {year} 固体废物 公告",  # 针对大城市
                    f"site:gov.cn {city} {year}年 固体废物污染环境防治信息公告",  # 标准
                    f"site:gov.cn {city} {year} 固废 公告"  # 兜底
                ]

                unique_urls = set()

                for idx, query in enumerate(strategies):
                    if found_for_year and len(unique_urls) >= 1:
                        break

                    print(f"  > [{year}] 策略 {idx + 1}: {query[:30]}...")

                    items = perform_search(page, query)

                    if items:
                        for item in items:
                            # --- 修正点 3: 移除严格的年份过滤 ---
                            # 既然搜索词里已经带了年份，Bing给出的前几条大概率是相关的
                            # 如果强行检查 str(year) in title，会漏掉 "二〇一一年" 或 "去年" 这种描述
                            if item['url'] not in unique_urls:
                                print(f"    [命中] {item['title']}")
                                results_data.append({
                                    "city": city,
                                    "year": year,
                                    "title": item['title'],
                                    "url": item['url'],
                                    "source": f"Strategy_{idx + 1}"
                                })
                                unique_urls.add(item['url'])
                                found_for_year = True

                        time.sleep(random.uniform(1.0, 2.0))

                    if found_for_year:
                        # 如果某个策略找到了结果，通常不需要再试后续策略了，节省时间
                        break

                if not found_for_year:
                    print(f"  [X] {year} 年未找到结果")

                # 随机等待
                time.sleep(random.uniform(1.0, 2.5))

            # 定期保存
            df = pd.DataFrame(results_data)
            df.to_csv("solid_waste_urls_part.csv", index=False, encoding="utf-8-sig")

        browser.close()

    final_df = pd.DataFrame(results_data)
    final_df.to_csv("solid_waste_urls_final.csv", index=False, encoding="utf-8-sig")
    print("全部任务结束")


if __name__ == "__main__":
    run_scraper()