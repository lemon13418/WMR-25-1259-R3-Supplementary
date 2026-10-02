# -*- coding: utf-8 -*-
import json
import time
import pandas as pd
import random
import os
import urllib.parse
import re
from urllib.parse import quote

# 引入 Selenium 相关库
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("【错误】缺少必要库，请运行: pip install selenium webdriver_manager")
    exit()

# ================= 用户配置区域 =================

# 【新增】如果报错 "cannot find Chrome binary"，请在这里填入 Chrome 浏览器的exe实际完整路径
# Windows 示例: r"C:\Program Files\Google\Chrome\Application\chrome.exe"
# 如果为空，则使用默认路径
CHROME_PATH = r""

# 这里填入你的 Cookie 字符串
# 这里填入你自己的 Cookie 字符串（留空则启动后手动登录百度指数账号）
# 注意：作者的个人会话 Cookie 已从发布版本中移除，请勿上传个人凭证。
COOKIES_STR = ''

KEYWORD = '固废'

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

START_YEAR = 2011
END_YEAR = 2023


# =======================================================

class SeleniumBaiduScraper:
    def __init__(self, cookie_str, keyword):
        self.cookie_str = cookie_str
        self.keyword = keyword
        self.city_map = self.load_city_mapping()
        # 初始化时先不提取 CipherKey，改在运行时动态从浏览器获取
        self.driver = self.init_driver()

    def init_driver(self):
        """初始化 Selenium Chrome 浏览器"""
        print("正在启动 Chrome 浏览器...")
        chrome_options = Options()

        if CHROME_PATH:
            chrome_options.binary_location = CHROME_PATH

        # 禁用自动化控制特征
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # 移除 navigator.webdriver 特征
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        return driver

    def load_city_mapping(self):
        return {
            '北京': '911', '天津': '923', '上海': '910', '重庆': '904', '石家庄': '141',
            '太原': '231', '呼和浩特': '20', '沈阳': '150', '长春': '154', '哈尔滨': '152',
            '南京': '125', '杭州': '138', '合肥': '189', '福州': '50', '南昌': '5',
            '济南': '1', '郑州': '168', '武汉': '28', '长沙': '43', '广州': '95',
            '南宁': '90', '海口': '239', '成都': '97', '贵阳': '2', '昆明': '117',
            '拉萨': '466', '西安': '165', '兰州': '166', '西宁': '139', '银川': '140',
            '乌鲁木齐': '467', '大连': '29', '青岛': '77', '宁波': '289', '厦门': '54',
            '深圳': '94', '秦皇岛': '146', '唐山': '261', '保定': '259', '邯郸': '292',
            '长治': '228', '临汾': '232', '阳泉': '236', '大同': '227', '包头': '13',
            '赤峰': '21', '鞍山': '215', '抚顺': '222', '本溪': '220', '锦州': '217',
            '吉林': '270', '牡丹江': '322', '齐齐哈尔': '319', '大庆': '153', '苏州': '126',
            '南通': '163', '连云港': '156', '无锡': '127', '常州': '162', '扬州': '158',
            '徐州': '161', '温州': '149', '嘉兴': '304', '绍兴': '303', '台州': '287',
            '湖州': '305', '马鞍山': '185', '芜湖': '188', '泉州': '55', '九江': '6',
            '烟台': '78', '淄博': '81', '泰安': '353', '威海': '88', '枣庄': '85',
            '济宁': '352', '潍坊': '80', '日照': '366', '洛阳': '378', '安阳': '370',
            '焦作': '265', '开封': '264', '平顶山': '266', '荆州': '31', '宜昌': '35',
            '岳阳': '44', '湘潭': '47', '张家界': '226', '株洲': '46', '常德': '68',
            '湛江': '197', '珠海': '200', '汕头': '212', '佛山': '196', '中山': '207',
            '韶关': '201', '桂林': '91', '北海': '128', '三亚': '243', '柳州': '89',
            '绵阳': '98', '攀枝花': '112', '泸州': '103', '宜宾': '96', '遵义': '59',
            '曲靖': '339', '咸阳': '277', '延安': '401', '宝鸡': '273', '铜川': '271',
            '金昌': '343', '石嘴山': '472', '克拉玛依': '317'
        }

    def prepare_login(self):
        """注入 Cookie 并检查登录状态"""
        print("正在访问百度指数主页...")
        self.driver.get("https://index.baidu.com/v2/main/index.html")
        time.sleep(3)

        # 1. 解析并注入 Cookie
        print("正在注入 Cookies...")
        cookies = []
        for item in self.cookie_str.split(';'):
            if '=' in item:
                k, v = item.strip().split('=', 1)
                cookies.append({'name': k, 'value': v, 'domain': '.baidu.com'})

        for c in cookies:
            try:
                self.driver.add_cookie(c)
            except:
                pass

        print("刷新页面...")
        self.driver.refresh()
        time.sleep(3)

        if "登录" in self.driver.title or "login" in self.driver.current_url:
            print("\n" + "=" * 50)
            print("【注意】Cookie 似乎已过期或未生效。")
            print("请在弹出的浏览器窗口中【手动登录】百度账号。")
            print("登录成功后，请按回车键继续...")
            print("=" * 50 + "\n")
            input(">>> 登录完成后，请按回车键继续: ")
            print("正在继续运行...")
        else:
            print("Cookie 似乎有效。")

        # 4. 【关键步骤】强制跳转到关键词趋势页面
        print(f"正在跳转到关键词 [{self.keyword}] 的趋势详情页...")
        kw_enc = urllib.parse.quote(self.keyword)
        trend_url = f"https://index.baidu.com/v2/main/index.html#/trend/{kw_enc}?words={kw_enc}"
        self.driver.get(trend_url)

        print("等待趋势页加载...")
        time.sleep(5)
        print("准备就绪，开始执行抓取任务。")

    def decrypt(self, key, data):
        """保持原有的解密算法"""
        try:
            a = key
            i = data
            n = {}
            s = []
            for o in range(len(a) // 2):
                n[a[o]] = a[len(a) // 2 + o]
            for r in range(len(data)):
                s.append(n[i[r]])
            return "".join(s).split(",")
        except Exception as e:
            return []

    def get_latest_cipher_key(self):
        """
        【关键修复】从 Selenium 浏览器中动态读取最新的 BAIDUID
        selenium 的 driver.get_cookies() 权限很高，可以读取 HttpOnly Cookie
        """
        try:
            cookies = self.driver.get_cookies()
            for c in cookies:
                # 优先找 BAIDUID
                if c.get('name') == 'BAIDUID':
                    # BAIDUID 的格式通常是 FG=1 结尾，或者纯 hash
                    # 百度要求 CipherKey 就是 BAIDUID 的值
                    return c.get('value')

            # 如果没找到，尝试找 BAIDUID_BFESS
            for c in cookies:
                if c.get('name') == 'BAIDUID_BFESS':
                    return c.get('value')
        except Exception as e:
            print(f" [Python] 获取 CipherKey 失败: {e}")
        return ""

    def fetch_data_via_js(self, url, cipher_key):
        """
        核心黑科技：在浏览器内部使用 JS 的 fetch 发起异步请求。
        【核心修复】接受 Python 传来的、确保是最新的 cipher_key
        """
        js_script = """
            var url = arguments[0];
            var cipherKey = arguments[1]; // 接收 Python 传来的 CipherKey (必填)
            var callback = arguments[arguments.length - 1];

            // 简单检查
            if (!cipherKey) {
                callback(JSON.stringify({error: "No CipherKey provided from Python"}));
                return;
            }

            fetch(url, {
                method: 'GET',
                credentials: 'include',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Accept': 'application/json, text/plain, */*',
                    'User-Agent': navigator.userAgent,
                    'Cipher-Text': cipherKey // 使用 Python 读取到的 HttpOnly Cookie 值
                }
            })
            .then(response => {
                // 无论成功失败，都返回更多调试信息
                return response.text().then(text => {
                    return JSON.stringify({
                        status: response.status,
                        statusText: response.statusText,
                        body: text
                    });
                });
            })
            .then(jsonStr => callback(jsonStr))
            .catch(err => callback(JSON.stringify({error: err.toString()})));
        """
        try:
            self.driver.set_script_timeout(30)
            # 【重要】将 cipher_key 传给 JS
            response_json_str = self.driver.execute_async_script(js_script, url, cipher_key)

            if not response_json_str:
                return None

            try:
                resp_obj = json.loads(response_json_str)
            except:
                print(f" [JS] 返回非 JSON 字符串: {response_json_str[:100]}")
                return None

            if "error" in resp_obj:
                print(f" [JS Fetch 错误] {resp_obj['error']}")
                return None

            if resp_obj['status'] != 200:
                print(f" [HTTP {resp_obj['status']}] {resp_obj['statusText']}")
                return None

            return resp_obj['body']

        except Exception as e:
            print(f" [Python] JS执行异常: {e}")
            return None

    def get_ptbk(self, uniqid, cipher_key):
        url = f"https://index.baidu.com/Interface/api/ptbk?uniqid={uniqid}"
        # 同样需要传 Key
        resp_text = self.fetch_data_via_js(url, cipher_key)
        if resp_text:
            try:
                return json.loads(resp_text).get('data')
            except:
                print(f" ptbk 解析失败: {resp_text[:50]}...")
                pass
        return None

    def fetch_data_by_year(self, area_code, year):
        # 1. 每次请求前，先获取最新的 CipherKey (解决 HttpOnly 问题)
        current_cipher_key = self.get_latest_cipher_key()
        if not current_cipher_key:
            print(" [调试] 无法获取 BAIDUID，跳过")
            return None

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        word_json = json.dumps([{"name": self.keyword, "wordType": 1}], ensure_ascii=False)
        params = {
            'area': area_code,
            'word': word_json,
            'startDate': start_date,
            'endDate': end_date
        }
        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        url = f"https://index.baidu.com/api/Search/getIndex?{query_string}"

        # 2. 传入 Key
        resp_text = self.fetch_data_via_js(url, current_cipher_key)

        if not resp_text:
            print(" [调试] 无响应内容")
            return None

        try:
            data_json = json.loads(resp_text)

            if data_json['status'] != 0:
                msg = str(data_json.get('message', ''))
                if "not login" in msg:
                    return "COOKIE_EXPIRED"
                if "10018" in str(data_json['status']):
                    print(f" [风控10018]", end="")
                    return "RISK_CONTROL"
                print(f" [API Status {data_json['status']}] {msg}")
                return None

            data_content = data_json['data']
            if not data_content or 'userIndexes' not in data_content:
                return 0

            uniqid = data_content['uniqid']
            user_indexes = data_content.get('userIndexes', [])
            if not user_indexes:
                return 0

            enc_data = user_indexes[0]['all']['data']

            # 3. ptbk 也要传 Key
            ptbk = self.get_ptbk(uniqid, current_cipher_key)
            if not ptbk:
                print(" 获取ptbk失败", end="")
                return None

            decrypted_data = self.decrypt(ptbk, enc_data)
            result_values = [int(x) if x else 0 for x in decrypted_data]

            if result_values:
                avg_val = sum(result_values) / len(result_values)
                return avg_val
            return 0

        except json.JSONDecodeError:
            print(f" JSON解析失败: {resp_text[:100]}...")
            return None
        except Exception as e:
            print(f" 处理逻辑异常: {e}")
            return None

    def run(self, city_list, start_year, end_year):
        print(f"启动 Selenium 抓取: [{self.keyword}] ({start_year}-{end_year})")

        self.prepare_login()

        backup_file = f"baidu_index_{self.keyword}_backup.xlsx"
        final_file = f"baidu_index_{self.keyword}_{start_year}_{end_year}.xlsx"

        if os.path.exists(backup_file):
            print(f"加载备份: {backup_file}")
            try:
                df = pd.read_excel(backup_file, index_col=0)
                df.columns = [int(c) for c in df.columns]
            except:
                df = pd.DataFrame(index=city_list, columns=range(start_year, end_year + 1))
        else:
            df = pd.DataFrame(index=city_list, columns=range(start_year, end_year + 1))

        for city in city_list:
            if city not in df.index:
                df.loc[city] = None

        for city in city_list:
            city_code = self.city_map.get(city) or self.city_map.get(city + "市")
            if not city_code:
                print(f"警告: 跳过未知城市 {city}")
                continue

            print(f"处理: {city} (Code: {city_code})")

            for year in range(start_year, end_year + 1):
                current_val = df.loc[city, year]
                if pd.notna(current_val) and str(current_val) != "":
                    continue

                print(f"  > {year}: ", end="")

                retries = 3
                success = False
                while retries > 0:
                    val = self.fetch_data_by_year(city_code, year)

                    if val == "COOKIE_EXPIRED":
                        print("\n【需要登录】Cookie失效，请在浏览器中手动登录！")
                        input("登录后按回车继续...")
                        continue

                    if val == "RISK_CONTROL":
                        print("\n【被风控】暂停 30 秒...")
                        time.sleep(30)
                        retries -= 1
                        continue

                    if val is not None:
                        df.loc[city, year] = round(val, 2)
                        print(f"{round(val, 2)}")
                        success = True
                        break
                    else:
                        retries -= 1
                        time.sleep(2)

                if not success:
                    print("无数据/失败")

                time.sleep(random.uniform(1.5, 3.0))

            df.to_excel(backup_file)
            print("-" * 30)

        df.to_excel(final_file)
        print(f"完成！文件保存至: {final_file}")
        self.driver.quit()


if __name__ == "__main__":
    scraper = SeleniumBaiduScraper(COOKIES_STR, KEYWORD)
    scraper.run(CITY_LIST, START_YEAR, END_YEAR)