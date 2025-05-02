import requests
import json
import re
import pandas as pd
from collections import Counter


class scraper:

    def __init__(self, keywords):
        self.keywords = keywords
        self.data = self.scrap()


    def scrap(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://www.104.com.tw/jobs/search/"
        }

        pages = list(range(1, 30)) # 讀前 29 頁
        url = "https://www.104.com.tw/jobs/search/list"
        jobs = []

        for keyword in self.keywords:
            for page in pages:
                params = {
                    "ro": "1",  # 限定全職或兼職 (0: 全部, 1:全職)
                #     "kwop": "7",  # 關鍵字匹配方式
                    "keyword": keyword,  # 搜尋關鍵字S
                    "order": "15",  # 排序 (15: 相關性)
                    "asc": "0",  # 降冪排列
                    "page": str(page),  # 頁數
                    "mode": "s"  # 搜尋模式
                }
                response = requests.get(url, headers=headers, params=params)
                data = response.json()

                tmp = data["data"]["list"]
                tmp = [{**obj, 'query': keyword} for obj in tmp] #  把查詢的字放入
                jobs += tmp

        data = pd.DataFrame(jobs)
        print(f'爬取職缺總數: {len(jobs)} 筆')
        print(f'欄位資訊: {list(data.columns)}')

        return data


    def preprocessing(self):

        temp = self.data.copy()
        temp = self.remove_duplicate_and_unrelated(temp) # 移除重複跟不相關的職缺

        temp['descWithoutHighlight'] = temp['descWithoutHighlight'].str.replace(r'\n|\t', '', regex=True) # 移除空行字元
        temp['job_category'] = temp['jobName'].str.replace(r'[^\w\u4e00-\u9fff]', ' ', regex=True).str.lower().apply(lambda x: self.get_job_category(x))
        temp['applyCnt'] = temp['applyCnt'].astype(int) # 應徵人數: 原本是字串
        temp['salaryLow'] = temp['salaryLow'].astype(int) # 原本是字串
        temp['salaryHigh'] = temp['salaryHigh'].astype(int) # 原本是字串

        # 統整成月薪範圍
        temp['min_salary'] = temp.apply(lambda row: self.get_monthly_salary(row['salaryType'], row['salaryLow'], row['salaryHigh'])[0], axis=1)
        temp['max_salary'] = temp.apply(lambda row: self.get_monthly_salary(row['salaryType'], row['salaryLow'], row['salaryHigh'])[1], axis=1)

        temp, major_cols = self.get_top_majors(temp)

        # 解析公司資訊
        temp['tags'] = temp['tags'].apply(lambda i: i if isinstance(i, dict) else {}) # 因為 tags 裡面會有空的陣列，所以要先將空的陣列轉換成空的 dict
        company_info = self.get_company_info(temp['tags'])

        cols = [
                'jobType', 'jobNo', 'jobName', 'jobRole', 'jobRo',
                'jobAddrNoDesc', 'jobAddress',
                'descWithoutHighlight', 'optionEdu', 'period', 'periodDesc', 'applyCnt',
                'applyType', 'applyDesc', 'custNo', 'custName', 'coIndustry',
                'coIndustryDesc', 'salaryDesc',
                'landmark','jobsource', 'lon', 'lat',
                'remoteWorkType', 'major', 'salaryType', 'dist', 'mrt', 'mrtDesc',
                'query', 'min_salary', 'max_salary', 'emp_desc', 'emp_param',
                'zone_desc', 'zoneForeign_desc',
                'remote_desc', 'employees', 'job_category'
        ]
        cols += major_cols

        result = pd.concat([temp, company_info], axis=1)[cols]

        return result


    # 月薪資料處理
    def get_monthly_salary(self, salary_type, low, high):

        if salary_type == '': return [0, 0] # 待遇面議
        if salary_type == 'M': return [low, high]
        if salary_type == 'D': return [low*30, high*30]
        if salary_type == 'Y': return [low/12, high/12]
        if salary_type == 'H': return [low*8*20, high*8*20] # 時薪


    def remove_duplicate_and_unrelated(self, data):
        temp = data.copy()
        temp = temp.drop_duplicates(subset=['jobNo', 'jobName'], keep='last')# 移除重複職缺

        # 只留下相關的職缺
        titles = '數據分析|分析師|AI|Analytics|data scientist|Data Analysis|analyst|Machine Learning|BI|數據工程師|資料工程師|資料分析|data engineer|business intelligence|商業分析|資料科學|數據|Data Mining|人工智慧|機器學習|深度學習|Deep Learning'
        temp = temp[temp['jobName'].str.contains(titles, regex=True, case=False)]

        # 調整 index 後面才能合併
        temp.reset_index(drop=True, inplace=True)

        return temp


    def get_company_info(self, tags):

        info = pd.json_normalize(tags, sep='_')
        info['employees'] = info['emp_desc'].str.extract(r'員工(\d+)人')

        return info


    def get_top_majors(self, data):

        temp = data.copy()
        # 計算每種科系要求出現次數
        majors = pd.DataFrame.from_dict(Counter(list(sum(temp['major'], []))), orient='index', columns=['count'])

        # 取出出現 30 次以上的科系名稱
        major_list = list(majors[majors['count'] > 30].sort_values(by='count', ascending=False).index)

        print(f'出現 30 次以上的科系: {major_list}')

        major_cols = []

        # 將出現較30次以上的科系，做 one-hot encoding
        for major in major_list:
            col_name =  'major_' + major
            major_cols.append(col_name)
            temp[col_name] = temp['major'].apply(lambda x: 1 if major in x else 0)

        return temp, major_cols


    def get_job_category(self, name):

        if re.search(r'machine learning|機器學習|深度學習|deep learning|data mining|人工智慧|ml|ai|dl', name):
            return 'ML|AI'
        if re.search(r'數據工程師|資料工程師|data engineer', name):
            return 'DE'
        else:
            return 'DA/DS'