import json
import ast
import numpy as np
import pandas as pd

class preprocessor:
    
    def __init__(self, df):
        
        self.df = df
        self.skill_map = self.get_dict('skills.json')
        self.industry_map = self.get_dict('industry.json')
        self.industry_cols = []
        self.skill_cols = []
        self.skill_set_cols = [
            'skill_set_ml_ai', 'skill_set_data_engineer', 'skill_set_llm_image'
            ]
    
    def get_dict(self, file):
        
        with open(file, 'r', encoding='utf-8') as f:
            dictionary = json.load(f) # 讀取同義字字典
           
        return dictionary
        
    def col_preprocess(self):
        # 將整個欄位轉回 list 物件
        self.df['terms'] = self.df['terms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.df['log_max_salary'] = np.log1p(self.df['max_salary'])  # 加1避免log(0)
        self.df['log_min_salary'] = np.log1p(self.df['min_salary'])  # 加1避免log(0)
        # 薪資正規化
        self.df['max_salary_K'] = self.df['max_salary'] / 1000
        self.df['min_salary_K'] = self.df['min_salary'] / 1000
        
        self.df['employees'] = self.df['employees'].fillna(0)
        self.df['is_remote'] = self.df['remote_desc'].apply(lambda x: 1 if x=='遠端工作' else 0)
        self.df['employees'] = self.df['employees'].fillna(40) # 通常小型公司不會填員工數
        self.df['company_scale'] = self.df['employees'].apply(lambda x: self.scale_company(x))
        self.df['education'] = self.df['optionEdu'].apply(lambda x: self.label_encoding_education(x))
        self.df['education_master'] = self.df['optionEdu'].apply(lambda x: self.label_high_education(x))
        
        self.df['major_infomation'] =  self.df[['major_資訊工程相關', 'major_資訊工程相關']].sum(axis=1).gt(0).astype(int)
        self.df['major_engineer'] = self.df[['major_電機電子工程相關', 'major_工程學科類', 'major_機械工程相關']].sum(axis=1).gt(0).astype(int)
        self.df['major_math_and_stats'] =  self.df[['major_數學及電算機科學學科類', 'major_數理統計相關', 'major_其他數學及電算機科學相關', 'major_應用數學相關', 'major_統計學相關']].sum(axis=1).gt(0).astype(int)
        self.df['major_commercial'] = self.df[['major_商業及管理學科類', 'major_一般商業學類']].sum(axis=1).gt(0).astype(int)
        
        result = self.encoding()
        
        return result, self.skill_cols, self.skill_set_cols, self.industry_cols
    
    def encoding(self):
        
        temp = self.get_skill_cols()
        temp = self.get_skill_set(temp)
        temp = self.get_industry_group(temp)
        temp = self.get_region(temp)
        
        return temp
    
    # 新增技能相關欄位
    def get_skill_cols(self): 
        
        result = self.df.copy()

        for skill in self.skill_map.keys():
            col = 'skill_'+skill
            result[col] = 0 *len(result)
            self.skill_cols.append(col)
            
        for i, val in result.iterrows():
            self.check_skill(result, i, val['terms'])
        
        return result

    def check_skill(self, data, i, terms):

        for skill in self.skill_map.items():
            category = 'skill_'+skill[0]
            skill_list = skill[1]

            if len(set(terms)&set(skill_list))>0:
                data.loc[i, category] = 1

    def get_skill_set(self, data):
        result = data.copy()
        # 組合技能詞
        result['skill_set_ml_ai'] = result[['skill_深度學習','skill_機器學習 ai', 'skill_特徵工程']].sum(axis=1).gt(0).astype(int)
        result['skill_set_data_engineer'] = result[['skill_ETL', 'skill_資料倉儲']].sum(axis=1).gt(0).astype(int)
        result['skill_set_llm_image'] = result[['skill_llm','skill_影像處理']].sum(axis=1).gt(0).astype(int)
       
        return result
    
    def scale_company(self, num):
    
        if num <= 50:
            return 1
        if num <= 200:
            return 2
        return 3
    
    def label_encoding_education(self, x):
    
        if x == '高中以下':
            return 1
        if x == '專科':
            return 1
        if x == '高中':
            return 1
        if x == '大學':
            return 2
        if x == '碩士':
            return 3
        return 0

    def label_high_education(self, x):

        if x == '碩士':
            return 1
        return 0
    
    def get_industry_group(self, data): 
        
        result = data.copy()

        for i in self.industry_map.keys():
            col = 'industry_'+ self.industry_map[i]
            result[col] = 0 *len(result)
            self.industry_cols.append(col)
            
        for i, val in result.iterrows():
            industry_no = str(val['coIndustry'])[:7]
            
            if industry_no in self.industry_map.keys():
                col = 'industry_'+ self.industry_map[industry_no]
                result.loc[i, col] = 1
        
        return result
    
    def get_region(self, data): 
        
        result = data.copy()
        north = ['台北市', '新北市','基隆市', '桃園市', '新竹市','新竹縣']
        result['is_region_north'] = result['jobAddrNoDesc'].apply(lambda x: 1 if x[:3] in north else 0)

        return result
      