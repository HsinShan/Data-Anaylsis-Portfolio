import pandas as pd
import jieba
import matplotlib.pyplot as plt
import re
import json
from collections import Counter
import spacy
from spacy.symbols import ORTH
from nltk.stem import PorterStemmer
import numpy as np


class term_processor:

    def __init__(self, df):

        self.nlp_en = spacy.load("en_core_web_sm") # 載入英文模型

        # 讀取英文 library
        en_dict = self.load_files("data_dict_en.txt")
        for en_word in en_dict: # 載入英文自訂 library
            self.nlp_en.tokenizer.add_special_case(en_word, [{ORTH: en_word}])

        self.nlp_ch = jieba
        jieba.load_userdict("data_dict.txt") # 引入中文字典

        self.stemmer = PorterStemmer() # 英文的 stemming
        self.stopwords = self.load_files("stopwords.txt") # 讀取中文停用詞
        self.df = df.copy()

        with open('synonym.json', 'r', encoding='utf-8') as f:
            self.synonym_dict = json.load(f) # 讀取同義字字典


    def run(self):
        temp = self.df.copy()

        # 移除標點符號符號並將英文都轉成小寫
        temp['text_normalized'] = temp['descWithoutHighlight'].str.replace(r'[^\w\u4e00-\u9fff]|_', ' ', regex=True).str.lower()
        temp['terms'] = temp['text_normalized'].apply(lambda x: self.tokenize_and_process(x))

        terms = self.get_term_df(temp) # 取得所有 terms
        terms = self.get_term_freq(temp, terms) # 取得 tfidf

        return temp, terms


    def load_files(self, file_path):
        words = []
        with open(file_path, "r", encoding="utf-8") as file:
            words = [line.strip() for line in file.readlines()]
        return words

    def detect_lang(self, text):
        hasChinese = re.search(r'[\u4e00-\u9fff]', text)
        # 只要有中文字就用中文的詞庫
        if hasChinese:
            return 'Chinese'
        else:
            return 'English'

    def tokenize_and_process(self, text):
        lang = self.detect_lang(text)

        if lang == 'Chinese':
            words = list(self.nlp_ch.cut(text))
        else:
            words = list(self.nlp_en(text))

        terms = []

        # 去除停用詞與單個字，並去除空格
        for w in words:
            word = str(w).strip()
            word = self.stemmer.stem(word) # 主要用於英文的 stemming
            word = self.synonym_dict[word] if word in self.synonym_dict.keys() else word # 中英文同義字轉換
            
            if word.isdigit(): #移除純數字
                continue
            if word not in self.stopwords and len(word) > 1:
                terms.append(word)

        return terms


    def get_term_df(self, df):

        terms_df = pd.DataFrame.from_dict(Counter(list(sum(df['terms'], []))), orient='index', columns=['count']).sort_values(by='count', ascending=False).reset_index()
        terms_df.columns = ['terms', 'count']

        terms_df = terms_df[terms_df['count'] > 2] # 排除僅出現 2 次的字

        return terms_df


    def get_term_freq(self, data, term_df):

        N = len(data['text_normalized'])
        tf = [] # 計算這個 term 在所有文本中出現的次數
        df = [] # 計算出現在幾篇職缺中

        temp = data.copy()
        temp_term_df = term_df.copy()
        terms = list(term_df['terms'])

        for term in terms:
            text = str(term)

            temp['freq'] = temp['terms'].apply(lambda x: x.count(text))
            temp['df'] = temp['freq'].apply(lambda x: 1 if x > 0 else 0)

            # group_df = temp.groupby(by='job_category').sum()[['df', 'freq']]
            # freq = list(result['terms'].apply(lambda x: x.count(text)))
            term_tf = sum(temp['freq'])
            term_df = sum(temp['df'])
            tf.append(term_tf)
            df.append(term_df)


        temp_term_df['TF'] = tf
        temp_term_df['DF'] = df
        temp_term_df['iDF'] = np.log(N/temp_term_df['DF'])
        temp_term_df['TFIDF'] = temp_term_df['TF'] * temp_term_df['iDF']
        temp_term_df['TFDF'] = temp_term_df['TF'] * temp_term_df['DF']

        return temp_term_df
