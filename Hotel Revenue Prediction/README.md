# 飯店營收預測
碩一上 機器學習技法期末專案

## Objective
本專案希望利用現有的飯店訂房資料去預測每天的營收,將採用 Logistic regression、Linear regression 以及 XGBoost 演算法,試驗三種不同的預測方式，最後會比較所有方法並推薦較佳的方法。

## Methodlogy
我們會試驗三種方法:
- 方法一:直接預測 label
- 方法二:計算每日非取消訂單佔每日所有訂單的營收比例平均值作為對於每日營收高估的調整比例,再預測每筆訂單的 ADR 並且計算每日的營收後,將每日的營收依照算出的比例進行調整。
- 方法三:先預測測試資料的訂單是否是取消訂單,再預測每筆訂單的 ADR 並且計算每日的營收,最後將營收以萬為單位紀錄,得到預測的 label。

## Result
我們推薦使用方法三用來預測 revenue,因為以預測結果而言,方法三的結果是最為準確的。以下為方法三的優劣分析:
#### 優點:
1. 透過個別預測訂單是否被取消及訂單的 ADR,可以更準確估計 daily revenue
2. 方法三所使用的 Package XGBoost 原理就是 gradient boosting decision tree,通常 gradient boosting 的準確度較高,但是計算速度會比較慢。但因 XGBoost 有做 Parallelization、Distributed Computing、Out-of-Core
Computing 的設計,所以在於計算速度上相對於其他 gradient boosting decision tree 的 package 快,且模型表現也不錯
#### 缺點:
1. 因為此方法對每筆訂單同時做了是否會被取消及 ADR 的預測,但 ADR 的準確度並不高,因而也連帶影響最後計算 daily revenue label 的準確度
2. 因為資料前處理對類別特徵做 dummy 的轉換,造成 feature sparse,會降低 learning 的效率

## 相關文件
[完整專題文檔](https://drive.google.com/file/d/1YfU7ww1j9_Lg1fuZooSDS_BYYX3BZen9/view?usp=sharing)