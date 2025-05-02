import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind

def t_test(df, col_name, X, y, threshold):
    
    t_test_results = []

    for x in X:
        groups = [
            df[df[col_name]==x][y], 
            df[df[col_name]!=x][y]
        ]

         # 若樣本數不足就跳過
        if len(groups[0]) < 10 or len(groups[1]) < 10:
            continue

         # 做 Welch's t-test（不假設變異數相等）
        t_stat, p_val = ttest_ind(groups[0], groups[1], equal_var=False)
        diff = groups[0].mean() - groups[1].mean()

        t_test_results.append({
            f'{col_name}': x,
            f'mean {y}_with': groups[0].mean(),
            f'mean {y}_not_with': groups[1].mean(),
            'mean_diff': diff,
            'p_value': p_val
        })

    # 整理成 DataFrame 並排序
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df = t_test_df[t_test_df['p_value'] <= threshold].sort_values(by='p_value')
    
    return t_test_df

def t_test_binary_col(df, X, y, threshold):
    
    t_test_results = []

    for x in X:
        groups = [
            df[df[x]==1][y], 
            df[df[x]==0][y]
        ]

         # 若樣本數不足就跳過
        if len(groups[0]) < 10 or len(groups[1]) < 10:
            continue

         # 做 Welch's t-test（不假設變異數相等）
        t_stat, p_val = ttest_ind(groups[0], groups[1], equal_var=False)
        diff = groups[0].mean() - groups[1].mean()

        t_test_results.append({
            'name': x,
            f'mean {y}_with': groups[0].mean(),
            f'mean {y}_not_with': groups[1].mean(),
            'mean_diff': diff,
            'p_value': p_val
        })

    # 整理成 DataFrame 並排序
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df = t_test_df[t_test_df['p_value'] <= threshold].sort_values(by='p_value')
    
    return t_test_df

def anova(df, X, y):
    
    group_df = df.groupby(by=[X]).agg({
        y: 'mean'
    }).sort_values(by=y, ascending=False)
    groups = [df[df[X] == idx][y] for idx, temp in group_df.iterrows()]

    f_stats, p_vals = f_oneway(*groups)
    print(f'F統計量: {f_stats:.2f}')
    print(f'p-value: {p_vals:.4f}')
    tukey = pairwise_tukeyhsd(endog=df[y], groups=df[X], alpha=0.05)
    print(tukey)