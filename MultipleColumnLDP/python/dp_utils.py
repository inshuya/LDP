from typing import List
import numpy as np
import pandas as pd

from statsmodels.stats.power import TTestIndPower
from math import exp
import scipy.stats
from tqdm import tqdm


def report(val, m, epsilon):
    p = 1/(np.exp(epsilon) + 1) + (val/m) * ((np.exp(epsilon) - 1)/(np.exp(epsilon) + 1))
    r = np.random.uniform()
    return 1 if r < p else 0

def estimate(reports, m, epsilon):
    return m * np.mean(
        [((b * (np.exp(epsilon) + 1)) - 1 )
        / (np.exp(epsilon) - 1) 
        for b in reports])
    
def report_table(
    df: pd.DataFrame, 
    dp_cols: List[str], 
    total_epsilon_budget: float = 4, 
    suffix: str = '_dp'
):
    df_dp = df.copy()
    
    # Assume Epsilon budget is equally shared among all columns
    shared_epsilon = total_epsilon_budget/len(dp_cols)
    
    for col in dp_cols:
        df_dp[col+suffix] = [report(val, m=df_dp[col].max(), epsilon=shared_epsilon) for val in df_dp[col]]
        
    return df_dp


def report_dbit_dp(
    df, 
    bins_col, 
    epsilon = 4, 
    suffix = '_dp'
):
    df_dp = df.copy()
    for col in bins_col:
        #Since each histogram bin takes value either {0,1} so sensitivity = range = 1
        df_dp[col+suffix] = [report(val, m=1, epsilon=epsilon) for val in df_dp[col]]
    return df_dp

def estimate_dbit_histogram(
    df, 
    dp_bins_col, 
    bins_midpoint = None, 
    epsilon = 4 
): 
    dp_count_estimate = [estimate(df[col], m=1, epsilon=epsilon)*len(df) for col in dp_bins_col]
    noisy_count = pd.DataFrame(
        np.vstack([bins_midpoint, dp_count_estimate]).T, 
        columns = ['bins_midpoint', f'Count_NonDP_epsilon{epsilon}_n{len(df)}']
    ).sort_values('bins_midpoint')
    
    return noisy_count    

def sample_data(df, n, replace:bool=False):
    if (n<len(df)) or (replace==True):
        sample_index = np.random.choice([n for n in range(len(df))], size=n, replace=replace)
        df_sample = df.iloc[sample_index]
    
    else: 
        df_sample = df
        
    return df_sample


def dp_z_test_of_proportion(
    input_df, 
    test_col, 
    test_indicator_col, 
    m, 
    epsilon, 
    d0: float = 0
):  
    control_mean = input_df.loc[input_df[test_indicator_col]==0][test_col].mean()
    control_n = len(input_df.loc[input_df[test_indicator_col]==0])

    test_mean = input_df.loc[input_df[test_indicator_col]==1][test_col].mean()
    test_n = len(input_df.loc[input_df[test_indicator_col]==1])

    meandiff = test_mean - control_mean
    meandiff_se = np.sqrt(test_mean * (1 - test_mean)/(test_n) + control_mean * (1-control_mean)/(control_n))
    dp_h0 = (d0/m)*(exp(epsilon)-1)/(exp(epsilon)+1)
    
    test_statistics = (meandiff - dp_h0)/meandiff_se
    p_value = scipy.stats.norm.sf(abs(test_statistics))*2
    
    return (test_statistics, p_value)

def z_test(
    input_df, 
    test_col, 
    test_indicator_col, 
    d0: float = 0
):  
    control_mean = input_df.loc[input_df[test_indicator_col]==0][test_col].mean()
    control_s = input_df.loc[input_df[test_indicator_col]==0][test_col].std()
    control_n = len(input_df.loc[input_df[test_indicator_col]==0])

    test_mean = input_df.loc[input_df[test_indicator_col]==1][test_col].mean()
    test_s = input_df.loc[input_df[test_indicator_col]==1][test_col].std()
    test_n = len(input_df.loc[input_df[test_indicator_col]==1])

    meandiff = test_mean - control_mean
    meandiff_se = np.sqrt(test_s**2/(test_n) + control_s**2/(control_n))

    test_statistics = (meandiff - d0)/meandiff_se
    p_value = scipy.stats.norm.sf(abs(test_statistics))*2
    
    return (test_statistics, p_value)


def simulate_dp_ztest(    
    simulation_df,
    col,
    epsilon,
    sample_n,
    bootstrap_n,
    d0 = 0,
    lift_pct = 0.1
):
    dp_test = []
    nondp_test = []

    for _ in tqdm(range(bootstrap_n)): 
        ab_test_df2 = sample_data(simulation_df[[col]].copy(), n=sample_n)
        ab_test_df2['test'] = np.random.randint(0, 2, size=len(ab_test_df2))
        
        if lift_pct>0:
            lift = np.min(int(ab_test_df2[col].mean()*lift_pct), 0)
        else:
            lift = 0
            
        ab_test_df2.loc[ab_test_df2.test==1, col] = ab_test_df2.loc[ab_test_df2.test==1, col] + lift

        m = ab_test_df2[col].max()
        ab_test_df2[col+'_dp'] = [report(val, m=m, epsilon=epsilon) for val in ab_test_df2[col]]
        
        dp_test.append(dp_z_test_of_proportion(ab_test_df2, test_col=col+'_dp', test_indicator_col='test', d0=d0, m=m, epsilon=epsilon))
        nondp_test.append(z_test(ab_test_df2, test_col=col, test_indicator_col='test', d0=d0))

    test_stat_dp = [x[0] for x in dp_test]
    pvalue_dp = [x[1] for x in dp_test]
    test_stat_nondp = [x[0] for x in nondp_test]
    pvalue_nondp = [x[1] for x in nondp_test]

    simulation_result = pd.DataFrame(
        np.vstack([test_stat_dp, pvalue_dp, test_stat_nondp, pvalue_nondp]).T, 
        columns = ['stat_dp', 'p_dp', 'stat_nondp', 'p_nondp']
    )
    
    return (simulation_result, lift)