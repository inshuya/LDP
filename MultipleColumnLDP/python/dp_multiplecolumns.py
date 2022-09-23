from dp_utils import *
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=3)

def gen_data(sample_size = 10):
    
    kept_count = int(sample_size/2)
    kept = [1] * kept_count
    row_count_kept = np.random.randint(2,10, size=kept_count)
    col_count_kept = np.random.randint(2,5, size=kept_count)
    
    rejected_count = sample_size - kept_count
    rejected = [0] * rejected_count
    row_count_rejected = np.random.randint(7,15, size=rejected_count)
    col_count_rejected = np.random.randint(4,10, size=rejected_count)
    
    is_kept = kept + rejected
    row_count = np.concatenate([row_count_kept, row_count_rejected])
    column_count = np.concatenate([col_count_kept, col_count_rejected])
    
    return [is_kept, row_count, column_count]


def calc_mean(df_wnoise, per_col_epsilon, row_count_range, column_count_range):
    print("**For all data**")
    print("Actual Row Count Mean: ", df_wnoise['RowCount'].mean())
    print("LDP Row Count Mean: ", estimate(df_wnoise['RowCount_dp'],row_count_range,per_col_epsilon))
    
    print("Actual Column Count Mean: ", df_wnoise['ColumnCount'].mean())
    print("LDP Column Count Mean: ", estimate(df_wnoise['ColumnCount_dp'], column_count_range, per_col_epsilon))
    
    print("\n\n**Where IsKept == 1**")
    kept_wnoise = df_wnoise.loc[df_wnoise['IsKept'] == 1]
    print("Actual Row Count Mean: ", kept_wnoise['RowCount'].mean())
    print("LDP Row Count Mean: ", estimate(kept_wnoise['RowCount_dp'], row_count_range,per_col_epsilon))
    
    print("Actual Column Count Mean: ", kept_wnoise['ColumnCount'].mean())
    print("LDP Column Count Mean: ", estimate(kept_wnoise['ColumnCount_dp'], column_count_range, per_col_epsilon))
    
    print("\n\n**Where IsKept == 0**")
    rejected_wnoise = df_wnoise.loc[df_wnoise['IsKept'] == 0]
    print("Actual Row Count Mean: ", rejected_wnoise['RowCount'].mean())
    print("LDP Row Count Mean: ", estimate(rejected_wnoise['RowCount_dp'], row_count_range,per_col_epsilon))
    
    print("Actual Column Count Mean: ", rejected_wnoise['ColumnCount'].mean())
    print("LDP Column Count Mean: ", estimate(rejected_wnoise['ColumnCount_dp'], column_count_range, per_col_epsilon))
    

def onebit_main(excel_sample, epsilon, no_cols_dp, column_count_range, row_count_range):  
    per_col_epsilon = epsilon/no_cols_dp
    
    excel_sample_wnoise = report_table(excel_sample, dp_cols = ['RowCount', 'ColumnCount'])
    
    calc_mean(excel_sample_wnoise, per_col_epsilon, row_count_range, column_count_range)
    

def dbit_main(excel_sample, epsilon, no_cols_dp, column_count_range, row_count_range):
    excel_row_col_map = pd.DataFrame(columns=['RowCount','ColumnCount','MapValue','row_col_bin'])
    map = 1
    bin_names = []
    for i in range(1,row_count_range+1):
        for j in range(1,column_count_range+1):
            bin_name = "R"+str(i)+"C"+str(j)
            bin_names += [bin_name]
            excel_row_col_map.loc[len(excel_row_col_map.index)] = [i, j, map, bin_name]
            map += 1

    excel_sample_whist = pd.merge(excel_sample,excel_row_col_map,how="inner",on=['RowCount','ColumnCount'])

    bins_col = []
    dp_bins_col = []
    for i in range(1, row_count_range*column_count_range + 1):
        col_name = "map_bin_"+str(i)
        dp_col_name = col_name+"_dp"
        bins_col += [col_name]
        dp_bins_col += [dp_col_name]
        excel_sample_whist[col_name] = np.where(excel_sample_whist['MapValue'] == i, 1, 0)

    excel_sample_whist_noise = report_dbit_dp(excel_sample_whist, bins_col)
    
    #all data
    plot_est_vs_actual(excel_sample_whist_noise, excel_sample_whist, dp_bins_col, bin_names, 0, "Suggestions accepted and rejected")
    
    #kept=1
    kept_wnoise = excel_sample_whist_noise.loc[excel_sample_whist_noise['IsKept'] == 1]
    kept = excel_sample_whist.loc[excel_sample_whist['IsKept'] == 1]
    plot_est_vs_actual(kept_wnoise, kept, dp_bins_col, bin_names, 1, "Suggestions accepted")
    
    #kept=0
    reject_wnoise = excel_sample_whist_noise.loc[excel_sample_whist_noise['IsKept'] == 0]
    reject = excel_sample_whist.loc[excel_sample_whist['IsKept'] == 0]
    plot_est_vs_actual(reject_wnoise, reject, dp_bins_col, bin_names, 2, "Suggestions rejected")
    
    plt.show()

def plot_est_vs_actual(sample_wnoise,sample, dp_bins_col, bin_names, pos, title):
    noisy_est = estimate_dbit_histogram(sample_wnoise, dp_bins_col, bin_names)
     
    actual_count = sample.groupby('row_col_bin')['row_col_bin'].count().reset_index(name="actual_count")

    noisy_est.columns = ["row_col_bin", "estimates"]
    
    actual_noisy_plot = pd.merge(
        actual_count,
        noisy_est,
        how="outer",
        on="row_col_bin")
    
    actual_noisy_plot['estimates'] = pd.to_numeric(actual_noisy_plot['estimates'])
    actual_noisy_plot['actual_count'] = actual_noisy_plot['actual_count'].fillna(0)
    
    
    actual_noisy_plot = actual_noisy_plot.sort_values(by = "actual_count", ascending = True)
    result = actual_noisy_plot.plot.barh(x="row_col_bin", y=["actual_count", "estimates"], figsize=(60, 60), ax=axes[pos])
    result.set_ylabel('RowColumnCount',fontdict={'fontsize':8})
    result.set_xlabel('RowColumnCount',fontdict={'fontsize':8})
    result.set_title(title)
    

if __name__ == "__main__":
    sample_size = 10000
    epsilon = 4
    no_cols_dp = 2
    column_count_range = 10
    row_count_range = 15

    data = gen_data(sample_size)
    excel_sample = pd.DataFrame({'IsKept':data[0],'RowCount': data[1], 'ColumnCount': data[2]})

    #onebit_main(excel_sample, epsilon, no_cols_dp, column_count_range, row_count_range)
    dbit_main(excel_sample, epsilon, no_cols_dp, column_count_range, row_count_range)