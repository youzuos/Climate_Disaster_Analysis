import dask.dataframe as dd
import pandas as pd

def read_and_process_file(file_path, usecols=None, rename_cols=None, drop_duplicates_col='Timestamp'):
    # 使用dask读取文件
    df = dd.read_csv(file_path, usecols=usecols)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    if drop_duplicates_col in df.columns:
        df = df.drop_duplicates(subset=drop_duplicates_col)
    else:
        print(f"Warning: {file_path} is missing the column '{drop_duplicates_col}'")
    return df

# 获取所有列名
def get_all_columns(file_path):
    all_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    return all_columns

# 定义文件路径和处理参数
file_info = [
    ("combined_weather_data.csv", get_all_columns("combined_weather_data.csv"), None),
    ("expanded_大豆面积_2001_2020.csv", ['Timestamp', '值'], {'值': "大豆面积"}),
    ("expanded_有机土壤排放_2001_2020.csv", ['Timestamp', '值'], {'值': "有机土壤排放"}),
    ("expanded_亚马逊森林退化_2001_2020.csv", ['Timestamp', 'AMZ LEGAL'], {'AMZ LEGAL': "亚马逊森林面积"}),
    ("expanded_地面温度变化_2001_2020.csv", ['Timestamp', '值'], {'值': "地面温度变化"}),
    ("US Soybeans Futures Hourly Data 2001_2020.csv", ['Timestamp', 'Change %'], None)
]

# 读取和处理每个文件
dfs = []
for file_path, usecols, rename_cols in file_info:
    df = read_and_process_file(file_path, usecols, rename_cols)
    if 'Timestamp' not in df.columns:
        raise ValueError(f"Error: {file_path} is missing the column 'Timestamp'")
    dfs.append(df)

# 合并数据帧
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = dd.merge(merged_df, df, on='Timestamp', how='left')

# 将 Dask DataFrame 转换为 Pandas DataFrame
merged_df = merged_df.compute()

# 将结果保存到新的 CSV 文件中
merged_df.to_csv("2001_2020.csv", index=False)

print("合并后的数据前五行：")
print(merged_df.head())

print("合并后的数据基本信息：")
print(merged_df.info())

# 输出完整的列名
print("完整的列名：")
print(merged_df.columns.tolist())
