import pandas as pd

# 读取CSV文件
df = pd.read_csv("US Soybeans Futures Historical Data.csv")

# 确保时间列是字符串类型
df['Date'] = df['Date'].astype(str)

# 将时间列转换为日期时间格式
df['Timestamp'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# 提取2001年至2020年的数据
df_2001_2020 = df[(df['Timestamp'].dt.year >= 2001) & (df['Timestamp'].dt.year <= 2020)]

# 初始化一个空的DataFrame来存储展开后的数据
expanded_df = pd.DataFrame()

# 遍历2001年至2020年的数据
for year in range(2001, 2021):
    # 提取当前年份的数据
    df_year = df_2001_2020[df_2001_2020['Timestamp'].dt.year == year]
    
    # 生成当前年份的所有小时数据范围
    all_hours_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='H')
    
    # 将当前年份的数据重新索引到所有小时数据上，并用前一个有效值填充缺失数据
    df_year = df_year.set_index('Timestamp').reindex(all_hours_year, method='ffill')
    
    # 将索引转换为所需的字符串格式
    df_year.index = df_year.index.strftime('%Y-%m-%d %H:00:00')
    
    # 重置索引并重命名列
    df_year.reset_index(inplace=True)
    df_year.rename(columns={'index': 'Timestamp'}, inplace=True)
    
    # 将当前年份的数据添加到展开后的数据中
    expanded_df = pd.concat([expanded_df, df_year])

# 保存到新的CSV文件
expanded_df.to_csv("US Soybeans Futures Hourly Data 2001_2020.csv", index=False)

print("2001年至2020年的数据转换和补全完成，并已保存到 'US Soybeans Futures Hourly Data 2001_2020.csv' 文件中。")
