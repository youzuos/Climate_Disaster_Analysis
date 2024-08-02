import pandas as pd

# 读取CSV文件
df = pd.read_csv('大豆面积.csv')

# 过滤出年份为2001至2020的行
df = df[df.iloc[:, 8].isin(range(2001, 2021))]

# 提取年份列和其他数据列
years = df.iloc[:, 8]
other_columns = df.drop(df.columns[8], axis=1)

# 初始化空DataFrame来存储扩展后的数据
expanded_df = pd.DataFrame()

# 处理2001年至2020年的数据
for year in range(2001, 2021):
    # 生成当前年份的每小时时间戳
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='H')
    
    # 创建一个DataFrame来存储当前年份的扩展数据
    temp_df = pd.DataFrame(date_range, columns=['Datetime'])
    
    # 添加年份列
    temp_df['Year'] = year
    
    # 添加其他数据列
    for col in other_columns.columns:
        temp_df[col] = df.loc[years == year, col].values[0]
    
    # 将当前年份的扩展数据添加到总的DataFrame中
    expanded_df = pd.concat([expanded_df, temp_df])

# 重置索引
expanded_df.reset_index(drop=True, inplace=True)
expanded_df = expanded_df.rename(columns={expanded_df.columns[0]: "Timestamp"})

# 保存结果到新的CSV文件
expanded_df.to_csv('expanded_大豆面积_2001_2020.csv', index=False)

print("年度数据已扩展为小时数据，并保存到 'expanded_大豆面积_2001_2020.csv'")
print(expanded_df)