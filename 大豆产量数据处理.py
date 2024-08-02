import pandas as pd

# 读取 production.csv 文件
df = pd.read_csv('production.csv')

# 只保留“年份”和“值”两列，并重命名
df = df[['年份', '值']]
df.columns = ['Timestamp', '产量']

# 过滤出2001年到2020年的数据
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y')
df = df[df['Timestamp'].dt.year.isin(range(2001, 2021))]

# 创建一个包含2001年到2020年所有日期的DataFrame，按小时频率
date_range = pd.date_range(start='2001-01-01', end='2020-12-31 23:00:00', freq='H')
expanded_df = pd.DataFrame(date_range, columns=['Timestamp'])

# 将每年的产量值填充到对应的日期和小时
for year in range(2001, 2021):
    # 获取当前年份的产量值
    year_production = df[df['Timestamp'].dt.year == year]['产量'].values
    if year_production:
        production_value = year_production[0]
    else:
        production_value = 0  # 若无当前年份数据，默认产量为0
    
    # 将产量值填充到当前年份的所有日期和小时
    expanded_df.loc[expanded_df['Timestamp'].dt.year == year, '产量'] = production_value

# 保存到新的 CSV 文件中
expanded_df.to_csv('expanded_production_2001_2020_hourly.csv', index=False)

print(expanded_df)
print(expanded_df.info())
