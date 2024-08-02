import pandas as pd

# 读取CSV文件
file_path = "亚马逊森林退化.csv"
data = pd.read_csv(file_path)

# 将年份列转换为整数
data['Ano/Estados'] = data['Ano/Estados'].astype(int)

# 提取年份为2001至2020年的数据
year_2001_2020_data = data[data['Ano/Estados'].isin(range(2001, 2021))]

if year_2001_2020_data.empty:
    print("No data found for the years 2001 to 2020.")
else:
    print(year_2001_2020_data)


# 展开成小时数据
hourly_data = pd.DataFrame()

# 每年有8760小时（365天 * 24小时）
hours_in_year = 8760

# 检查是否有2001年至2020年的数据
if not year_2001_2020_data.empty:
    for index, row in year_2001_2020_data.iterrows():
        year = int(row['Ano/Estados'])  # 确保年份是整数形式
        for hour in range(hours_in_year):
            new_row = row.copy()
            new_row['Ano/Estados'] = pd.Timestamp(year, 1, 1) + pd.Timedelta(hours=hour)
            hourly_data = hourly_data._append(new_row, ignore_index=True)
    hourly_data = hourly_data.rename(columns={hourly_data.columns[0]: "Timestamp"})
    # 打印结果
    print(hourly_data)
    
    # 将结果保存到新的CSV文件
    hourly_data.to_csv("expanded_亚马逊森林退化_2001_2020.csv", index=False)
else:
    print("No data found for the years 2001 to 2020.")
