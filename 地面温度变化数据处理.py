import pandas as pd
from datetime import datetime, timedelta

# 读取CSV文件
df = pd.read_csv('地面温度变化.csv')

# 过滤出年份为2001至2020的行
df = df[df.iloc[:, 8].isin(range(2001, 2021))]

# 定义月份到数字的映射
month_to_num = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

# 初始化一个空的DataFrame来存储展开后的数据
expanded_df = pd.DataFrame()

# 遍历筛选后的数据中的每一行，展开为小时数据
for _, row in df.iterrows():
    year = int(row[df.columns[8]])  # 假设年份在第九列（从0开始索引，即索引8）
    month_name = row[df.columns[7]]  # 假设月份在第八列（从0开始索引，即索引7）
    month = month_to_num[month_name]
    
    # 获取当前月的天数
    if month == 12:
        num_days = 31
    else:
        num_days = (datetime(year, month + 1, 1) - timedelta(days=1)).day
    
    # 生成每小时的数据
    for day in range(1, num_days + 1):
        for hour in range(24):
            timestamp = datetime(year, month, day, hour)
            hourly_data = {
                'Year': year,
                'Month': month,
                'Day': day,
                'Hour': hour,
                'Timestamp': timestamp
            }
            # 添加其他列的数据
            for col in df.columns:
                if col not in [df.columns[8], df.columns[7]]:  # 排除年份和月份列
                    hourly_data[col] = row[col]
            expanded_df = expanded_df._append(hourly_data, ignore_index=True)

# 保存展开后的数据到新的CSV文件
output_filename = "expanded_地面温度变化_2001_2020.csv"

# 将 "Timestamp" 列从第五列移动到第一列
timestamp_column = expanded_df['Timestamp']
expanded_df = expanded_df.drop(columns=['Timestamp'])
expanded_df.insert(0, 'Timestamp', timestamp_column)

expanded_df.to_csv(output_filename, index=False)

print(f"数据已成功展开并保存到 {output_filename}")

# 查看数据框的前五行
print(expanded_df.head())

# 查看数据框的概要信息
print(expanded_df.info())
