import pandas as pd
import os

# 定义列名字典
columns_dict = {
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': '降水总量，每小时（毫米）',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': '站点气压，每小时（毫巴）',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': '前一小时最高气压（自动记录）（毫巴）',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': '前一小时最低气压（自动记录）（毫巴）',
    'RADIACAO GLOBAL (KJ/m²)': '全球辐射（千焦耳/平方米）',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': '空气干球温度，每小时（摄氏度）',
    'TEMPERATURA DO PONTO DE ORVALHO (°C)': '露点温度（摄氏度）',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': '前一小时最高温度（自动记录）（摄氏度）',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': '前一小时最低温度（自动记录）（摄氏度）',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': '前一小时最高露点温度（自动记录）（摄氏度）',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': '前一小时最低露点温度（自动记录）（摄氏度）',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': '前一小时最高相对湿度（自动记录）（百分比）',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': '前一小时最低相对湿度（自动记录）（百分比）',
    'UMIDADE RELATIVA DO AR, HORARIA (%)': '每小时相对湿度（百分比）',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': '风向，每小时（度）',
    'VENTO, RAJADA MAXIMA (m/s)': '最大风速（米/秒）',
    'VENTO, VELOCIDADE HORARIA (m/s)': '每小时风速（米/秒）',
    'ESTACAO': '气象站名称'
}

# 定义archive目录路径
base_folder_path = "archieve(11)"
output_file_path = "combined_weather_data.csv"

# 如果输出文件已存在，则删除
if os.path.exists(output_file_path):
    os.remove(output_file_path)

# 遍历2001年到2020年的数据文件夹
for year in range(2001, 2021):
    folder_path = os.path.join(base_folder_path, f"weather_{year}")  # 文件夹路径
    file_name = f"weather_{year}.csv"  # CSV文件名
    file_path = os.path.join(folder_path, file_name)  # CSV文件路径
    
    # 逐块读取CSV文件
    chunks = pd.read_csv(file_path, chunksize=10000)
    for chunk in chunks:
        # 合并日期和时间列为一个时间戳列
        chunk['Timestamp'] = pd.to_datetime(chunk['DATA (YYYY-MM-DD)'] + ' ' + chunk['Hora UTC'], format='%Y-%m-%d %H:%M', errors='coerce')

        # 删除原始的日期和时间列
        chunk = chunk.drop(columns=['DATA (YYYY-MM-DD)', 'Hora UTC'])

        # 将列名字典应用于数据
        chunk = chunk.rename(columns=columns_dict)

        # 将当前块的数据添加到 CSV 文件中
        chunk.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)
    
    print(f"{year}年的数据已添加到总数据中")

print("所有年份的数据已保存到 combined_weather_data.csv 文件中")
