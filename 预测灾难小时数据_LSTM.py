import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# 读取 CSV 文件
file_path = '2001_2020.csv'
data = pd.read_csv(file_path)

# 定义特征列
features = ['降水总量，每小时（毫米）', '全球辐射（千焦耳/平方米）', '空气干球温度，每小时（摄氏度）', '前一小时最高温度（自动记录）（摄氏度）', '前一小时最低温度（自动记录）（摄氏度）', 
            '前一小时最高相对湿度（自动记录）（百分比）', '前一小时最低相对湿度（自动记录）（百分比）', '最大风速（米/秒）', '每小时风速（米/秒）', '地面温度变化']

# 提取特征数据
feature_data = data[features]

# 检查是否有缺失值
if feature_data.isnull().values.any():
    print("数据中存在缺失值，正在处理缺失值...")

    # 填充缺失值（此处使用均值填充，您可以根据需要选择其他方法）
    feature_data = feature_data.fillna(feature_data.mean())

# 标准化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(feature_data)

# 创建用于LSTM的序列数据
def create_sequences(data, seq_length):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
    return np.array(X)

# 设置序列长度
seq_length = 10

# 创建序列
X = create_sequences(scaled_features, seq_length)

# 加载模型
model_flood = load_model('lstm_model_flood.h5')
model_storm = load_model('lstm_model_storm.h5')
model_drought = load_model('lstm_model_drought.h5')

model_fire_Miscellaneous = load_model('lstm_model_fire_Miscellaneous.h5')
model_earthquake = load_model('lstm_model_earthquake.h5')
model_extreme_temperature = load_model('lstm_model_extreme_temperature.h5')

model_water = load_model('lstm_model_water.h5')
model_volcanic_activity = load_model('lstm_model_volcanic_activity.h5')
model_wildfire = load_model('lstm_model_wildfire.h5')

# 进行预测
pred_flood = model_flood.predict(X)
pred_storm = model_storm.predict(X)
pred_drought = model_drought.predict(X)

pred_fire_Miscellaneous = model_fire_Miscellaneous.predict(X)
pred_earthquake = model_earthquake.predict(X)
pred_extreme_temperature = model_extreme_temperature.predict(X)

pred_water = model_water.predict(X)
pred_volcanic_activity = model_volcanic_activity.predict(X)
pred_wildfire = model_wildfire.predict(X)

# 二分类处理，使用适当的阈值
threshold = 0.5
pred_flood = (pred_flood >= threshold).astype(int)
pred_storm = (pred_storm >= threshold).astype(int)
pred_drought = (pred_drought >= threshold).astype(int)

pred_fire_Miscellaneous = (pred_fire_Miscellaneous >= threshold).astype(int)
pred_earthquake = (pred_earthquake >= threshold).astype(int)
pred_extreme_temperature = (pred_extreme_temperature >= threshold).astype(int)

pred_water = (pred_water >= threshold).astype(int)
pred_volcanic_activity = (pred_volcanic_activity >= threshold).astype(int)
pred_wildfire = (pred_wildfire >= threshold).astype(int)

# 创建一个新的 DataFrame 存储预测结果
predictions = pd.DataFrame({
    'flood_Count': np.concatenate([np.full(seq_length, np.nan), pred_flood.flatten()]),
    'storm_Count': np.concatenate([np.full(seq_length, np.nan), pred_storm.flatten()]),
    'drought_Count': np.concatenate([np.full(seq_length, np.nan), pred_drought.flatten()]),
    'fire_Miscellaneous_Count': np.concatenate([np.full(seq_length, np.nan), pred_fire_Miscellaneous.flatten()]),
    'earthquake_Count': np.concatenate([np.full(seq_length, np.nan), pred_earthquake.flatten()]),
    'extreme_temperature_Count': np.concatenate([np.full(seq_length, np.nan), pred_extreme_temperature.flatten()]),
    'water_Count': np.concatenate([np.full(seq_length, np.nan), pred_water.flatten()]),
    'volcanic_activity_Count': np.concatenate([np.full(seq_length, np.nan), pred_volcanic_activity.flatten()]),
    'wildfire_Count': np.concatenate([np.full(seq_length, np.nan), pred_wildfire.flatten()])
})

# 将预测结果与原数据合并
result = pd.concat([data, predictions], axis=1)

# 保存结果到新的 CSV 文件
result.to_csv('predictions_2001_2020_LSTM.csv', index=False)

print("预测完成并已保存到 'predictions_2001_2020_LSTM.csv'")