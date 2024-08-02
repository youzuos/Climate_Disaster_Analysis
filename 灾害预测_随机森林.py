import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取数据
disaster_model_data_filled = pd.read_csv('disaster_model_data_filled.csv')

# 将 storm_Count, massmovement_Count, flood_Count 转换为二分类变量
disaster_model_data_filled['flood_Count'] = disaster_model_data_filled['flood_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['storm_Count'] = disaster_model_data_filled['storm_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['drought_Count'] = disaster_model_data_filled['flood_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['fire_Miscellaneous_Count'] = disaster_model_data_filled['drought_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['earthquake_Count'] = disaster_model_data_filled['earthquake_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['extreme_temperature_Count'] = disaster_model_data_filled['extreme_temperature_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['water_Count'] = disaster_model_data_filled['water_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['volcanic_activity_Count'] = disaster_model_data_filled['volcanic_activity_Count'].apply(lambda x: 1 if x > 0 else 0)
disaster_model_data_filled['wildfire_Count'] = disaster_model_data_filled['wildfire_Count'].apply(lambda x: 1 if x > 0 else 0)

# 提取特征和目标变量
features = ['rain_max', 'rad_max', 'temp_avg', 'temp_max', 'temp_min', 
            'hum_max', 'hum_min', 'wind_max', 'wind_avg', '地面温度']
target_1 = 'flood_Count'
target_2 = 'storm_Count'
target_3 = 'drought_Count'
target_4 = 'fire_Miscellaneous_Count'
target_5 = 'earthquake_Count'
target_6 = 'extreme_temperature_Count'
target_7 = 'water_Count'
target_8 = 'volcanic_activity_Count'
target_9 = 'wildfire_Count'

# 标准化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(disaster_model_data_filled[features])

# 将目标变量与标准化的特征结合
scaled_data_1 = np.hstack((scaled_features, disaster_model_data_filled[[target_1]].values))
scaled_data_2 = np.hstack((scaled_features, disaster_model_data_filled[[target_2]].values))
scaled_data_3 = np.hstack((scaled_features, disaster_model_data_filled[[target_3]].values))
scaled_data_4 = np.hstack((scaled_features, disaster_model_data_filled[[target_4]].values))
scaled_data_5 = np.hstack((scaled_features, disaster_model_data_filled[[target_5]].values))
scaled_data_6 = np.hstack((scaled_features, disaster_model_data_filled[[target_6]].values))
scaled_data_7 = np.hstack((scaled_features, disaster_model_data_filled[[target_7]].values))
scaled_data_8 = np.hstack((scaled_features, disaster_model_data_filled[[target_8]].values))
scaled_data_9 = np.hstack((scaled_features, disaster_model_data_filled[[target_9]].values))

# 创建用于随机森林模型的输入和输出
X_1 = scaled_data_1[:, :-1]
y_1 = scaled_data_1[:, -1]
X_2 = scaled_data_2[:, :-1]
y_2 = scaled_data_2[:, -1]
X_3 = scaled_data_3[:, :-1]
y_3 = scaled_data_3[:, -1]
X_4 = scaled_data_4[:, :-1]
y_4 = scaled_data_4[:, -1]
X_5 = scaled_data_5[:, :-1]
y_5 = scaled_data_5[:, -1]
X_6 = scaled_data_6[:, :-1]
y_6 = scaled_data_6[:, -1]
X_7 = scaled_data_7[:, :-1]
y_7 = scaled_data_7[:, -1]
X_8 = scaled_data_8[:, :-1]
y_8 = scaled_data_8[:, -1]
X_9 = scaled_data_9[:, :-1]
y_9 = scaled_data_9[:, -1]

# 分割数据集
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=42)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4, y_4, test_size=0.2, random_state=42)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, y_5, test_size=0.2, random_state=42)
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_6, y_6, test_size=0.2, random_state=42)
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(X_7, y_7, test_size=0.2, random_state=42)
X_train_8, X_test_8, y_train_8, y_test_8 = train_test_split(X_8, y_8, test_size=0.2, random_state=42)
X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(X_9, y_9, test_size=0.2, random_state=42)

# 创建和训练随机森林模型
def create_and_train_rf(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

rf_model_1 = create_and_train_rf(X_train_1, y_train_1)
rf_model_2 = create_and_train_rf(X_train_2, y_train_2)
rf_model_3 = create_and_train_rf(X_train_3, y_train_3)
rf_model_4 = create_and_train_rf(X_train_4, y_train_4)
rf_model_5 = create_and_train_rf(X_train_5, y_train_5)
rf_model_6 = create_and_train_rf(X_train_6, y_train_6)
rf_model_7 = create_and_train_rf(X_train_7, y_train_7)
rf_model_8 = create_and_train_rf(X_train_8, y_train_8)
rf_model_9 = create_and_train_rf(X_train_9, y_train_9)

# 预测结果
y_pred_1 = rf_model_1.predict(X_test_1)
y_pred_2 = rf_model_2.predict(X_test_2)
y_pred_3 = rf_model_3.predict(X_test_3)
y_pred_4 = rf_model_4.predict(X_test_4)
y_pred_5 = rf_model_5.predict(X_test_5)
y_pred_6 = rf_model_6.predict(X_test_6)
y_pred_7 = rf_model_7.predict(X_test_7)
y_pred_8 = rf_model_8.predict(X_test_8)
y_pred_9 = rf_model_9.predict(X_test_9)

# 计算其他指标
accuracy_1 = accuracy_score(y_test_1, y_pred_1)
recall_1 = recall_score(y_test_1, y_pred_1)
precision_1 = precision_score(y_test_1, y_pred_1)
f1_1 = f1_score(y_test_1, y_pred_1)

accuracy_2 = accuracy_score(y_test_2, y_pred_2)
recall_2 = recall_score(y_test_2, y_pred_2)
precision_2 = precision_score(y_test_2, y_pred_2)
f1_2 = f1_score(y_test_2, y_pred_2)

accuracy_3 = accuracy_score(y_test_3, y_pred_3)
recall_3 = recall_score(y_test_3, y_pred_3)
precision_3 = precision_score(y_test_3, y_pred_3)
f1_3 = f1_score(y_test_3, y_pred_3)

accuracy_4 = accuracy_score(y_test_4, y_pred_4)
recall_4 = recall_score(y_test_4, y_pred_4)
precision_4 = precision_score(y_test_4, y_pred_4)
f1_4 = f1_score(y_test_4, y_pred_4)

accuracy_5 = accuracy_score(y_test_5, y_pred_5)
recall_5 = recall_score(y_test_5, y_pred_5)
precision_5 = precision_score(y_test_5, y_pred_5)
f1_5 = f1_score(y_test_5, y_pred_5)

accuracy_6 = accuracy_score(y_test_6, y_pred_6)
recall_6 = recall_score(y_test_6, y_pred_6)
precision_6 = precision_score(y_test_6, y_pred_6)
f1_6 = f1_score(y_test_6, y_pred_6)

accuracy_7 = accuracy_score(y_test_7, y_pred_7)
recall_7 = recall_score(y_test_7, y_pred_7)
precision_7 = precision_score(y_test_7, y_pred_7)
f1_7 = f1_score(y_test_7, y_pred_7)

accuracy_8 = accuracy_score(y_test_8, y_pred_8)
recall_8 = recall_score(y_test_8, y_pred_8)
precision_8 = precision_score(y_test_8, y_pred_8)
f1_8 = f1_score(y_test_8, y_pred_8)

accuracy_9 = accuracy_score(y_test_9, y_pred_9)
recall_9 = recall_score(y_test_9, y_pred_9)
precision_9 = precision_score(y_test_9, y_pred_9)
f1_9 = f1_score(y_test_9, y_pred_9)


# 提取特征和目标变量
features = ['rain_max', 'rad_max', 'temp_avg', 'temp_max', 'temp_min', 
            'hum_max', 'hum_min', 'wind_max', 'wind_avg', '地面温度']
targets = ['flood_Count', 'storm_Count', 'drought_Count', 'fire_Miscellaneous_Count', 
           'earthquake_Count', 'extreme_temperature_Count', 'water_Count', 
           'volcanic_activity_Count', 'wildfire_Count']

# 保存预测结果和评估指标的DataFrame
results = []

# 创建和训练随机森林模型
def create_and_train_rf(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# 预测并保存结果
for target in targets:
    X = scaled_features
    y = disaster_model_data_filled[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = create_and_train_rf(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append([target, accuracy, recall, precision, f1])
    
    # 保存预测结果到CSV文件
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(f'predictions_{target}.csv', index=False)
    
    # 绘制特征重要性图
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {target}")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'feature_importances_{target}.png')
    plt.show()

# 保存评估指标到CSV文件
results_df = pd.DataFrame(results, columns=['Target', 'Accuracy', 'Recall', 'Precision', 'F1 Score'])
results_df.to_csv('evaluation_metrics.csv', index=False)

print("结果已保存到CSV文件。")