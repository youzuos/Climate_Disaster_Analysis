import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
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

# 创建用于全连接神经网络的输入和输出
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
split_size_1 = int(0.8 * len(X_1))
split_size_2 = int(0.8 * len(X_2))
split_size_3 = int(0.8 * len(X_3))
split_size_4 = int(0.8 * len(X_4))
split_size_5 = int(0.8 * len(X_5))
split_size_6 = int(0.8 * len(X_6))
split_size_7 = int(0.8 * len(X_7))
split_size_8 = int(0.8 * len(X_8))
split_size_9 = int(0.8 * len(X_9))

X_train_1, X_test_1 = X_1[:split_size_1], X_1[split_size_1:]
y_train_1, y_test_1 = y_1[:split_size_1], y_1[split_size_1:]
X_train_2, X_test_2 = X_2[:split_size_2], X_2[split_size_2:]
y_train_2, y_test_2 = y_2[:split_size_2], y_2[split_size_2:]
X_train_3, X_test_3 = X_3[:split_size_3], X_3[split_size_3:]
y_train_3, y_test_3 = y_3[:split_size_3], y_3[split_size_3:]

X_train_4, X_test_4 = X_4[:split_size_4], X_4[split_size_4:]
y_train_4, y_test_4 = y_4[:split_size_4], y_4[split_size_4:]
X_train_5, X_test_5 = X_5[:split_size_5], X_5[split_size_5:]
y_train_5, y_test_5 = y_5[:split_size_5], y_5[split_size_5:]
X_train_6, X_test_6 = X_6[:split_size_6], X_6[split_size_6:]
y_train_6, y_test_6 = y_6[:split_size_6], y_6[split_size_6:]

X_train_7, X_test_7 = X_7[:split_size_7], X_7[split_size_7:]
y_train_7, y_test_7 = y_7[:split_size_7], y_7[split_size_7:]
X_train_8, X_test_8 = X_8[:split_size_8], X_8[split_size_8:]
y_train_8, y_test_8 = y_8[:split_size_8], y_8[split_size_8:]
X_train_9, X_test_9 = X_9[:split_size_9], X_9[split_size_9:]
y_train_9, y_test_9 = y_9[:split_size_9], y_9[split_size_9:]

# 创建和编译模型
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # 使用sigmoid激活函数用于二分类
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 使用binary_crossentropy损失函数
    return model

input_shape = (len(features),)

model_1 = create_model(input_shape)
model_2 = create_model(input_shape)
model_3 = create_model(input_shape)
model_4 = create_model(input_shape)
model_5 = create_model(input_shape)
model_6 = create_model(input_shape)
model_7 = create_model(input_shape)
model_8 = create_model(input_shape)
model_9 = create_model(input_shape)

# 早停法
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 训练模型并保存历史记录
history_1 = model_1.fit(X_train_1, y_train_1, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_2 = model_2.fit(X_train_2, y_train_2, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_3 = model_3.fit(X_train_3, y_train_3, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_4 = model_4.fit(X_train_4, y_train_4, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_5 = model_5.fit(X_train_5, y_train_5, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_6 = model_6.fit(X_train_6, y_train_6, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_7 = model_7.fit(X_train_7, y_train_7, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_8 = model_8.fit(X_train_8, y_train_8, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history_9 = model_9.fit(X_train_9, y_train_9, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 评估模型
metrics_1 = model_1.evaluate(X_test_1, y_test_1)
metrics_2 = model_2.evaluate(X_test_2, y_test_2)
metrics_3 = model_3.evaluate(X_test_3, y_test_3)
metrics_4 = model_4.evaluate(X_test_4, y_test_4)
metrics_5 = model_5.evaluate(X_test_5, y_test_5)
metrics_6 = model_6.evaluate(X_test_6, y_test_6)
metrics_7 = model_7.evaluate(X_test_7, y_test_7)
metrics_8 = model_8.evaluate(X_test_8, y_test_8)
metrics_9 = model_9.evaluate(X_test_9, y_test_9)

# 预测结果
y_pred_1 = (model_1.predict(X_test_1) > 0.5).astype(int)
y_pred_2 = (model_2.predict(X_test_2) > 0.5).astype(int)
y_pred_3 = (model_3.predict(X_test_3) > 0.5).astype(int)
y_pred_4 = (model_4.predict(X_test_4) > 0.5).astype(int)
y_pred_5 = (model_5.predict(X_test_5) > 0.5).astype(int)
y_pred_6 = (model_6.predict(X_test_6) > 0.5).astype(int)
y_pred_7 = (model_7.predict(X_test_7) > 0.5).astype(int)
y_pred_8 = (model_8.predict(X_test_8) > 0.5).astype(int)
y_pred_9 = (model_9.predict(X_test_9) > 0.5).astype(int)

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

# 保存模型
model_1.save('fcnn_model_flood.h5')
model_2.save('fcnn_model_storm.h5')
model_3.save('fcnn_model_drought.h5')
model_4.save('fcnn_model_fire_Miscellaneous.h5')
model_5.save('fcnn_model_earthquake.h5')
model_6.save('fcnn_model_extreme_temperature.h5')
model_7.save('fcnn_model_water.h5')
model_8.save('fcnn_model_volcanic_activity.h5')
model_9.save('fcnn_model_wildfire.h5')

# 绘制训练和验证损失曲线
plt.figure(figsize=(20, 15))

for i, history in enumerate([history_1, history_2, history_3, history_4, history_5, history_6, history_7, history_8, history_9], 1):
    plt.subplot(3, 3, i)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Model {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()
# 将评估指标保存到 CSV 文件
metrics_df = pd.DataFrame({
    'Model': ['Flood', 'Storm', 'Drought','fire Miscellaneous','earthquake','extreme_temperature','water','volcanic activity','wildfire'],
    'Loss': [metrics_1[0], metrics_2[0], metrics_3[0],metrics_4[0], metrics_5[0], metrics_6[0],metrics_7[0], metrics_8[0], metrics_9[0]],
    'Accuracy': [accuracy_1, accuracy_2, accuracy_3,accuracy_4, accuracy_5, accuracy_6,accuracy_7, accuracy_8, accuracy_9],
    'Recall': [recall_1, recall_2, recall_3,recall_4, recall_5, recall_6,recall_7, recall_8, recall_9],
    'Precision': [precision_1, precision_2, precision_3,precision_4, precision_5, precision_6,precision_7, precision_8, precision_9],
    'F1-Score': [f1_1, f1_2, f1_3,f1_4, f1_5, f1_6,f1_7, f1_8, f1_9]
})

metrics_df.to_csv('FCNN_model_metrics.csv', index=False)

print("预测完成并已保存模型和评估指标到 'FCNN_model_metrics.csv'")