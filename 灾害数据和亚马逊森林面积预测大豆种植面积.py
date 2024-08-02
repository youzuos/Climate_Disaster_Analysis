import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('predictions_2001_2020.csv')

# 检查缺失值
print(data.isna().sum())

# 处理缺失值 - 用数值列的均值填充NaN
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 再次检查缺失值
print(data.isna().sum())

# 提取特征和目标变量
features = ['亚马逊森林面积', 'flood_Count', 'massmovement_Count', 'storm_Count']
target = '大豆面积'

X = data[features]
y = data[target]

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'XGBoost': XGBRegressor(random_state=42)
}

# 训练模型并评估性能
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}

# 打印评估结果
for name, metrics in results.items():
    print(f'{name}:\n  MSE: {metrics["MSE"]:.4f}\n  R²: {metrics["R²"]:.4f}\n')

# 可视化预测结果
plt.figure(figsize=(15, 10))
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    plt.subplot(3, 2, i+1)
    plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name}')
plt.tight_layout()
plt.show()