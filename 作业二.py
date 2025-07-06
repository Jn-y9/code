import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
df = pd.read_csv('US-pumpkins.csv', encoding='utf-8')

# 初步查看数据
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据列名:", df.columns.tolist())
print("\n缺失值统计:")
print(df.isnull().sum())

# 删除全为空的列
df = df.dropna(axis=1, how='all')

# 删除重复行
df = df.drop_duplicates()

# 处理缺失值 - 对于数值列用中位数填充，分类列用众数填充
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 转换日期列为datetime类型
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 提取年份和月份
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 查看处理后的数据信息
print("\n处理后的数据信息:")
print(df.info())

# 选择数值列计算相关性
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("\n数值列:", numeric_cols)

# 1. 价格分布分析
plt.figure(figsize=(12, 6))
sns.histplot(df['Low Price'], kde=True, color='blue', label='最低价')
sns.histplot(df['High Price'], kde=True, color='red', label='最高价')
plt.title('南瓜价格分布')
plt.xlabel('价格')
plt.ylabel('频率')
plt.legend()
plt.show()

# 2. 品种分布
plt.figure(figsize=(12, 6))
df['Variety'].value_counts().plot(kind='bar')
plt.title('南瓜品种分布')
plt.xlabel('品种')
plt.ylabel('数量')
plt.xticks(rotation=45)
plt.show()

# 3. 价格随时间变化
plt.figure(figsize=(15, 6))
monthly_avg = df.groupby(['Year', 'Month'])['Low Price'].mean().reset_index()
monthly_avg['Year-Month'] = monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month'].astype(str)
sns.lineplot(data=monthly_avg, x='Year-Month', y='Low Price', marker='o')
plt.title('南瓜最低价随时间变化')
plt.xlabel('年月')
plt.ylabel('平均最低价')
plt.xticks(rotation=45)
plt.show()

# 4. 不同品种的价格比较
plt.figure(figsize=(15, 6))
sns.boxplot(data=df, x='Variety', y='Low Price')
plt.title('不同品种南瓜的价格比较')
plt.xlabel('品种')
plt.ylabel('最低价')
plt.xticks(rotation=45)
plt.show()

# 5. 产地分布
plt.figure(figsize=(12, 6))
df['Origin'].value_counts().head(10).plot(kind='bar')
plt.title('南瓜产地分布(前10)')
plt.xlabel('产地')
plt.ylabel('数量')
plt.xticks(rotation=45)
plt.show()

# 选择相关特征
features = ['Type', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Month']
target = 'Low Price'

# 创建特征和标签
X = df[features]
y = df[target]

# 分类变量编码
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col].astype(str))  # 使用.loc进行索引
    label_encoders[col] = le

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# 线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型评估:")
print("均方误差(MSE):", mse)
print("R平方值:", r2)

# 特征重要性
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_
}).sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(importance)

# 1. 实际价格 vs 预测价格
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('实际价格 vs 预测价格')
plt.show()

# 2. 残差图
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.title('残差分析图')
plt.show()

# 3. 特征重要性可视化
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('特征重要性')
plt.xlabel('重要性系数')
plt.ylabel('特征')
plt.show()

# 4. 品种与价格的热力图
pivot_table = df.pivot_table(values='Low Price', index='Variety', columns='Month', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".1f")
plt.title('不同品种在不同月份的平均价格热力图')
plt.xlabel('月份')
plt.ylabel('品种')
plt.show()

# 5. 产地与价格的关系
plt.figure(figsize=(12, 6))
top_origins = df['Origin'].value_counts().head(10).index
sns.boxplot(data=df[df['Origin'].isin(top_origins)], x='Origin', y='Low Price')
plt.title('不同产地南瓜的价格分布(前10产地)')
plt.xlabel('产地')
plt.ylabel('最低价')
plt.xticks(rotation=45)
plt.show()
