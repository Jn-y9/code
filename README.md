# 项目概述
本项目使用美国南瓜市场数据(US-pumpkins.csv)，通过数据处理、可视化分析和机器学习建模，探究影响南瓜价格的关键因素，并构建价格预测模型。

# 数据准备
- 原始数据包含多列特征，包括价格、品种、产地、日期等
- 数据清洗步骤:
  - 删除全为空值的列
  - 删除重复行
  - 数值列用中位数填充缺失值，分类列用众数填充
  - 日期列转换为datetime类型并提取年份和月份

# 数据分析

## 1. 南瓜价格分布
![Figure_1](https://github.com/user-attachments/assets/b398c77c-aea5-4296-be43-cda46de522ce)
展示了南瓜最低价和最高价的分布情况。价格主要集中在0到300之间，且最低价的分布较为集中，而最高价的分布较为分散。

## 2. 南瓜品种分布
![Figure_2](https://github.com/user-attachments/assets/284d20d3-38e9-4af2-8e55-15ff28ead342)
展示了不同品种南瓜的数量分布。'HOODEN TYPE'和'PIE TYPE'的数量最多，而'BLUE TYPE'和'MUSCLE HEAD'的数量最少。

## 3. 南瓜最低价价格随时间变化
![Figure_3](https://github.com/user-attachments/assets/4dbeec47-9a91-43bc-aad3-5f826e33b457)
展示了南瓜最低价随时间的变化趋势。价格在2016年12月达到最低点，然后在2017年6月达到最高点，之后有所回落。

## 4. 不同品种南瓜的价格比较
![Figure_4](https://github.com/user-attachments/assets/e12643c4-d0b0-46e2-ab37-ab7be1946ae2)
展示了不同品种南瓜的价格比较。'CINDERELLA'和'FAIRYTALE'的价格较高，而'MINIATURE'的价格最低。

## 5. 南瓜产地分布
![Figure_5](https://github.com/user-attachments/assets/4236e008-2a10-44c8-ab76-53a21ac45284)
展示了南瓜产地的分布情况。'PENNSYLVANIA'和'MICHIGAN'的南瓜产量最高。

# 建模与预测

## 特征选择
使用以下特征预测最低价(Low Price):
- Type(类型)
- Package(包装)
- Variety(品种)
- Origin(产地)
- Item Size(尺寸)
- Color(颜色)
- Month(月份)

## 数据预处理
- 对分类变量进行标签编码(Label Encoding)
- 将数据划分为训练集(80%)和测试集(20%)

## 线性回归模型
- 使用scikit-learn的LinearRegression
- 评估指标:
  - 均方误差（MSE）：5363.16
  - R平方值：0.2587

## 特征重要性
![Figure_8](https://github.com/user-attachments/assets/3f562fcf-f904-446a-9998-acb55c107df4)
展示了特征重要性。'Package'和'Origin'对价格预测的影响最大，而'Color'的影响最小。

# 结果可视化

## 1. 实际价格 vs 预测价格
![Figure_6](https://github.com/user-attachments/assets/0e3a7950-36b3-4666-858f-728dac824b7b)
展示了实际价格与预测价格的对比。预测价格与实际价格大致呈线性关系，但存在一定的偏差。

## 2. 残差分析图
![Figure_7](https://github.com/user-attachments/assets/c665dfc8-9dc3-416d-b5bf-761b56122b02)
展示了残差分析图。残差分布较为均匀，但存在一些较大的偏差点。

## 3. 不同品种在不同月份的平均价格热力图
![Figure_9](https://github.com/user-attachments/assets/e0cd3e1d-7d47-4f45-a108-27a2061cc3bd)
展示了不同品种在不同月份的平均价格热力图。不同品种的价格在不同月份存在显著差异。

## 4. 不同产地南瓜的价格分布
![Figure_10](https://github.com/user-attachments/assets/9f185646-0a98-4f92-baed-6ab57a72ae5f)
展示了不同产地南瓜的价格分布情况。不同产地的价格分布存在较大差异。
# 运行环境
- Python 3.10
- 主要库:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
    
# 结论
南瓜价格的分布主要集中在0到300之间，不同品种和产地的南瓜价格存在显著差异。时间因素对南瓜价格有较大影响，2017年6月价格达到最高点。特征重要性分析表明，Color、Package和Origin是影响南瓜价格的主要因素。
