import pandas as pd
import os

# 路径
CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/combined_data.csv"

# 尝试用更鲁棒的方式读取
df = pd.read_csv(CSV_PATH)

# 查看前几行 ID 和 状态，确认列名是否正确
print("列名列表:", df.columns.tolist())
print("\n前 5 行数据预览:")
print(df[['id', 'covid_status']].head())

# 查看所有的标签类型（这很重要，因为 Coswara 有多种状态）
print("\n所有标签分布:")
print(df['covid_status'].value_counts())