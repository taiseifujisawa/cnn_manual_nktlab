import pandas as pd
from sklearn.datasets import fetch_california_housing

# データのダウンロード
california_housing_data = fetch_california_housing()

# データの確認
print(f"california_housing_data.data = {california_housing_data.data}")                 # 説明変数の中身
print(f"california_housing_data.target = {california_housing_data.target}")             # 目的変数の中身
print()
print(f"type(california_housing_data.data) = {type(california_housing_data.data)}")     # 型
print(f"california_housing_data.target = {type(california_housing_data.target)}")
print()
print(f"california_housing_data.data.shape = {california_housing_data.data.shape}")     # サイズ
print(f"california_housing_data.target.shape = {california_housing_data.target.shape}")
print()

# 目的変数
exp_data = pd.DataFrame(california_housing_data.data, columns=california_housing_data.feature_names)
# 説明変数
tar_data = pd.DataFrame(california_housing_data.target, columns=['HousingPrices'])
# データを結合
data = pd.concat([exp_data, tar_data], axis=1)

print(data.head())                                      # 最初の5件を閲覧
print()
print(f"data.shape = {data.shape}")                     # データのサイズ (20640, 9)
print()
print(f"data.dtypes:\n{data.dtypes}")                   # データの型 全てfloat64
print()
print(f"data.isnull().sum():\n{data.isnull().sum()}")   # 欠損値数 全て0
