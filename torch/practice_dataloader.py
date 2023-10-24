from torch.utils.data import random_split, DataLoader
from sklearn.datasets import fetch_california_housing
from nn.dataset import CaliforniaDataset

california_housing_data = fetch_california_housing()
dataset = CaliforniaDataset(california_housing_data)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

# dataloaderを作成
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=100, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=100, shuffle=False)

print(f"type(train_loader) = {type(train_loader)}")                 # 型
first_batch = next(iter(train_loader))                              # イテレータにしてnextで最初のミニバッチを取り出す

# print(f"first_batch[0] = {first_batch[0]}")                         # 各ミニバッチは2つの要素を持つリストで、0番目がデータ
# print(f"first_batch[1] = {first_batch[1]}")                         # 1番目が教師ラベルのリスト
print(f"type(first_batch[0]) = {type(first_batch[0])}")             # 自動でtensorになる
print(f"type(first_batch[1]) = {type(first_batch[1])}")             
print(f"first_batch[0].shape = {first_batch[0].shape}")             # (100, 8) <-(batchsize, 説明変数数)
print(f"first_batch[1].shape = {first_batch[1].shape}")             # (100) <-(batchsize)
