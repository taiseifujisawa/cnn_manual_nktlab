import torch
from nn.model import MyModel
from torch.utils.data import random_split, DataLoader
from sklearn.datasets import fetch_california_housing
from nn.dataset import CaliforniaDataset

california_housing_data = fetch_california_housing()
dataset = CaliforniaDataset(california_housing_data)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
first_batch = next(iter(train_loader))                              # イテレータにしてnextで最初のミニバッチを取り出す


print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")   # GPUが使える環境ならTrue

device = "cuda" if torch.cuda.is_available() else "cpu"             # GPUが使える環境なら"cuda"

x, t = first_batch
print(f"x.device = {x.device}")                                     # まだCPU

x = x.to(device)                                                    # GPUへ 非破壊処理なので変数xに再代入
print(f"x.device = {x.device}")

x_np = x.cpu().numpy()                                              # numpyメソッドはCPUのtensorにしか使えない
print(f"type(x_np) = {type(x_np)}")

model = MyModel()
print(f"model.fc1.weight.device = {model.fc1.weight.device}")       # まだCPU

model.to(device)                                                    # GPUへ 破壊処理なので再代入はいらない
print(f"model.fc1.weight.device = {model.fc1.weight.device}")

y = model(x)                                                        # input tensor x, modelが両方GPUまたはCPUにいないとエラー
print(f"y.device = {y.device}")                                     # GPU上のtensorで計算されて生成されたtensorはGPU上に配置される
