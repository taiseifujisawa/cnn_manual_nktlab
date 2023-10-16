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


print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

device = "cuda" if torch.cuda.is_available() else "cpu"

x, t = first_batch
print(f"x.device = {x.device}")

x.to(device)
print(f"x.device = {x.device}")

x_np = x.cpu().numpy()
print(f"type(x_np) = {type(x_np)}")

model = MyModel()
model.to(device)
print(f"model.fc1.weight.device = {model.device}")

y = model(x)
print(f"y.device = {y.device}")
