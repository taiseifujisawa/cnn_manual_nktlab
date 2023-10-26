import torch
from torch.utils.data import random_split, DataLoader
import sys
from nn.set_seed import set_seed
from nn.logging_controller import Logger

from sklearn.datasets import fetch_california_housing
from nn.dataset import CaliforniaDataset

from nn.model import MyModel

import torch.nn.functional as F

from torch.optim import Adam

# 乱数シードを1で固定
set_seed(seed=1)
# ./log/log.txtをlog_fileに設定
sys.stdout = Logger("./log/log_nn.txt")

### GPUが使える環境なら"cuda" ###
device = "cuda" if torch.cuda.is_available() else "cpu"

# データ読み込み
california_housing_data = fetch_california_housing()
# データセット作成
dataset = CaliforniaDataset(california_housing_data)

# データセットをランダムに分割
train_set, val_set, test_set = random_split(dataset, [0.998, 0.001, 0.001])

# モデルのインスタンス化
model = MyModel()
### GPUへ ###
model.to(device)

# データローダーの作成
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=100, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=100, shuffle=False)

# 最適化アルゴリズム(Adam)のインスタンス化
optimizer = Adam(model.parameters(), lr=1e-3)

# エポック数
N_EPOCH = 50

# printの頻度
VERBOSE = False


# 訓練フェーズ開始
print("### Train ###")

# エポックのループ
for ep in range(N_EPOCH):
    # 訓練モードに切り替え(元から)
    model.train()

    ep_running_loss = 0.0   # エポック毎のlossの合計
    n_total = 0             # 全データ数をカウント

    print("[epoch, batch]")
    # ミニバッチのループ
    for i, data in enumerate(train_loader):

        # 勾配を全てリセット
        optimizer.zero_grad()

        inputs, labels = data
        ### GPUへ ###
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)                 ### このあたりの変数は全部GPU上にある ###
        loss = F.mse_loss(outputs, labels)
        loss.backward()

        # 現時点でのgradを使って全パラメータを一度に更新
        optimizer.step()

        ### GPU上でやらなくていい処理はCPUに戻してからやる ###
        loss = loss.cpu()
        labels = labels.cpu()
        ep_running_loss += loss.item()
        n_total += labels.shape[0]

        if VERBOSE:
            # ミニバッチ毎にlossの平均値をprint
            print('[{:5d}, {:5d}] train loss: {:.3f}'
                    .format(ep+1, i+1, loss / labels.shape[0]))

    # エポック毎にlossの平均値をprint
    print('[{:5d},   all] train loss: {:.3f}'
            .format(ep+1, ep_running_loss / n_total), end=", ")

    n_total = 0                     # 訓練はエポックのループの最後なのでリセット
    ep_running_loss = 0.0


    running_loss = 0.0              # エポック毎のlossの合計(val.用)

    # 推論モードに切り替え
    model.eval()

    # with torch.no_grad(): コンテキスト内は一切計算グラフが構築されない
    with torch.no_grad():
        # ミニバッチのループ
        for data in val_loader:
            inputs, labels = data
            ### GPUへ ###
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            outputs = outputs.cpu()
            labels = labels.cpu()

            loss = F.mse_loss(outputs, labels)
            running_loss += loss.item()
            n_total += labels.shape[0]

    # エポック毎にlossの平均値をprint
    print('val. loss: {:.3f}'
            .format(running_loss / n_total))

    n_total = 0                     # エポックのループの最後なのでリセット
    running_loss = 0.0

### torch.save, torch.loadはCPUでやる ###
model_save_path = "./log/model.pth"
torch.save(model.state_dict(), model_save_path)


# 試しに初期状態のモデルを使って回帰してみる(的外れな結果になる)
model = MyModel()       # 初期化
### GPUへ ###
model.to(device)

model.eval()
predictions_before = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        ### GPUへ ###
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        ### CPUへ ###
        outputs = outputs.cpu()
        labels = labels.cpu()

        for o, l in zip(outputs.numpy(), labels.numpy()):
            predictions_before.append(o.item())

# 保存したモデルをロードするときはまずはインスタンス化
model_test = MyModel()
model_test.load_state_dict(torch.load(model_save_path))

### GPUへ ###
model_test.to(device)

# テストフェーズ開始
print("### Test ###")
model_test.eval()
predictions = []
answers = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        ### GPUへ ###
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_test(inputs)

        ### CPUへ ###
        outputs = outputs.cpu()
        labels = labels.cpu()
        for o, l in zip(outputs.numpy(), labels.numpy()):
            predictions.append(o.item())
            answers.append(l.item())
        loss = F.mse_loss(outputs, labels)
        running_loss += loss.item()
        n_total += labels.shape[0]
print('test loss: {:.3f}'
        .format(running_loss / n_total))

# 推論結果と教師ラベル、ついでに訓練前のモデルの推論結果をcsv形式で出力
output_filepath = "./log/test_result.csv"
with open(output_filepath, "w") as f:
    f.write("index,prediction_before,prediction_after,answer\n")
    for i, (pb, p, a) in enumerate(zip(predictions_before, predictions, answers)):
        f.write(f"{i + 1},{pb},{p},{a}\n")
