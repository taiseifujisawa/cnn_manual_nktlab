import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
import sys
from nn.set_seed import set_seed
from nn.logging_controller import Logger

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import CIFAR10

from cnn.model import MyCnnModel

import torch.nn.functional as F

from torch.optim import Adam

# 乱数シードを1で固定
set_seed(seed=1)
# ./log/log.txtをlog_fileに設定
sys.stdout = Logger("./log/log_cnn.txt")

### GPUが使える環境なら"cuda" ###
device = "cuda" if torch.cuda.is_available() else "cpu"

# transforms
transforms = Compose([
    ToTensor(),                                         # torch.Tensorへ変換
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),        # RGB全て、平均0.5、標準偏差0.5に標準化 グレースケールなら、Normalize([0.5], [0.5])とかく
])

# データセット作成
train_set = CIFAR10(root="./data", train=True, download=True, transform=transforms)  # train=Trueで訓練データ
test_set = CIFAR10(root="./data", train=False, download=True, transform=transforms)  # train=Falseでテストデータ

# データセットをランダムに分割
_train_set, val_set = random_split(train_set, [0.8, 0.2])

# モデルのインスタンス化
model = MyCnnModel()
### GPUへ ###
model.to(device)

# データローダーの作成
train_loader = DataLoader(dataset=_train_set, batch_size=100, shuffle=True)
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
    n_correct = 0           # 正解数をカウント

    print("[epoch, batch]")
    # ミニバッチのループ
    for i, data in enumerate(train_loader):

        # 勾配を全てリセット
        optimizer.zero_grad()

        inputs, labels = data
        ### GPUへ ###
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)                     ### このあたりの変数は全部GPU上にある ###
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # 現時点でのgradを使って全パラメータを一度に更新
        optimizer.step()

        ### GPU上でやらなくていい処理はCPUに戻してからやる ###
        loss = loss.cpu()
        labels = labels.cpu()
        ep_running_loss += loss.item()
        n_total += labels.shape[0]
        predictions = torch.argmax(outputs.cpu(), dim=1)          # 値が最も大きいindexがpredictionになる
        n_correct += (predictions == labels).sum().item()   # 正解数を数える

        if VERBOSE:
            # ミニバッチ毎にlossの平均値をprint
            print('[{:5d}, {:5d}] train loss: {:.3f}'
                    .format(ep+1, i+1, loss / labels.cpu().shape[0]))

    # エポック毎にlossの平均値をprint
    print('[{:5d},   all] train loss: {:.3f}, train accuracy: {:.4f}'
            .format(ep+1, ep_running_loss / n_total, n_correct / n_total))

    n_total = 0                     # 訓練はエポックのループの最後なのでリセット
    n_correct = 0
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

            ### CPUへ ###
            outputs = outputs.cpu()
            labels = labels.cpu()

            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item()
            n_total += labels.shape[0]
            predictions = torch.argmax(outputs, dim=1)
            n_correct += (predictions == labels).sum().item()
    # エポック毎にlossの平均値をprint
    print('val. loss: {:.3f}, val. accuracy: {:.4f}'
            .format(running_loss / n_total, n_correct / n_total))

    n_total = 0                     # エポックのループの最後なのでリセット
    n_correct = 0
    running_loss = 0.0

### torch.save, torch.loadはCPUでやる ###
model_save_path = "./log/cnnmodel.pth"
torch.save(model.cpu().state_dict(), model_save_path)

# 保存したモデルをロードするときはまずはインスタンス化
model_test = MyCnnModel()
model_test.load_state_dict(torch.load(model_save_path))

### GPUへ ###
model_test.to(device)

# テストフェーズ開始
print("### Test ###")
model_test.eval()
predictions_list = []
answers_list = []
confidence_list = []
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

        softmax_out = F.softmax(outputs, dim=1)
        # 冗長だけど許してちょ
        for o, l, s in zip(outputs.numpy(), labels.numpy(), softmax_out.numpy()):
            predictions_list.append(np.argmax(o).item())
            answers_list.append(l.item())
            confidence_list.append(s.max().item())
        loss = F.cross_entropy(outputs, labels)
        running_loss += loss.item()
        n_total += labels.shape[0]
        predictions = torch.argmax(outputs, dim=1)
        n_correct += (predictions == labels).sum().item()
    print('test loss: {:.3f}, test accuracy: {:.4f}'
            .format(running_loss / n_total, n_correct / n_total))

# 推論結果と教師ラベルをcsv形式で出力
output_filepath = "./log/test_result_cnn.csv"
with open(output_filepath, "w") as f:
    f.write("index,confidence,prediction,answer\n")
    for i, (p, a, c) in enumerate(zip(predictions_list, answers_list, confidence_list)):
        f.write("{},{:.4f},{},{}\n".format(i + 1, c, p, a))
