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
sys.stdout = Logger("./log/log.txt")

# データ読み込み
california_housing_data = fetch_california_housing()
# データセット作成
dataset = CaliforniaDataset(california_housing_data)

# データセットをランダムに分割
train_set, val_set, test_set = random_split(dataset, [0.998, 0.001, 0.001])

# モデルのインスタンス化
model = MyModel()

# データローダーの作成
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=100, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=100, shuffle=False)

### 最適化アルゴリズム(Adam)のインスタンス化 ###
optimizer = Adam(model.parameters(), lr=1e-3)

# エポック数
N_EPOCH = 50

# printの頻度
VERBOSE = False

# 訓練フェーズ開始
print("### Train ###")
### エポックのループ ###
for ep in range(N_EPOCH):
    bat_running_loss = 0.0  # バッチ毎のlossの合計
    ep_running_loss = 0.0   # エポック毎のlossの合計
    n_total = 0             # 全データ数をカウント
    
    print("[epoch, batch]")
    #### ミニバッチのループ ###
    for i, data in enumerate(train_loader):
        
        #### 勾配を全てリセット ###
        optimizer.zero_grad()
        
        inputs, labels = data
        outputs = model(inputs)
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        
        #### 現時点でのgradを使って全パラメータを一度に更新 ###
        optimizer.step()
        
        bat_running_loss += loss.item()     # スカラーtensorはitemメソッドでpython標準の型にキャストできる
        ep_running_loss += loss.item()
        n_total += labels.shape[0]
        
        if VERBOSE:
            # ミニバッチ毎にlossの平均値をprint
            print('[{:5d}, {:5d}] train loss: {:.3f}'
                    .format(ep+1, i+1, bat_running_loss / labels.shape[0]))
            
        bat_running_loss = 0.0      # ミニバッチのループの最後なのでリセット
    
    # エポック毎にlossの平均値をprint
    print('[{:5d},   all] train loss: {:.3f}'
            .format(ep+1, ep_running_loss / n_total))
    
    n_total = 0                     # エポックのループの最後なのでリセット
    ep_running_loss = 0.0
    
