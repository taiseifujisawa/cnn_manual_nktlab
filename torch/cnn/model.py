import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

# PyTorchのモデルはtorch.nn.Module(=nn.Module)を継承するのがルール
class MyCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2dは、引数in_channelsに入力チャネル数(RGB画像を入力する場合最初のconv.はin_channels=3)、out_channelsに出力チャネル数
        # kernel_sizeにカーネルサイズを指定する paddingは画像の外側を0で埋めるピクセル数を指定(なくてもいい)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=32*4*4, out_features=100)      # 計算しておかなきゃいけない
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=10)          # 分類問題なのでout_features=クラス数にする
        self.flatten = nn.Flatten()
        self.pool = F.max_pool2d
        self.relu = F.relu      # 訓練するパラメータがないものは使い回して大丈夫
    
    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x), 2))      # F.max_pool2dは関数を呼ぶときにカーネルサイズを指定する
        x = self.relu(self.pool(self.conv2(x), 2))
        x = self.relu(self.pool(self.conv3(x), 2))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x        # Loss関数の直前の計算結果をreturnする 最終層活性化関数は書かない

if __name__ == "__main__":
    import torch
    import sys
    from set_seed import set_seed
    from logging_controller import Logger
    
    set_seed(seed=1)
    sys.stdout = Logger("./log/log.txt")
    
    transforms = Compose([
        ToTensor(),                                         # torch.Tensorへ変換
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),        # RGB全て、平均0.5、標準偏差0.5に標準化 グレースケールなら、Normalize([0.5], [0.5])とかく
    ])

    # transformを引数にとってデータセット作成
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
    
    model = MyCnnModel()   # モデルのインスタンス化
    
    print()
    
    # print(f"model.conv1.weight.data = {model.conv1.weight.data}")                       # 1層目conv.層のウェイト
    print(f"model.conv1.weight.data.shape = {model.conv1.weight.data.shape}")           # 1層目conv.層のウェイトのサイズ
    # print(f"model.conv2.bias.data = {model.conv2.bias.data}")                           # 2層目conv.層のバイアス
    print(f"model.conv2.bias.data.shape = {model.conv2.bias.data.shape}")               # 2層目conv.層のバイアスのサイズ
    print(f"model.conv3.weight.requires_grad = {model.conv3.weight.requires_grad}")     # 3層目conv.層のウェイトのrequires_grad(自動でTrueになっている)
    # print(f"model.conv3.weight.grad = {model.conv3.weight.grad}")                       # 3層目conv.層のウェイトのgrad(backwardがまだなのでNone)
    print()
    
    x, t = train_set.__getitem__(123)     # 例として123番目のデータを取得
    # 入力tensorは(batchsize, channel, height, width)のサイズでなければならない batchsize=1でも4次元にする DataLoaderを使えばこの問題は解消する
    x = x.expand([1, 3, 32, 32])
    
    # 入力サイズは(1, in_channels, height, width) 出力サイズは(1, out_channels, height, width)  (K=3, S=1, P=1なのでheight, widthは変わらない)
    conv_x = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1)(x)
    # 入力サイズは(1, channels, height, width) 出力サイズは(1, channels, height/2, width/2)  (K=S=2, P=0なのでheight, widthは半分 poolなのでchannelsも変わらない)
    pool_x = F.max_pool2d(x, kernel_size=2)
    print(f"in_size = {x.shape}, conv_out_size = {conv_x.shape}, pool_out_size = {pool_x.shape}")     
    print()
    
    
    x, t = train_set.__getitem__(123)     # 例として123番目のデータを取得
    
    # 入力tensorは(batchsize, channel, height, width)のサイズでなければならない batchsize=1でも4次元にする DataLoaderを使えばこの問題は解消する
    x = x.expand([1, 3, 32, 32])
    x.requires_grad = True
    # サイズを(batchsize)にする(F.cross_entropyの警告回避) batchsize=1でも1次元にする DataLoaderを使えばこの問題は解消する
    t = torch.tensor(t).expand([1])
    
    y = model(x)    
    
    # 今回は試しにcross-entropy Lossを計算 tはone-hotでなくてよい
    # この中でsoftmaxが勝手に計算されるのでモデルにsoftmaxを書かなくてよい
    loss = F.cross_entropy(y, t)
    # print(f"x = {x}\ny = {y}\nt = {t}\nloss = {loss}")      # 入力x、モデルの出力y、Cross-entropy Lossの値loss
    print(f"x_size = {x.shape}, y_size = {y.shape}, ploss_size = {loss.shape}")     
    print()
    
    y.retain_grad()
    loss.retain_grad()

    loss.backward()
    
    # 各tensorのgrad属性にlossに対する勾配が記録される
    # print(f"model.conv1.weight.grad = {model.conv1.weight.grad}")
    print(f"model.conv1.weight.grad.shape = {model.conv1.weight.grad.shape}")       # サイズ(out_channels, in_channels, kernel_height, kernel_width)
    # print(f"model.conv1.bias.grad = {model.conv1.bias.grad}")
    print(f"model.conv1.bias.grad.shape = {model.conv1.bias.grad.shape}")           # サイズ(out_channels)
    # print(f"x.grad = {x.grad}")
    print(f"x.grad.shape = {x.grad.shape}")
    # print(f"y.grad = {y.grad}")
    print(f"y.grad.shape = {y.grad.shape}")
    # print(f"loss.grad = {loss.grad}")
    print(f"loss.grad.shape = {loss.grad.shape}")
    print()
    
    # model.parameters()はモデルの全パラメータを生成するイテレータ 後でoptimizerに渡す
    print(f"model.parameters() = {model.parameters()}")
