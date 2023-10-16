import torch.nn as nn
import torch.nn.functional as F

# PyTorchのモデルはtorch.nn.Module(=nn.Module)を継承するのがルール
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # super().__init__()の後に、メンバ変数としてネットワークに用いる層のクラス(or関数)をインスタンス化して持っておく
        # nn.Linearは、引数in_featuresに入力ノード数、out_featuresに出力ノード数を指定する
        # forwardメソッドが呼ばれるまで計算グラフは構築されないが、計算グラフ構築時に、
        # 前の層のout_featuresと次の層のin_featuresの値が異なるとエラーとなるので、この時点で考えておかないといけない
        # また、同じin_features, out_featuresの層を使うからといって1つのインスタンスを使い回すと計算が狂うので、
        # 面倒くさいけど使う分だけインスタンス化しなければいけない
        self.fc1 = nn.Linear(in_features=8, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)
        self.relu = F.relu      # reluは訓練するパラメータがないので、使い回して大丈夫
    
    # forward関数で計算グラフが構築される
    # 呼び出しはMyModelクラスのインスタンスに対して、直接引数を与えればよい(__call__メソッドと同じ感じ)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x        # Loss関数の直前の計算結果をreturnする 最終層活性化関数は書かない

if __name__ == "__main__":
    import torch
    from dataset import CaliforniaDataset
    from sklearn.datasets import fetch_california_housing
    import sys
    from set_seed import set_seed
    from logging_controller import Logger
    
    set_seed(seed=1)
    sys.stdout = Logger("./log/log.txt")
    
    california_housing_data = fetch_california_housing()
    dataset = CaliforniaDataset(california_housing_data)
    
    model = MyModel()   # モデルのインスタンス化
    
    print(f"model.fc1.weight.data = {model.fc1.weight.data}")                       # 1層目全結合層のウェイト
    print(f"model.fc1.weight.data.shape = {model.fc1.weight.data.shape}")           # 1層目全結合層のウェイトのサイズ
    print(f"model.fc2.bias.data = {model.fc2.bias.data}")                           # 2層目全結合層のバイアス
    print(f"model.fc2.bias.data.shape = {model.fc2.bias.data.shape}")               # 2層目全結合層のバイアスのサイズ
    print(f"model.fc3.weight.requires_grad = {model.fc3.weight.requires_grad}")     # 3層目全結合層のウェイトのrequires_grad(自動でTrueになっている)
    print(f"model.fc3.weight.grad = {model.fc3.weight.grad}")                       # 3層目全結合層のウェイトのgrad(backwardがまだなのでNone)
    print()
    
    x, t = dataset.__getitem__(123)     # 例として123番目のデータを取得
    x = torch.tensor(x, requires_grad=True)
    t = torch.tensor([t])  # tはxと次元数が違うので[t]と書いて合わせる(F.mse_lossの警告回避) 後でDataLoaderを勉強すればこの問題は解消する
    
    # modelにtensorを入力(__call__メソッドのように、インスタンスに対して直接引数を入れればforwardメソッドが呼び出せる もちろん y = model.forward(x) でもOK)
    y = model(x)    
    loss = F.mse_loss(y, t)     # 今回は試しにMSE Lossを計算
    print(f"x = {x}\ny = {y}\nt = {t}\nloss = {loss}")      # 入力x、モデルの出力y、MSE Lossの値loss
    print()
    
    y.retain_grad()
    loss.retain_grad()

    loss.backward()
    
    # 各tensorのgrad属性にlossに対する勾配が記録される
    print(f"model.fc1.weight.grad = {model.fc1.weight.grad}")
    print(f"model.fc1.weight.grad.shape = {model.fc1.weight.grad.shape}")       # サイズ(out_features, in_features)
    print(f"model.fc1.bias.grad = {model.fc1.bias.grad}")
    print(f"model.fc1.bias.grad.shape = {model.fc1.bias.grad.shape}")           # サイズ(out_features)
    print(f"x.grad = {x.grad}")
    print(f"x.grad.shape = {x.grad.shape}")
    print(f"y.grad = {y.grad}")
    print(f"y.grad.shape = {y.grad.shape}")
    print(f"loss.grad = {loss.grad}")
    print(f"loss.grad.shape = {loss.grad.shape}")
    print()
    
    # model.parameters()はモデルの全パラメータを生成するイテレータ 後でoptimizerに渡す
    print(f"model.parameters() = {model.parameters()}")
