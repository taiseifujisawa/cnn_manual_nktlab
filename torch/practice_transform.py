import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)  # train=Trueで訓練データ
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)  # train=Falseでテストデータ

print()

print(f"type(train_set) = {type(train_set)}")                       # np.ndarray
print(f"type(test_set) = {type(test_set)}")                         # list

print(f"train_set.classes = {train_set.classes}")

print(f"type(train_set.data) = {type(train_set.data)}")             # data属性でデータ
print(f"type(test_set.data) = {type(test_set.data)}")
print(f"type(train_set.targets) = {type(train_set.targets)}")       # targets属性でラベル
print(f"type(test_set.targets) = {type(test_set.targets)}")

print(f"train_set.data.shape = {train_set.data.shape}")             # trainデータは50000枚 (50000, height, width, rgb_channel)の順
print(f"len(train_set.targets) = {len(train_set.targets)}")
print(f"test_set.data.shape = {test_set.data.shape}")               # testデータは10000枚 (10000, height, width, rgb_channel)の順
print(f"len(test_set.targets) = {len(test_set.targets)}")

print(f"train_set.data.max() = {train_set.data.max()}")             # 0~255の整数値になっている
print(f"train_set.data.min() = {train_set.data.min()}")    
print(f"train_set.data.dtype = {train_set.data.dtype}")         

print()

data_123 = train_set.__getitem__(123)       # 123番目のデータを取り出し

print(f"data_123 = {data_123}")                     # 中身を見てみる dataはPIL型になっている
print(f"data_123[0].size = {data_123[0].size}")     # PIL.sizeでサイズを取得 RGBでも2次元で表示される
print(f"data_123[0].mode = {data_123[0].mode}")     # PIL.modeはちゃんとRGBになる

print()

transforms = Compose([
    ToTensor(),                                         # torch.Tensorへ変換
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),        # RGB全て、平均0.5、標準偏差0.5に標準化 グレースケールなら、Normalize([0.5], [0.5])とかく
])

# transformを引数にとってデータセット作成
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms)

print()

data_123 = train_set.__getitem__(123)       # 123番目のデータを取り出し

print(f"data_123 = {data_123}")                     # 中身を見てみる dataはtorch.Tensor型に変わっている
print(f"data_123[0].size = {data_123[0].shape}")    # (channel, height, width)の順になっている
print(f"data_123[0].max() = {data_123[0].max()}")   # 標準化されていて型もtorch.float32になる
print(f"data_123[0].min() = {data_123[0].min()}")
print(f"data_123[0].dtype = {data_123[0].dtype}")

