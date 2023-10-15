import numpy as np
from torch.utils.data import random_split, Dataset

# PyTorchのデータセットはtorch.utils.data.Datasetを継承するのがルール
class CaliforniaDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        
        # modelのパラメータがfloat32なので、dataがfloat64だったりすると後で怒られる。回避するために明示する。
        # modelの出力層のサイズが2次元なので、targetが1次元だったりすると後で怒られる。回避するために明示する。
        self.data = data.data.astype(np.float32)
        if data.target.ndim == 1:
            self.target = np.expand_dims(data.target.astype(np.float32), axis=1)
        elif data.target.ndim == 2:
            self.target = data.target.astype(np.float32)
        self.instances = [(d, t) for d, t in zip(self.data, self.target)]
        
    # torch.utils.data.Datasetの__getitem__メソッドはオーバーライド必須
    # id番目のデータとターゲットを返すように書く
    def __getitem__(self, id):
        data, target = self.instances[id]
        return data, target
    
    # torch.utils.data.Datasetの__len__メソッドはオーバーライド必須
    # データセットのデータ数を返すように書く
    def __len__(self):
        return len(self.instances)

# このファイルから実行した場合はif文が実行されるが
# 他のファイルからこのファイルをimportした場合は実行されない
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    import sys
    from set_seed import set_seed
    from logging_controller import Logger
    
    set_seed(seed=1)                            # 乱数シードを1で固定
    sys.stdout = Logger("./log/log.txt")        # ./log/log.txtをlog_fileに設定
    
    california_housing_data = fetch_california_housing()
    dataset = CaliforniaDataset(california_housing_data)
    
    print(f"dataset.__len__ = {dataset.__len__()}")                     # データセットのデータ数
    print(f"dataset.__getitem__(123) = {dataset.__getitem__(123)}")     # 123番目のデータを取り出し
    
    # データセットをランダムに分割
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
    
    print(f"train_set.__len__ = {train_set.__len__()}")     # trainセットのデータ数
    print(f"val_set.__len__ = {val_set.__len__()}")         # validationセットのデータ数
    print(f"test_set.__len__ = {test_set.__len__()}")       # testセットのデータ数
