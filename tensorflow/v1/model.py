import tensorflow as tf_v2
tf = tf_v2.compat.v1
tf.disable_v2_behavior()

# tfのモデルはtf.keras.Modelを継承するのがルール Subclass API
class NnRegModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # super().__init__()の後に、メンバ変数としてネットワークに用いる層のクラス(or関数)をインスタンス化して持っておく
        # torchのnn.Linearは、tf.kerasではtf.keras.layers.Denseという名前になる
        # 引数には入力ノード数はいらない！unitsに出力ノード数だけを指定する
        # また、引数activationに"relu"や"softmax"などを指定するとセットで定義できる
        self.fc1 = tf.keras.layers.Dense(units=100, activation="relu")
        self.fc2 = tf.keras.layers.Dense(units=100)
        self.fc3 = tf.keras.layers.Dense(units=1)
        # もちろんReLUを別に定義してもいい
        self.relu = tf.keras.layers.ReLU()

    # torchではforward関数だったが、tfでは__call__関数に計算グラフ構築処理を書く
    # trainingは書かなくてもいい droputなど訓練と推論でモードが変わるものは書く
    def __call__(self, x, training=None):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x        # Loss関数の直前の計算結果をreturnする 最終層活性化関数は書かなきゃいけないけど今回は回帰だからない

class CnnClsModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # super().__init__()の後に、メンバ変数としてネットワークに用いる層のクラス(or関数)をインスタンス化して持っておく
        # torchのnn.Conv2dは、tf.kerasではtf.keras.layers.Conv2Dという名前になる
        # 引数には入力ノード数はいらない！filtersに出力フィルタ数と、kernel_sizeを指定する
        # padding="same"は入出力サイズが同じになるように自動でpadding数が計算される
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.pool = tf.keras.layers.MaxPool2D()
        self.relu = tf.keras.layers.ReLU()
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=100, activation="relu")
        self.fc2 = tf.keras.layers.Dense(units=100, activation="relu")
        # 最終層の活性化関数はsoftmax torchとは違い省略できないので注意
        self.fc3 = tf.keras.layers.Dense(units=10, activation="softmax")

    # torchではforward関数だったが、tfでは__call__関数に計算グラフ構築処理を書く
    # trainingは書かなくてもいい droputなど訓練と推論でモードが変わるものは書く
    def __call__(self, x, training=None):
        x = x / 255
        x = (x - 0.5) / 0.5
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x        # Loss関数の直前の計算結果をreturnする torchと違い最終層活性化関数は書かなきゃいけない!
