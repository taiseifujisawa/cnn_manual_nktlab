import tensorflow as tf

# tfのモデルはtf.keras.Modelを継承するのがルール Subclass API
class NnRegModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # super().__init__()の後に、メンバ変数としてネットワークに用いる層のクラス(or関数)をインスタンス化して持っておく
        # torchのnn.Linearは、tf.kerasではtf.keras.layersという名前になる
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
