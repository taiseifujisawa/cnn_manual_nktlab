import tensorflow as tf
import numpy as np
import sys
from set_seed import set_seed
from logging_controller import Logger

# 乱数シードを1で固定
set_seed(seed=1)
# ./log/log.txtをlog_fileに設定
sys.stdout = Logger("./log/log_cnn.txt")

# データ読み込み もともとtrainとtestに分かれている
[X_train, y_train], [X_test, y_test] = tf.keras.datasets.cifar10.load_data()

# モデルの定義 Functional API
inputs = tf.keras.Input(shape=(32,32,3))  # Functional APIでの定義ではInputレイヤーが必要
x = inputs / 255                     # 正規化
x = (x - 0.5) / 0.5                  # 平均0.5 標準偏差0.5で標準化
# Conv2Dのインスタンス化はfilters(出力フィルタ数), kernel_size(カーネルサイズ)が必須で、入力フィルタ数は書かない
# padding="same"は入出力サイズが同じになるように自動でpadding数が計算される
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
# tf.keras.layers.MaxPool2D()でインスタンス化、そのインスタンスに対して引数xを取って__call__を呼び出し
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=100, activation="relu")(x)
x = tf.keras.layers.Dense(units=100, activation="relu")(x)
# 最終層の活性化関数はsoftmax torchとは違い省略できないので注意
outputs = tf.keras.layers.Dense(units=10, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# loss関数(CrossEntropy)のインスタンス化 sparseはラベルがone-hotでなくてもいいことを示す
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

# [任意]評価値(Accuracy)クラスのインスタンス化
metric = tf.keras.metrics.SparseCategoricalAccuracy()

# 最適化アルゴリズム(Adam)のインスタンス化
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# compile
model.compile(optimizer=optimizer, loss=loss_func, metrics=[metric])

# エポック数
N_EPOCH = 50
# バッチサイズ
BATCH_SIZE = 100

# 訓練フェーズ開始
print("### Train ###")

# ミニバッチやエポックのループは全部fitメソッドがやってくれるので書かない
# 戻り値は辞書型で、エポックごとのlossとmetricsを記録したもの
# verbose=2ではエポックごとにlog出力
history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_split=0.2, shuffle=True, validation_batch_size=BATCH_SIZE, verbose=2)

print(history.history)

# モデルを保存
model_save_path = "./log/model_cnn"
model.save_weights(model_save_path)

# テストフェーズ開始
print("### Test ###")
# evaluate, predict関数がやってくれるのでループは書かない
test_loss, test_metrics = model.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE, verbose=2)
predictions = model.predict(x=X_test, batch_size=BATCH_SIZE, verbose=0)

# 推論結果と教師ラベルをcsv形式で出力
output_filepath = "./log/test_result_cnn.csv"
with open(output_filepath, "w") as f:
    f.write("index,prediction,answer\n")
    for i, (p, a) in enumerate(zip(predictions, y_test)):
        f.write("{},{},{}\n".format(i + 1, np.argmax(p), a.item()))
