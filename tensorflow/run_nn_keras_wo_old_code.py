import tensorflow as tf
import sys
from set_seed import set_seed
from logging_controller import Logger

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 乱数シードを1で固定
set_seed(seed=1)
# ./log/log.txtをlog_fileに設定
sys.stdout = Logger("./log/log_nn.txt")

# データ読み込み
california_housing_data = fetch_california_housing()

### データセットをランダムに分割(sklearnを利用) train+val, testに分ける ###
X_train, X_test, y_train, y_test = train_test_split(california_housing_data.data, california_housing_data.target, train_size=0.999)

### データセットは作らない ###
BATCH_SIZE = 100

### モデルを簡単に作る ###
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=1))

# loss関数(MSE)のインスタンス化
loss_func = tf.keras.losses.MeanSquaredError()

# [任意]評価値(MSE)クラスのインスタンス化
metric = tf.keras.metrics.MeanSquaredError()

# 最適化アルゴリズム(Adam)のインスタンス化
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

### compileでoptimizerやloss_funcなど必要なクラスのインスタンスを全部modelに持たせる ###
model.compile(optimizer=optimizer, loss=loss_func, metrics=[metric])

# エポック数
N_EPOCH = 50

# 訓練フェーズ開始
print("### Train ###")

### ミニバッチやエポックのループは全部fitメソッドがやってくれるので書かない ###
### 戻り値は辞書型で、エポックごとのlossとmetricsを記録したもの ###
history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_split=0.001, shuffle=True, validation_batch_size=BATCH_SIZE)
print(history.history)

# モデルを保存
model_save_path = "./log/model_nn"
model.save_weights(model_save_path)

# 試しに初期状態のモデルを使って回帰してみる(的外れな結果になる)
# 初期化
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=1))

metric.reset_states()
model.compile(optimizer=optimizer, loss=loss_func, metrics=[metric])

### evaluate, predict関数がやってくれるのでループは書かない ###
test_loss, test_metrics = model.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE)
predictions_before = model.predict(x=X_test, batch_size=BATCH_SIZE)

# 保存したモデルをロードするときはまずはインスタンス化
model_test = tf.keras.models.Sequential()
model_test.add(tf.keras.layers.Dense(units=100, activation="relu"))
model_test.add(tf.keras.layers.Dense(units=100, activation="relu"))
model_test.add(tf.keras.layers.Dense(units=1))
model_test.load_weights(model_save_path)

metric.reset_states()
model_test.compile(optimizer=optimizer, loss=loss_func, metrics=[metric])

# テストフェーズ開始
print("### Test ###")
### evaluate, predict関数がやってくれるのでループは書かない ###
test_loss, test_metrics = model_test.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE)
predictions = model_test.predict(x=X_test, batch_size=BATCH_SIZE)

# 推論結果と教師ラベル、ついでに訓練前のモデルの推論結果をcsv形式で出力
output_filepath = "./log/test_result.csv"
with open(output_filepath, "w") as f:
    f.write("index,prediction_before,prediction_after,answer\n")
    for i, (pb, p, a) in enumerate(zip(predictions_before, predictions, y_test)):
        f.write(f"{i + 1},{pb.item()},{p.item()},{a}\n")
