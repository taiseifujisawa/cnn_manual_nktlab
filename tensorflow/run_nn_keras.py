import tensorflow as tf
import sys
from set_seed import set_seed
from logging_controller import Logger

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

"""from utils import NnRegModel"""

# 乱数シードを1で固定
set_seed(seed=1)
# ./log/log.txtをlog_fileに設定
sys.stdout = Logger("./log/log_nn.txt")

# データ読み込み
california_housing_data = fetch_california_housing()

### データセットをランダムに分割(sklearnを利用) train+val, testに分ける ###
X_train, X_test, y_train, y_test = train_test_split(california_housing_data.data, california_housing_data.target, train_size=0.999)
"""X_train, X_val, y_train, y_val = train_test_split(_X_train, _y_train, train_size=0.999)"""

### データセットは作らない ###
BATCH_SIZE = 100
"""
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_set = train_set.shuffle(len(y_train)).batch(BATCH_SIZE)
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
"""

### モデルを簡単に作る Sequential API ###
"""model = NnRegModel()"""
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

# printの頻度
#VERBOSE = False

# 訓練フェーズ開始
print("### Train ###")

### ミニバッチやエポックのループは全部fitメソッドがやってくれるので書かない ###
### 戻り値は辞書型で、エポックごとのlossとmetricsを記録したもの ###
history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_split=0.001, shuffle=True, validation_batch_size=BATCH_SIZE)
print(history.history)

"""
# ミニバッチ1つ分の計算を関数化
@tf.function                        # このデコレータをつけると高速になるがデバッグしづらい デバッグ時はコメントアウト
def train_on_batch(inputs, labels):
    with tf.GradientTape() as tape:                 # 勾配を記録
        outputs = model(inputs, training=True)      # 訓練モードはTrue、推論モードはFalse
        loss = loss_func(labels, outputs)           # 引数の順序がtorchと逆なので注意(特にcross entropyなど)

    # 勾配計算 model.trainable_weightsでモデル内の全パラメータを取得
    gradients = tape.gradient(loss, model.trainable_weights)

    # パラメータ更新 optimizer.apply_gradientsメソッドを呼び出し 引数は「勾配、現パラメータ値」の順で、リストやジェネレータで与える
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # [任意]metricの記録
    metric_batch.update_state(outputs, labels)            # 引数の順序がtorchと逆なので注意

    return loss     # 任意

# 訓練フェーズ開始
print("### Train ###")

# エポックのループ
for ep in range(N_EPOCH):
    n_total = 0                 # 訓練はエポックのループの最後なのでリセット
    running_loss = 0

    print("[epoch, batch]")
    # ミニバッチのループ
    for i, data in enumerate(train_set):
        metric_batch.reset_states()
        inputs, labels = data
        loss = train_on_batch(inputs, labels)

        n_total += labels.shape[0]
        running_loss += metric_batch.result()

        if VERBOSE:
            # ミニバッチ毎にlossの平均値をprint
            print('[{:5d}, {:5d}] train loss: {:.3f}'
                    .format(ep+1, i+1, metric_batch.result() / labels.shape[0]))

    # エポック毎にlossの平均値をprint
    print('[{:5d},   all] train loss: {:.3f}'
            .format(ep+1, running_loss / n_total), end=", ")

    n_total = 0                     # 訓練はエポックのループの最後なのでリセット
    running_loss = 0

    # ミニバッチのループ
    for data in val_set:
        inputs, labels = data
        outputs = model(inputs, training=False)     # 訓練モードはTrue、推論モードはFalse
        loss = loss_func(labels, outputs)
        metric_batch.update_state(outputs, labels)
        running_loss += metric_batch.result()

        n_total += labels.shape[0]

    # エポック毎にlossの平均値をprint
    print('val. loss: {:.3f}'
            .format(running_loss / n_total))
"""


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

"""
predictions_before = []
for data in test_set:
    inputs, labels = data
    outputs = model(inputs, training=False)
    for o, l in zip(outputs.numpy(), labels.numpy()):
        predictions_before.append(o.item())
"""

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

"""
predictions = []
answers = []
for data in test_set:
    inputs, labels = data
    outputs = model_test(inputs, training=False)
    for o, l in zip(outputs.numpy(), labels.numpy()):
        predictions.append(o.item())
        answers.append(l.item())
    loss = loss_func(labels, outputs)
    metric_batch.update_state(outputs, labels)
    running_loss += metric_batch.result()
    n_total += labels.shape[0]
print('test loss: {:.3f}'
        .format(running_loss / n_total))
"""

# 推論結果と教師ラベル、ついでに訓練前のモデルの推論結果をcsv形式で出力
output_filepath = "./log/test_result.csv"
with open(output_filepath, "w") as f:
    f.write("index,prediction_before,prediction_after,answer\n")
    for i, (pb, p, a) in enumerate(zip(predictions_before, predictions, y_test)):
        f.write(f"{i + 1},{pb.item()},{p.item()},{a}\n")
