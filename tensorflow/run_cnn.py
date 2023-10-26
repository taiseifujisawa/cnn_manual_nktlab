import tensorflow as tf
import numpy as np
import sys
from set_seed import set_seed
from logging_controller import Logger

from sklearn.model_selection import train_test_split

from utils import CnnClsModel

# 乱数シードを1で固定
set_seed(seed=1)
# ./log/log.txtをlog_fileに設定
sys.stdout = Logger("./log/log_cnn.txt")

# データ読み込み もともとtrainとtestに分かれている
[_X_train, _y_train], [X_test, y_test] = tf.keras.datasets.cifar10.load_data()

# データセットをランダムに分割(sklearnを利用) trainとvalを分ける
X_train, X_val, y_train, y_val = train_test_split(_X_train, _y_train, train_size=0.8)

# データセット(兼データローダー)作成
BATCH_SIZE = 100
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(BATCH_SIZE)
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# モデルのインスタンス化
model = CnnClsModel()

# loss関数(MSE)のインスタンス化
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

# 最適化アルゴリズム(Adam)のインスタンス化
# torchでは引数にmodel.parameters()を渡したがtfはそうしない
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# エポック数
N_EPOCH = 50

# printの頻度
VERBOSE = False

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

    return outputs, loss     # 任意


# 訓練フェーズ開始
print("### Train ###")

# エポックのループ
for ep in range(N_EPOCH):
    n_total = 0                 # 訓練はエポックのループの最後なのでリセット
    n_correct = 0
    running_loss = 0

    print("[epoch, batch]")
    # ミニバッチのループ
    for i, data in enumerate(train_set):
        inputs, labels = data
        outputs, loss = train_on_batch(inputs, labels)

        n_total += labels.shape[0]
        running_loss += loss
        predictions = tf.argmax(outputs, axis=1)
        n_correct += (predictions.numpy() == tf.squeeze(labels).numpy()).sum().item()

        if VERBOSE:
            # ミニバッチ毎にlossの平均値をprint
            print('[{:5d}, {:5d}] train loss: {:.3f}'
                    .format(ep+1, i+1, loss / labels.shape[0]))

    # エポック毎にlossの平均値をprint
    print('[{:5d},   all] train loss: {:.3f}, train accuracy: {:.4f}'
            .format(ep+1, running_loss / n_total, n_correct / n_total), end=", ")

    n_total = 0                     # 訓練はエポックのループの最後なのでリセット
    n_correct = 0
    running_loss = 0

    # ミニバッチのループ
    for data in val_set:
        inputs, labels = data
        outputs = model(inputs, training=False)     # 訓練モードはTrue、推論モードはFalse
        loss = loss_func(labels, outputs)
        running_loss += loss
        predictions = tf.argmax(outputs, axis=1)
        n_correct += (predictions.numpy() == tf.squeeze(labels).numpy()).sum().item()
        n_total += labels.shape[0]

    # エポック毎にlossの平均値をprint
    print('val. loss: {:.3f}, val. accuracy: {:.4f}'
            .format(running_loss / n_total, n_correct / n_total))

    n_total = 0                     # エポックのループの最後なのでリセット
    n_correct = 0
    running_loss = 0

# モデルを保存
model_save_path = "./log/model_cnn"
model.save_weights(model_save_path)

# テストフェーズ開始
print("### Test ###")
preds = []
anss = []
confs = []
for data in test_set:
    inputs, labels = data
    outputs = model(inputs, training=False)
    for o, l in zip(outputs.numpy(), labels.numpy()):
        preds.append(np.argmax(o))
        anss.append(l.item())
        confs.append(o.max().item())
    loss = loss_func(labels, outputs)
    running_loss += loss
    n_total += labels.shape[0]
    predictions = tf.argmax(outputs, axis=1)
    n_correct += (predictions.numpy() == tf.squeeze(labels).numpy()).sum().item()
print('test loss: {:.3f}, test accuracy: {:.4f}'
        .format(running_loss / n_total, n_correct / n_total))

# 推論結果と教師ラベルをcsv形式で出力
output_filepath = "./log/test_result_cnn.csv"
with open(output_filepath, "w") as f:
    f.write("index,confidence,prediction,answer\n")
    for i, (p, a, c) in enumerate(zip(preds, anss, confs)):
        f.write("{},{:.4f},{},{}\n".format(i + 1, c, p, a))
