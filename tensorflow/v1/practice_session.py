import tensorflow as tf_v2
tf = tf_v2.compat.v1
tf.disable_v2_behavior()

# sessの外では、計算グラフを構築するだけ Variableの実態はまだない
x_1 = tf.Variable([[1.,2.],[3.,4.]])
print(f"x_1 = {x_1}")                   # 実態はまだない
print(f"type(x_1) = {type(x_1)}")
print(f"x_1.shape = {x_1.shape}")
print(f"x_1.dtype = {x_1.dtype}")
print()

# placeholderは計算グラフの入力に使う 引数のようなもの 型は指定しないといけない
x_2 = tf.placeholder(tf.float32)
print(f"x_2 = {x_2}")
print(f"type(x_2) = {type(x_2)}")
print(f"x_2.shape = {x_2.shape}")
print(f"x_2.dtype = {x_2.dtype}")
print()

# sessの外では、計算グラフを構築するだけ y, zの実態はまだない
y = x_1 * x_1 + 2 * x_2                     # dy/dx_1 = 2x_1, dy/dx_2 = 2
z = tf.reduce_sum(y * y * y)                # dz/dy = 3y^2
# tf.gradients({微分される変数}, {微分する変数のリスト})で各勾配を取得 戻り値は{微分する変数のリスト}の順番
# sessの外では、計算グラフを構築するだけ gradientsの実態はまだない
gradients = tf.gradients(z, [x_1, x_2, y, z])
init = tf.global_variables_initializer()        # Variableの初期化

# ここから計算グラフに値を流し込む
with tf.Session() as sess:
    sess.run(init)      # Variableの初期化
    # x_1, x_2, y, zを実際に計算 placeholderに与える値はfeed_dict引数に辞書で与える
    # 戻り値はそれぞれの計算結果になっている
    _x_1, _x_2, _y, _z = sess.run([x_1, x_2, y, z], feed_dict={x_2: [[1.,-2.],[-3.,-4.]]})
    # gradientsを実際に計算
    x_1_grad, x_2_grad, y_grad, z_grad= sess.run(gradients, feed_dict={x_2: [[1.,-2.],[-3.,-4.]]})

# 計算結果
print(f"_x_1 = {_x_1}")         # sessionの戻り値が計算結果になっている
print(f"_x_2 = {_x_2}")
print(f"_y = {_y}")
print(f"_z = {_z}")

# 勾配
print(f"x_1_grad = {x_1_grad}")
print(f"x_2_grad = {x_2_grad}")
print(f"y_grad = {y_grad}")
print(f"z_grad = {z_grad}")
