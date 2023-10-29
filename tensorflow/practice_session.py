import tensorflow as tf_v2
tf = tf_v2.compat.v1
tf.disable_v2_behavior()

# tf.Variableは計算グラフ上の変数 NNで例えると、パラメータ(訓練で更新するもの)
# sessの外では、計算グラフを構築するだけ Variableの計算はまだされていない
x_1 = tf.Variable([[1.,2.],[3.,4.]])
print(f"x_1 = {x_1}")                   # 計算はまだされていない
print(f"type(x_1) = {type(x_1)}")
print(f"x_1.shape = {x_1.shape}")
print(f"x_1.dtype = {x_1.dtype}")
print()

# tf.placeholderは計算グラフ上の入力引数 NNで例えると、入力側のノード
# 型とshapeを指定して、メモリだけ確保する (shapeは省略可)
# session内でfeed_dictに実際の入力値を与える
x_2 = tf.placeholder(dtype=tf.float32, shape=(2,2))
print(f"x_2 = {x_2}")
print(f"type(x_2) = {type(x_2)}")
print(f"x_2.shape = {x_2.shape}")
print(f"x_2.dtype = {x_2.dtype}")
print()

# tf.constantは計算グラフ上の定数 NNで例えると、ハイパーパラメータ
# sessの外では、計算グラフを構築するだけ constantの計算はまだされていない
c = tf.constant(4.)
print(f"c = {c}")
print(f"type(c) = {type(c)}")
print(f"c.shape = {c.shape}")
print(f"c.dtype = {c.dtype}")
print()

# sessの外では、計算グラフを構築するだけ y, zの計算はまだされていない
y = x_1 * x_1 + 2 * x_2                     # dy/dx_1 = 2x_1, dy/dx_2 = 2
z = tf.reduce_sum(y * y * y) + c            # dz/dy = 3y^2
# tf.gradients({微分される変数}, {微分する変数のリスト})で各勾配を取得 戻り値は{微分する変数のリスト}の順番
# sessの外では、計算グラフを構築するだけ gradientsの計算はまだされていない
gradients = tf.gradients(z, [x_1, x_2, y, z])
init = tf.global_variables_initializer()        # Variableの初期化

print(f"y = {y}")                   # 計算はまだされていない
print(f"z = {z}")                   
print(f"gradients = {gradients}")                   


# ここから計算グラフに値を流し込む
with tf.Session() as sess:
    sess.run(init)      # Variableの初期化
    # gradientsを実際に計算 placeholderに与える値はfeed_dict引数に辞書で与える 戻り値はそれぞれの計算結果になっている
    # gradientsの計算に必要なy, zなどをsessionの中で計算していない状態でも正常に動く
    _gradients = sess.run(gradients, feed_dict={x_2: [[1.,-2.],[-3.,-4.]]})
    
    # y, zを実際に計算
    _y, _z = sess.run([y, z], feed_dict={x_2: [[1.,-2.],[-3.,-4.]]})
    # x_1, x_2, cを実際に計算
    _x_1 = sess.run(x_1)
    _x_2 = sess.run(x_2, feed_dict={x_2: [[1.,-2.],[-3.,-4.]]})
    _c = sess.run(c)

# 計算結果
print(f"_x_1 = {_x_1}")         # sessionの戻り値が計算結果になっている
print(f"_x_2 = {_x_2}")
print(f"_c = {_c}")
print(f"_y = {_y}")
print(f"_z = {_z}")

x_1_grad, x_2_grad, y_grad, z_grad = _gradients

# 勾配
print(f"_gradients = {_gradients}")
print(f"x_1_grad = {x_1_grad}")
print(f"x_2_grad = {x_2_grad}")
print(f"y_grad = {y_grad}")
print(f"z_grad = {z_grad}")
