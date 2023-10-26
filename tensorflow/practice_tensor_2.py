import tensorflow as tf

# torchはrequires_grad=Trueとすれば勾配計算を受け付け、backwardを呼んだ後grad属性で勾配を確認できたが、tfはそうはしてくれない
# そのかわり、tf.GradientTapeを使って手動で実装する
x_1 = tf.Variable([[1.,2.],[3.,4.]])
x_2 = tf.Variable([[1.,-2.],[-3.,-4.]])

# このコンテキスト内の演算はtapeに保存され、後でgradientsメソッドで勾配を取得できる
with tf.GradientTape() as tape:
    y = x_1 * x_1 + 2 * x_2                     # dy/dx_1 = 2x_1, dy/dx_2 = 2
    # tfでは、sumなどのように、入力次元よりも出力次元が少なくなる演算はおおむね「reduce_」がつくように命名されている
    z = tf.reduce_sum(y * y * y)                # dz/dy = 3y^2

# tape.gradient({微分される変数}, {微分する変数のリスト})で各勾配を取得 戻り値は{微分する変数のリスト}の順番
[x_1_grad, x_2_grad, y_grad, z_grad] = tape.gradient(z, [x_1, x_2, y, z])

# 計算結果
print(f"x_1 = {x_1}")
print(f"x_2 = {x_2}")
print(f"y = {y}")
print(f"z = {z}")

# 勾配
print(f"x_1_grad = {x_1_grad}")
print(f"x_2_grad = {x_2_grad}")
print(f"y_grad = {y_grad}")
print(f"z_grad = {z_grad}")
