import tensorflow as tf

print("tensorflow version: {}".format(tf.__version__))      # ver.2であることを確認

a = tf.Variable([1])                # テンソルの作成にはtf.Variableを使う
print(f"type(a) = {type(a)}")       # tensorの型
print(f"a.shape = {a.shape}")       # tensorの形状
print(f"a.dtype = {a.dtype}")       # tensorの要素の型
print()

b = tf.Variable([1., 2., 3.])
print(f"type(b) = {type(b)}")
print(f"b.shape = {b.shape}")
print(f"b.dtype = {b.dtype}")       # 1.と書くとfloat型(tf.float32)になる
print()

c = tf.Variable([[1, 2, 3],
                  [4, 5, 6]])
print(f"type(c) = {type(c)}")
print(f"c.shape = {c.shape}")
print(f"c.dtype = {c.dtype}")       # 1と書くとint型(tf.int32)になる
print()

c_ex = tf.expand_dims(c, axis=0)          # (2,3)tensorに1次元追加
print(f"type(c_ex) = {type(c_ex)}")
print(f"c_ex.shape = {c_ex.shape}")
print(f"c_ex.dtype = {c_ex.dtype}")
print()

d = tf.Variable([[[1., 2., 3],[4, 5, 6]],
                  [[7, 8, 9],[0, 1, 2]],
                  [[3, 4, 5],[6, 7., 8]],
                  [[9, 0, 1],[2, 3, 4]]
                  ])
print(f"type(d) = {type(d)}")
print(f"d.shape = {d.shape}")
print(f"d.dtype = {d.dtype}")       # int型とfloat型が混在しているとfloat型になる
print()

e = tf.Variable([[
                  [[1, 2, 3],[4, 5, 6]],
                  [[7, 8, 9],[0, 1, 2]],
                  [[3, 4, 5],[6, 7, 8]],
                  [[9, 0, 1],[2, 3, 4]]
                  ],
                  [
                  [[1, 2, 3],[4, 5, 6]],
                  [[7, 8, 9],[0, 1, 2]],
                  [[3, 4, 5],[6, 7, 8]],
                  [[9, 0, 1],[2, 3, 4]]
                  ]
                  ])
print(f"type(e) = {type(e)}")
print(f"e.shape = {e.shape}")
print(f"e.dtype = {e.dtype}")
print()

e_np = e.numpy()            # numpy.ndarrayに変換
print(f"type(e_np) = {type(e_np)}")

e_tt = tf.Variable(e_np)   # tf.Tensorに変換
print(f"type(e_tt) = {type(e_tt)}")
