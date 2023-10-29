import torch

a = torch.tensor([1])
print(f"type(a) = {type(a)}")       # tensorの型
print(f"a.shape = {a.shape}")       # tensorの形状
print(f"a.ndim = {a.ndim}")         # tensorの次元
print(f"a.dtype = {a.dtype}")       # tensorの要素の型
print(f"a.numel() = {a.numel()}")   # tensorの要素数
print()

b = torch.tensor([1., 2., 3.])
print(f"type(b) = {type(b)}")
print(f"b.shape = {b.shape}")
print(f"b.ndim = {b.ndim}")
print(f"b.dtype = {b.dtype}")       # 1.と書くとfloat型(torch.float32)になる
print(f"b.numel() = {b.numel()}")
print()

c = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"type(c) = {type(c)}")                  
print(f"c.shape = {c.shape}")
print(f"c.ndim = {c.ndim}")
print(f"c.dtype = {c.dtype}")       # 1と書くとint型(torch.int64)になる
print(f"c.numel() = {c.numel()}")
print()

c_ex = c.expand([1, 2, 3])          # (2,3)tensorに1次元追加
print(f"type(c_ex) = {type(c_ex)}")                  
print(f"c_ex.shape = {c_ex.shape}")
print(f"c_ex.ndim = {c_ex.ndim}")
print(f"c_ex.dtype = {c_ex.dtype}")
print(f"c_ex.numel() = {c_ex.numel()}")
print()

d = torch.tensor([[[1., 2., 3],[4, 5, 6]],
                  [[7, 8, 9],[0, 1, 2]],
                  [[3, 4, 5],[6, 7., 8]],
                  [[9, 0, 1],[2, 3, 4]]
                  ])
print(f"type(d) = {type(d)}")                  
print(f"d.shape = {d.shape}")
print(f"d.ndim = {d.ndim}")
print(f"d.dtype = {d.dtype}")       # int型とfloat型が混在しているとfloat型になる
print(f"d.numel() = {d.numel()}")
print()

e = torch.tensor([[
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
print(f"e.ndim = {e.ndim}")
print(f"e.dtype = {e.dtype}")
print(f"e.numel() = {e.numel()}")
print()

e_np = e.numpy()            # numpy.ndarrayに変換
print(f"type(e_np) = {type(e_np)}")

e_tt = torch.tensor(e_np)   # torch.Tensorに変換
print(f"type(e_tt) = {type(e_tt)}")
