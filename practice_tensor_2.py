import torch

# requires_grad=Trueで勾配計算を受け付ける
# その場合tensorの要素の型は小数の型でなければならない
x_1 = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)

# x.requires_grad = Trueで後から設定することもできる
x_2 = torch.tensor([[1.,-2.],[-3.,-4.]])
x_2.requires_grad = True

y = x_1 * x_1 + 2 * x_2                     # dy/dx_1 = 2x_1, dy/dx_2 = 2
z = torch.sum(y * y * y)                    # dz/dy = 3y^2

# 計算結果
print(f"x_1 = {x_1}")
print(f"x_2 = {x_2}")
print(f"y = {y}")
print(f"z = {z}")

# 計算グラフが自動で構築される
print(f"x_1.grad_fn = {x_1.grad_fn}")       # 入力ノードにはgrad_fnはない
print(f"x_2.grad_fn = {x_2.grad_fn}")
print(f"y.grad_fn = {y.grad_fn}")
print(f"z.grad_fn = {z.grad_fn}")

# 計算グラフの途中の勾配を取得するにはretain_gradが必要
y.retain_grad()
z.retain_grad()

# backwardを呼ぶまではgrad属性にはまだ何もない
print(f"x_1.grad = {x_1.grad}")
print(f"x_2.grad = {x_2.grad}")
print(f"y.grad = {y.grad}")
print(f"z.grad = {z.grad}")

# zに対するそれまでの計算に関わった変数による勾配を一度に計算
z.backward()

# zに対する各変数の勾配が取得できる
print(f"x_1.grad = {x_1.grad}")
print(f"x_2.grad = {x_2.grad}")
print(f"y.grad = {y.grad}")
print(f"z.grad = {z.grad}")
