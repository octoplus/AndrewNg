## 向量化逻辑回归

注意

w.shape=(n, 1)

b.shape=(1,)

X.shape=(n, m)

Z.shape=(1, m)

```python
import numpy as np

for iter in range(10000):
	Z = np.dot(w.T, X) + b #利用了广播机制
    A = sigma(Z)
    dZ = A - Y
    dw = 1/m * X * dZ.T
    db = np.mean(dZ)
    w = w - alpha * dw
    b = b - alpha * db
```



## numpy向量提示

尽量不要用1维的array，既不是行向量，也不是列向量。

```Python
import numpy as np

a = np.random.randn(5)
a.shape # (5,)

a.T.shape # (5,)

a = np.random.randn(5, 1)
a.shape # (5,1)
a.T.shape # (1,5)
```

