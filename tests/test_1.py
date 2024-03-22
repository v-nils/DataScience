import numpy as np


x = np.array([1, 2, 3])


y = x.reshape(-1,1) @ x.reshape(1,-1)

print(y)