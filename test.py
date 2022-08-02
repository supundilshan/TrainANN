import numpy as np

x = np.arange(6).reshape(2,3).flatten()
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
print(x.shape)
z = np.polyfit(x, y, 3)
print(z)
out = np.concatenate(input_list).ravel().tolist()