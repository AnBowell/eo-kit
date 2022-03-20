from statistics import mode
import numpy as np


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4])


z = np.convolve(x, y, mode="valid")
print(z)

z = np.convolve(x, y, mode="full")
print(z)
