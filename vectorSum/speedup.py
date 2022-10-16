import numpy as np
import matplotlib.pyplot as plt

plt.xlabel('N')
plt.ylabel('Ускорение')
x = np.array([1000, 10000, 50000, 100000, 500000, 1000000])
cpu = np.array([1.2e-05, 2.5e-05, 0.000125, 0.000255, 0.001259, 0.002538])
gpu = np.array([1.6768e-05, 1.8176e-05, 2e-05, 2.3456e-05, 6.048e-05, 9.6224e-05])
plt.plot(x, cpu/gpu)
plt.show()