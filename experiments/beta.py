import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

alpha = 1
beta = 0.5

x = np.linspace(0, 1, 1000)
pdf = stat.beta.pdf(x, alpha, beta)

plt.figure()
plt.plot(x, pdf)
plt.show()
