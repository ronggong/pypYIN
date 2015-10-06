from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

sigma2Note = 0.7  # note transition probability standard deviation used in pYin

noteDistanceDistr = norm(loc=0, scale=sigma2Note)

fig, ax = plt.subplots(1, 1)

x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)

ax.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')

ax.legend(loc='best', frameon=False)
plt.title('note transition probability function')
plt.xlabel('note transition distance in semitone')
plt.show()



