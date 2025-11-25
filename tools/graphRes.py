import numpy as np 
import matplotlib.pyplot as plt 


data = np.loadtxt("results.csv", delimiter=",").T


plt.scatter(data[:, 0], data[:, 1])
plt.grid()
plt.show()
