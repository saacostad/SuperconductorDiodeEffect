import numpy as np 
import matplotlib.pyplot as plt 


data = np.loadtxt("New Test 2.csv", delimiter=",")

plt.scatter(data[:, 0], data[:, 1])
plt.grid()
plt.show()
