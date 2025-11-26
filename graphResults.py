import numpy as np 
import matplotlib.pyplot as plt 


data = np.loadtxt("possibly something goof.csv", delimiter=",")

plt.scatter(data[:, 0], data[:, 1])
plt.show()
