import numpy as np 
import matplotlib.pyplot as plt 
import os 



param = "bridge_width/"
PARAM_NAME = "BRIDGE WIDTH"
param_short = "bw"
folder = "Resultados/densityBW/"

files_and_dirs = os.listdir(folder + param)
files_and_dirs.sort()
plt.style.use('seaborn-v0_8-dark-palette') 
for path in files_and_dirs:

    if path == "critical_currents.csv" or path == "meta.txt":
        continue 

    A = np.loadtxt(folder + param + path, delimiter=",")
    data = A_sorted = A[A[:, 0].argsort()]

    plt.plot(data[:, 0], data[:, 1], label = f"{param_short}: {round(float(path[:-4]), 2)}", marker = ".")
    plt.legend()

plt.title(f"SDE WITH VARIABLE {PARAM_NAME}")
plt.xlabel(rf"applied current $J_B$ [$\mu$A]")
plt.ylabel(rf"measured voltage $V$ [u.a]")
plt.ylim(-0.02, 0.02)
plt.xlim(-0.02, 0.02)
plt.grid()
plt.show()

plt.clf() 


A = np.loadtxt(folder + param + "critical_currents.csv", delimiter=",")
data = A_sorted = A[A[:, 0].argsort()]

efficiency = abs((abs(data[:, 1]) - abs(data[:, 2]) )) / np.mean([abs(data[:, 1]), abs(data[:, 2])])

plt.plot(data[:, 0], efficiency, lw = 2.0, marker = "o")

plt.title(f"SDE EFFICIENCY WITH VARIABLE {PARAM_NAME}")
plt.xlabel(rf"{PARAM_NAME}")
plt.ylabel(rf"SDE Efficiency")

plt.grid()
plt.show()
