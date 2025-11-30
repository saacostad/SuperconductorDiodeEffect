""" Main script of the project """

# Avoid warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="cupyx.jit.rawkernel is experimental. The interface can change in the future."
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="cupyx.jit._interface"
)
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*cupyx\.jit\.rawkernel is experimental.*"
)

# pyTDGL imports
from graphSDE import findCriticalCurrents
import tdgl
from tdgl.solver.solver import ConstantField
from tdgl.visualization.animate import create_animation
from tdgl.sources import LinearRamp, ConstantField

# Other imports
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed    # To run different systems on paralel
import os 

# Custom headers imports 
from tools.animation import *
from geometry import *
from checkVorticity import checkParameters
from graphSDE import *



""" MATERIAL PARAMETERS """
xi = 0.1                # Coherence leght 
london_lambda =  10.0    # Penetration depth
d = 0.01                # thickness
layer = createLayer(xi, london_lambda, d)


""" CODE PARAMETERS """
PATH = "test"
critical_param = "bridge_width"
critical_param_value_list = [0.1 + 0.05 * n for n in range(10)]


THERMTIME = 10 
AVRTIME = 10


MAGNETIC_FIELD = 5.0
INITIAL_CURRENT_CHOICE = 0.1


""" CREATING THE DEVICE 
all code is found in geometr.py """

# Geometry parameters 
bridge_width = 0.4          # Width of the bridge 
bulge_radius = 0.6          # Radius of the bulge in the middle 

noise_amplitude = 0.1      # Amplitude of the spikes in the assymetrical region 
noise_w = 35.0             # Frequency of the spikes 
circle_def = 0.01           # How often to create a new point (the less, the more pointy)

theta_min = np.pi / 2       # Left angle to start introducing the assymetry
theta_max = 0.0             # Right angle to introduce the assymetry





os.makedirs(f"{PATH}/{critical_param}", exist_ok=True)

for value in critical_param_value_list:

    # WE CREATE THE GIVEN DEVICE
    # TODO: remember to change the variable parameter value
    device = createDevice(layer,
                          value, bulge_radius,
                          noise_amplitude, noise_w, circle_def, 
                          theta_min, theta_max).rotate(90)      # Rotate 90d for a better look in the videos

    device.make_mesh(max_edge_length=xi / 2)


    findCriticalCurrents(device, 5.0, 0.1, PATH, presicion = 1e-2, skiptime = THERMTIME, solvetime = AVRTIME, critical_param=critical_param, critical_param_value=value)



# checkParameters(device, path = "realCurrentsVar", name="C", currents = [n for n in np.linspace(-0.03, 0.03, 7)], mfields = [5.0], simulation_time=500)


