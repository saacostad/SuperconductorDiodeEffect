""" Main script of the project """

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

# Custom headers imports 
from tools.animation import *
from geometry import *
from checkVorticity import checkParameters
from graphSDE import *

""" MATERIAL PARAMETERS """
# xi = 0.1                # Coherence leght 
# london_lambda =  10.0    # Penetration depth
# d = 0.01                # thickness
#
# layer = createLayer(xi, london_lambda, d)


xi = 0.12            # God knows what the fucking shit this is
london_lambda =  2.4  # Penetration longitude I think 
d = 0.01             # thickness
layer = createLayer(xi, london_lambda, d)


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


# WE CREATE THE GIVEN DEVICE
# device = createDevice(layer,
#                       bridge_width, bulge_radius,
#                       noise_amplitude, noise_w, circle_def, 
#                       theta_min, theta_max).rotate(90)      # Rotate 90d for a better look in the videos
device = createDevice(layer,
                      noise_w=25.0, noise_amplitude=0.15,
                      bridge_width=0.8, bulge_radius=1.0).rotate(90)
device.make_mesh(max_edge_length=xi / 2)


# 0.375
findCriticalCurrents(device, 0.1, 5.0, "possibly something goof.csv", skiptime = 100, solvetime = 50)

# checkParameters(device, path = "Current2", currents = [n for n in np.linspace(-5.0, 5.0, 30)], mfields = [0.0385])


