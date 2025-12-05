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
import argparse



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
# critical_param = "bridge_width"
# critical_param_value_list = [0.1 + 0.05 * n for n in range(10)]

PRESICION = 1e-3

THERMTIME = 700 
AVRTIME = 300


MAGNETIC_FIELD = 15.0
INITIAL_CURRENT_CHOICE = 0.1

CORES = 100

parser = argparse.ArgumentParser()
parser.add_argument("--vary", required=True,
                    choices=[
                        "bridge_width", "bulge_radius", "noise_amplitude",
                        "noise_w", "circle_def", "theta_min", "theta_max"
                    ])

parser.add_argument("--start", type=float, required=True,
                    help="Start value for variation")
parser.add_argument("--stop", type=float, required=True,
                    help="Stop value for variation")
parser.add_argument("--num", type=int, required=True,
                    help="Number of values")


args = parser.parse_args()

vary_param = args.vary
critical_param_value_list = np.linspace(args.start, args.stop, args.num)




""" CREATING THE DEVICE 
all code is found in geometr.py """

# Geometry parameters 
bridge_width = 0.27777          # Width of the bridge 
bulge_radius = 0.6          # Radius of the bulge in the middle 

noise_amplitude = 0.1      # Amplitude of the spikes in the assymetrical region 
noise_w = 30.0             # Frequency of the spikes 
circle_def = 0.01           # How often to create a new point (the less, the more pointy)

theta_min = np.pi / 2       # Left angle to start introducing the assymetry
theta_max = 0.0             # Right angle to introduce the assymetry


params = {
    "bridge_width": bridge_width,
    "bulge_radius": bulge_radius,
    "noise_amplitude": noise_amplitude,
    "noise_w": noise_w,
    "circle_def": circle_def,
    "theta_min": theta_min,
    "theta_max": theta_max,
}


os.makedirs(f"{PATH}/Images/{vary_param}", exist_ok=True)

for value in critical_param_value_list:
    # params[vary_param] = value

    device = createDevice(
        layer,
        params["bridge_width"],
        params["bulge_radius"],
        params["noise_amplitude"],
        params["noise_w"],
        params["circle_def"],
        params["theta_min"],
        params["theta_max"],
    ).rotate(90)

    device.make_mesh(max_edge_length=xi / 2)
    
    # device.draw()
    # plt.show()

    findCriticalCurrents(device, value, INITIAL_CURRENT_CHOICE, PATH, presicion = PRESICION, skiptime = THERMTIME, solvetime = AVRTIME, critical_param=vary_param, critical_param_value=value, cores = CORES)


# def runSimulation(I, B, device, simulation_time, paramval, path):
#
#     opts = tdgl.SolverOptions(
#                 solve_time = simulation_time,
#                 current_units = 'uA',
#                 field_units = 'mT',
#                 gpu = False 
#             )
#
#     sol = tdgl.solve(device, opts,
#                      applied_vector_potential = B,
#                      terminal_currents = {"source": I, "drain": -I})
#
#     fig, axs = sol.plot_order_parameter()
#     fig.canvas.draw()  
#
#     for i, ax in enumerate(axs):
#         name = "OP" if i == 0 else "Phase"
#         bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#
#         fig.savefig(
#             f"images/checkVorticity/{path}/{paramval}_{name}.png",
#             bbox_inches=bbox
#         )
#
#     plt.close(fig)
#
#     fig.savefig(f"images/checkVorticity/{path}/{paramval}.png",)
#
#
# PATH = "Varying_current"
# os.makedirs(f"images/checkVorticity/{PATH}", exist_ok=True)
# for value in [-0.020, -0.015, -0.010, 0.0, 0.010, 0.015, 0.020]:
#     # params[vary_param] = value
#
#     device = createDevice(
#         layer,
#         params["bridge_width"],
#         params["bulge_radius"],
#         params["noise_amplitude"],
#         params["noise_w"],
#         params["circle_def"],
#         params["theta_min"],
#         params["theta_max"],
#     ).rotate(90)
#     device.make_mesh(max_edge_length=xi / 2, smooth = 100)
#
#     runSimulation(value, 13.0, device, 500, value, PATH)
