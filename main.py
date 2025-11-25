""" Main script of the project """

# pyTDGL imports
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



""" MATERIAL PARAMETERS """
xi = 0.1                # Coherence leght 
london_lambda =  3.0    # Penetration depth
d = 0.01                # thickness

layer = createLayer(xi, london_lambda, d)


""" CREATING THE DEVICE 
all code is found in geometr.py """

# Geometry parameters 
bridge_width = 0.3          # Width of the bridge 
bulge_radius = 0.7          # Radius of the bulge in the middle 

noise_amplitude = 0.02      # Amplitude of the spikes in the assymetrical region 
noise_w = 100.0             # Frequency of the spikes 
circle_def = 0.01           # How often to create a new point (the less, the more pointy)

theta_min = np.pi / 2       # Left angle to start introducing the assymetry
theta_max = 0.0             # Right angle to introduce the assymetry


# WE CREATE THE GIVEN DEVICE
device = createDevice(layer,
                      bridge_width, bulge_radius,
                      noise_amplitude, noise_w, circle_def, 
                      theta_min, theta_max).rotate(90)      # Rotate 90d for a better look in the videos

device.make_mesh(max_edge_length=xi / 2)



# IV   = []                            # list of (I, V) tuples
# sol  = None                          # seed container
#
# I_vals = [{"drain" : i, "source" : -i} for i in np.linspace(-50.0, 50.0, 41)]
#
# R_list = []
# def runSimulation(I_val, B):
#     opts = tdgl.SolverOptions(
#         solve_time=30,
#         skip_time = 250,         field_units='mT',
#         current_units='uA', 
#         gpu = True)
#
#     sol = tdgl.solve(device, opts,
#                      applied_vector_potential=B,
#                      terminal_currents=I_val)          # warm start
#
#     # ---- time-averaged voltage over last 100 τ₀ ----
#     v_mean = sol.dynamics.mean_voltage(0, 1, tmin=0, tmax = 30)  # V
#     return((I_val["drain"], v_mean))

# Parallel execution
# for b in B_vals:
#         
#     results = Parallel(n_jobs=-1)(
#             delayed(lambda I: runSimulation(I, b))(I) for I in I_vals
#     )
#     IV = sorted(results, key=lambda x: x[0])  # sort by current
#
#     np.savetxt(f"One way current B = {b}.csv", np.array(IV).T, delimiter=",")
#
#     I, V = np.array(IV).T
#     plt.plot(I, V, marker='o', label = f"B = {b}")
#     plt.xlabel("bias current (μA)")
#     plt.ylabel("⟨V⟩ (μV)")
#     plt.title(f"IV  mT")
#     plt.legend()
#     plt.grid()
#     plt.savefig("Graph.png")
#     plt.show()





# options = tdgl.SolverOptions(
#         solve_time=solv_time,              # ≥ 200 τ₀ plateau
#         skip_time = 100.0,
#         field_units='mT',
#         output_file = outFile,
#         current_units='uA',
#         gpu = False)
#
#
#
# applied_vector_potential = (
#         # LinearRamp(tmin=0, tmax=1000)
#         ConstantField(B_vals[0], field_units=options.field_units, length_units=device.length_units)
#     )
#
# steady_current_u = {"source": corr_max, "drain": -corr_max}
# steady_current_d = {"source": -corr_max, "drain": corr_max}
#
# def var_current(t):
#     return {"source": corr_max/(solv_time / 2.0) * (t- (solv_time / 2.0) ), "drain": - corr_max/(solv_time / 2.0) * (t- (solv_time / 2.0) )}
#
# zero_current_solution = tdgl.solve(
#     device,
#     options,
#     applied_vector_potential=B_vals[0],
#     terminal_currents = steady_current_d
# )
#
# zero_current_video = make_video_from_solution(zero_current_solution, "current1.mp4")
#
#
#
#
#
# zero_current_solution = tdgl.solve(
#     device,
#     options,
#     applied_vector_potential=B_vals[0],
#     terminal_currents = steady_current_u
# )
#
# zero_current_video = make_video_from_solution(zero_current_solution, "current2.mp4")
