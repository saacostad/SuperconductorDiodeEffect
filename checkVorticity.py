""" This script creates the functions to check the vorticity with tunable parameters for a single device """

# pyTDGL imports
import tdgl
from tdgl.solver.solver import ConstantField
from tdgl.visualization.animate import create_animation
from tdgl.sources import LinearRamp, ConstantField

# Other imports
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed   # To run different systems on paralel
import itertools


def runSimulation(I, B, device, simulation_time):

    opts = tdgl.SolverOptions(
                solve_time = simulation_time,
                current_units = 'uA',
                field_units = 'mT',
                gpu = False 
            )

    sol = tdgl.solve(device, opts,
                     applied_vector_potential = B,
                     terminal_currents = {"source": I, "drain": -I})

    return sol.plot_order_parameter()


def process_one(I, B, path, name, device, simulation_time):
    fig, axs = runSimulation(I, B, device, simulation_time)
    fig.canvas.draw()  

    for i, ax in enumerate(axs):
        name = "OP" if i == 0 else "Phase"
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        fig.savefig(
            f"images/checkVorticity/{path}/{I}_{name}.png",
            bbox_inches=bbox
        )

    plt.close(fig)


def checkParameters(device, path = "set1", name="C", currents=[0], mfields=[0.005 * i for i in range(0, 10)], simulation_time = 750):
    """ This function takes a device and runs simulations for different values of 
    extern parameters (bias currents and applied magnetic fields) and outputs a graph of them """
    
    jobs = itertools.product(currents, mfields)

    Parallel(n_jobs=-1, backend="loky")(
    delayed(process_one)(I, B, path, name, device, simulation_time)
        for I, B in jobs
    )

    print("Finished plotting")
