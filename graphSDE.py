""" Functions to correctly graph the supercritical currents """


# pyTDGL imports
import tdgl
from tdgl.solver.solver import ConstantField
from tdgl.visualization.animate import create_animation
from tdgl.sources import LinearRamp, ConstantField

# Other imports
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count   

import warnings
warnings.filterwarnings("ignore")





IV   = []                            # list of (I, V) tuples
sol  = None                          # seed container

I_vals = [{"drain" : i, "source" : -i} for i in np.linspace(-50.0, 50.0, 41)]
R_list = []


def runSimulation(I_val, B, device, solvetime, skiptime):
    opts = tdgl.SolverOptions(
        solve_time=solvetime,
        skip_time = skiptime,         
        field_units='mT',
        current_units='uA', 
        gpu = False)

    sol = tdgl.solve(device, opts,
                     applied_vector_potential=B,
                     terminal_currents={"source": I_val, "drain": -I_val})          # warm start

    # ---- time-averaged voltage over last 100 τ₀ ----
    v_mean = sol.dynamics.mean_voltage(0, 1, tmin=0, tmax = solvetime)  # V
    return((I_val, v_mean))


def findCriticalCurrents(device, B, c0, path, skiptime = 750, solvetime = 100):
    """ This function will find and graph the critical currents for a system. 
    It uses a bijection algorithm to avoid simulating currents we do not need. 
    It uses half the CPU count to process one polarity. 
    Once it finishes simulating all the found voltages parallely, it checks in which region we should increase the number of points
    to have a more precise measurement of the critical current """

    cpuS = int(cpu_count() / 2)     # No of CPUs we'll use per polarization


    cint = c0 / cpuS                # Currents intervals to simualte
    currents = [-c0 + cint*p for p in range(cpuS)] + [cint*(p+1) for p in range(cpuS)]  # First currents we'll check
    
    print(f"================================\n INITIAL CURRENTS")

    while True:
        print(currents)
        # Run the paralelized process and get the results
        results = Parallel(n_jobs=-1)(
                delayed(lambda I: runSimulation(I, B, device, solvetime, skiptime))(I) for I in currents
        )

        # Format the results just to check
        IV = sorted(results, key=lambda x: x[0])  # sort by current
        Is, V = np.array(IV).T

        # Save current results to a file
        with open(path, "a") as f:
            np.savetxt(f, np.array(IV), delimiter=",", fmt="%.6f")       



        # Check if we found a critical current 
        mask = abs(V) > 0.01
        vTest = np.where(mask)[0]

        print(f'\n\n -------------------------------- \n {V} \n {Is}\n')

        if vTest.size == 0:
            # If we haven't found the critical current, look in the next big interval
            print("--------------------------------------\n Haven't found critical currents \n\n ========================== \n NEW ATTEMP \n ")
            c0 += c0
            currents = [-c0 + cint*p for p in range(cpuS)] + [(c0/2) + cint*(p+1) for p in range(cpuS)]
            continue
        

        changes = np.where(abs(np.diff(mask.astype(int))))[0]


        if changes.size == 0:
            print("--------------------------------------\n Haven't found critical currents \n\n ========================== \n NEW ATTEMP \n ")
            c0 /= 2
            cint = c0 / cpuS  
            currents = [-c0 + cint*p for p in range(cpuS)] + [(c0/2) + cint*(p+1) for p in range(cpuS)]
            continue

        print(f"------------------------------------\n FOUND CRITICAL CURRENTS\n----------------------------------------- \n ")
        # This will give me the indexes where there's a jump



        low_neg_currents = Is[int(changes[0])]
        up_neg_currents = Is[int(changes[0])+1]

        low_pos_currents = Is[int(changes[1])]
        up_pos_currents = Is[int(changes[1])+1]
       

        cint_neg_temp = abs(up_neg_currents - low_neg_currents)
        cint_pos_temp = abs(up_pos_currents - low_pos_currents) 
        
        if cint_neg_temp < 1e-2 and cint_pos_temp < 1e-2:
            print("\n\n\n===============================================\n FINISHED SIMULATION")
            break

        cint_neg = cint_neg_temp / cpuS
        cint_pos = cint_pos_temp / cpuS
        print(low_pos_currents)
        print(cint_pos)
        currents = [low_neg_currents + cint_neg*p for p in range(cpuS)] + [low_pos_currents + cint_pos*(p+1) for p in range(cpuS)]
