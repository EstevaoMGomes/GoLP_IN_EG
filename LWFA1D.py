# Laser Wakefield Accelerator
# taskset --cpu-list 0-3 screen python-jl LWFA1D.py to run on 4 cores

import sympy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pysr
from sklearn.model_selection import train_test_split
import os
import time
from mdutils.mdutils import MdUtils
plt.rcParams['text.usetex'] = True

mdFile = MdUtils(file_name='LWFA1DAnalysis', title='Laser Wakefield Accelerator Data Analysis')

# Reading TwoStream data - All (1000) lines
inputs = pd.read_csv("/home/estevao/GoLP/LWFA1D_data.csv", usecols=[0, 1])
target = pd.read_csv("/home/estevao/GoLP/LWFA1D_data.csv", usecols=[2])

# Plotting the data
fig, ax1 = plt.subplots()
ax1.plot(inputs.iloc[:,0].values, inputs.iloc[:,1].values, label = "$E_1$" )
ax1.set_xlabel("x")
ax1.set_ylabel("E1")
ax2 = ax1.twinx()
ax2.plot(inputs.iloc[:,0].values, target.iloc[:,0].values,'r', label = "$|n|$" , alpha = 0.8)
ax2.set_ylabel("$|n|$")
ax2.set_ylim(0,2)
plt.title("Longitudinal Electric Field and Plasma Density")
plt.grid(True)
fig.legend(loc = (0.75,0.70))
fig.tight_layout()
plt.savefig("LWFA1D.png")
plt.close()
plt.clf()

np.random.seed(1)
model = pysr.PySRRegressor(
    # Run indefinitely
    niterations=10000000,

    # Stop early if we find a good and simple equation
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-4 && complexity < 10"
    ),
    # Alternatively, stop after 24 hours have passed.
    timeout_in_seconds=60 * 60 * 24,
    populations=12, # 3 times the number of cores used
    binary_operators=["+", "*","-", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    constraints={
        "/": (-1, 9),
        "exp": 9,
        "cos": 9,
        "sin": 9,
        "log": 9,
    },
    nested_constraints={
        "cos": {"cos": 1, "sin": 1, "exp": 1, "log": 1},
        "sin": {"cos": 1, "sin": 1, "exp": 1, "log": 1},
        "exp": {"cos": 1, "sin": 1, "exp": 1, "log": 1},
        "log": {"cos": 1, "sin": 1, "exp": 1, "log": 1}
    },
    delete_tempfiles=True,
    equation_file=f"LWFA1D.csv",
    turbo=True,
    model_selection="score",
)

starttime = time.time()
model.fit(inputs, target)
endtime = time.time()

best_idx = model.equations_.query(
    f"loss < {2 * model.equations_.loss.min()}"
).score.idxmax()

# Plotting the data
fig, ax1 = plt.subplots()
ax1.plot(inputs.iloc[:,0].values, inputs.iloc[:,1].values, label = "$E_1$" )
ax1.set_xlabel("x")
ax1.set_ylabel("E1")
ax2 = ax1.twinx()
ax2.plot(inputs.iloc[:,0].values, target.iloc[:,0].values,'r', label = "$|n|$" , alpha = 0.8)
ax2.set_ylabel("|$n$|")
ax2.set_ylim(0,2)
ax3 = ax1.twinx()
ax3.plot(inputs.iloc[:,0].values, model.predict(inputs.to_numpy(), index=best_idx),'g', label = "$|n|$ fit" , alpha = 0.8)
ax3.set_ylabel("|$n$|")
ax3.set_ylim(0,2)
plt.title("Longitudinal Electric Field and Plasma Density")
plt.grid(True)
fig.legend(loc = (0.75,0.70))
fig.tight_layout()
plt.savefig("LWFA1Dfit.png")
plt.close()
plt.clf()

analysis = pd.read_csv("LWFA1D.csv")
for i in range(len(analysis)):
    analysis.loc[i,'Equation'] = "$"+model.latex(i)+"$"
mdFile.new_header(level=1, title='24h Analysis')
mdFile.new_paragraph(f"Here we have the results of the 24h analysis of the Laser Wakefield Accelerator data made from ZPIC. The analysis was run on a 4 core machine, with 8GB of RAM. The analysis was run for 24h, and the best result was found in {endtime-starttime} seconds. The results are on the following table:")
mdFile.new_line()
mdFile.new_table(columns=len(analysis.columns), rows=len(analysis)+1, text=np.concatenate((analysis.columns.to_numpy().ravel(),analysis.to_numpy().ravel())), text_align='center')

mdFile.create_md_file()