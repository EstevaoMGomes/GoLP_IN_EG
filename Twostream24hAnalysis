import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pysr
from sklearn.model_selection import train_test_split
import os
import time
from mdutils.mdutils import MdUtils
from sympy import latex
plt.rcParams['text.usetex'] = True

# Reading TwoStream data - All (1000) lines
inputs = pd.read_csv("/home/estevao/GoLP/datasets/inputs.csv", header=None, skiprows=[0])
inputs.columns = ["F","E_f","v","x","F_x","F_xx","E_x","E_xx","F_u","F_uu"]
targets = pd.read_csv("/home/estevao/GoLP/datasets/target.csv", header=None)

try:
    os.mkdir("data/real/twostream/24hAnalysis")
except:
    pass

mdFile = MdUtils(file_name='data/real/twostream/24hAnalysis', title='TwoStream Data Analysis')

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
    delete_tempfiles=True,
    equation_file=f"data/real/twostream/24hAnalysis.csv",
    turbo=True,
    model_selection="score",
)

starttime = time.time()
model.fit(inputs, targets)
endtime = time.time()

analysis = pd.read_csv("data/real/twostream/24hAnalysis.csv")
mdFile.new_header(level=1, title='24h Analysis')
mdFile.new_paragraph(f"Here we have the results of the 24h analysis of the TwoStream data. The analysis was run on a 4 core machine, with 8GB of RAM. The analysis was run for 24h, and the best result was found in {endtime-starttime} seconds. The results are on the following table:")
mdFile.new_line()
mdFile.new_table(columns=6, rows=1, text=analysis, text_align='center')
