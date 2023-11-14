import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pysr
from sklearn.model_selection import train_test_split
import os
import time
plt.rcParams['text.usetex'] = True

inputs = pd.read_csv("/home/estevao/GoLP/datasets/inputs.csv", header=None, skiprows=[0])
inputs.columns = ["F","Ef","v","x","F_x","F_xx","E_x","E_xx","F_u","F_uu"]

targets = pd.read_csv("/home/estevao/GoLP/datasets/target.csv", header=None)

start_time = time.time()
model = pysr.PySRRegressor(
    niterations=20,
    populations=50,
    binary_operators=["+", "*","-", "/"],
    #unary_operators=["exp", "cos", "sin"],
    equation_file="/home/estevao/GoLP/data/real/twostream.csv",
    )

model.fit(inputs, targets)
end_time = time.time()
print(f"Time: {end_time-start_time}")
for i in range(10):
    print(f"Complexity {i*2+1}: {model.sympy(i)}")

print(f"Best fit: {model.sympy()}")

