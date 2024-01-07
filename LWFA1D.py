# Laser Wakefield Accelerator

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

# Reading TwoStream data - All (1000) lines
inputs = pd.read_csv("/home/estevao/GoLP/datasets/inputs.csv", header=None, skiprows=[0])
inputs.columns = ["F","E_f","v","x","F_x","F_xx","E_x","E_xx","F_u","F_uu"]
inputsmean = inputs.mean()
normalizedinputs = inputs-inputsmean
targets = pd.read_csv("/home/estevao/GoLP/datasets/target.csv", header=None)