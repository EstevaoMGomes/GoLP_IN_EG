import pysr
import numpy as np
import pandas as pd
from sympy import latex
import time

np.random.seed(0)
n = 3000
X = 6 * np.random.rand(n,2)-3
H = 113 * np.exp(-2 * (X[:, 0]-1) * (X[:, 0]-1))
y = np.random.poisson(H, n)

model = pysr.PySRRegressor(
    niterations=10,
    populations=12,
    binary_operators=["plus", "mult"],
    unary_operators=["exp"],
    equation_file="./ficheiro.csv",
    model_selection="score",
    turbo=True,
)

print('Start Fit')

start_time = time.time()
model.fit(X, y)
end_time = time.time()
model.sympy()

print(f"Best fit: {model.sympy()}")
model.equations_
#print(pd.read_csv("ficheiro.csv"))
print(f"Time: {end_time - start_time}")
print(model)
print(model.equations_.iloc[:,0])
print(model.equations_.iloc[:,1])
print(model.equations_.iloc[:,2])
print(model.equations_.iloc[:,3])
print(model.equations_.iloc[:,4])
print(model.equations_.iloc[:,5])

maxscore=0
for i, score in enumerate(model.equations_.score):
    if score > maxscore:
        maxscore= score
        index = i
print(index)
print(maxscore)
print(f"Best fit:\n Complexity: {model.equations_.iloc[index, 0]}\n Loss: {model.equations_.iloc[index, 1]}\n Score: {model.equations_.iloc[index,2]}\n Equation: {model.equations_.iloc[index,3]}\n Simpy: {model.equations_.iloc[index,4]}\n Lambda: {model.equations_.iloc[index,5]}\n")
print(latex(model.equations_.iloc[index,4]))
