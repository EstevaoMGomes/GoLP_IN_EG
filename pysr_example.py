import pysr
import numpy as np

np.random.seed(0)
n = 3000
X = 6 * np.random.rand(n,2)-3
H = 113 * np.exp(-2 * (X[:, 0]-1) * (X[:, 0]-1))
y = np.random.poisson(H, n)

model = pysr.PySRRegressor(
    niterations=5,
    populations=10,
    binary_operators=["plus", "mult"],
    unary_operators=["exp"],
)

print('Start Fit')

model.fit(X, y)
model.sympy()

print('Finished')
