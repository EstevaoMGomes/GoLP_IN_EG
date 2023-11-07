import sympy
import numpy as np
from matplotlib import pyplot as plt
import pysr
from sklearn.model_selection import train_test_split
import os
plt.rcParams['text.usetex'] = True


seed = np.array([1,2,3,4])
n = np.array([100,1000,10000,100000])
error = np.array([0,0.01,0.1,1])
times = np.array([1,2,4])
np.random.seed(seed[0])

def gauss(x,x0,sigma):
    np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))

x0 = 1
sigma = 0.3
for j in range(4):
    np.random.seed(seed[j])
    for i in range(4):
        X = 2 * np.random.rand(2,n[i])
        y=np.zeros(n[i])
        for k in range(4):
            for index in range(n[i]):
                    y[index] = np.exp(-(X[0,index] - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))*(1 + np.random.uniform(-error[k],error[k]))
                    #y[index] = gauss(X[0,index],x0,sigma)*(1 + np.random.uniform(-error[3],error[3]))
            for l in range(3):
                model = pysr.PySRRegressor(
                niterations=20*times[l],
                populations=50*times[l],
                binary_operators=["+", "*","-", "/"],
                unary_operators=["exp"],
                #temp_equation_file="hall_of_fame.csv",
                delete_tempfiles=True,
                equation_file=f"data/{n[i]}dp/seed{seed[j]}/error{error[k]*100}%/iterations_x{times[l]}/data.csv",)               
                """
                try:
                    os.mkdir(f"data/{n[i]}dp")
                except:
                    pass
                try:
                    os.mkdir(f"data/{n[i]}dp/seed{seed[j]}")
                except:
                    pass
                try:
                    os.mkdir(f"data/{n[i]}dp/seed{seed[j]}/error{error[k]*100}%")
                except:
                    pass
                try:
                    os.mkdir(f"data/{n[i]}dp/seed{seed[j]}/error{error[k]*100}%/iterations_x{times[l]}")
                except:
                    pass
                try:
                    open(f"data/{n[i]}dp/seed{seed[j]}/error{error[k]*100}%/iterations_x{times[l]}/data.csv","x")
                except:
                    pass
                """
                model.fit(X, y)

                

