import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
plt.rcParams['text.usetex'] = True


seed = np.array([1,2,3,4])
n = np.array([100,1000,10000,100000])
np.random.seed(seed[0])

def gaussian2D(A,a,b,c,x,y):
    return A*np.exp(-((x-b)**2+(y-c)**2)/a)

i=1
X = 4 * np.random.rand(n[i],2) - 1
y = gaussian2D(1.7,1,1,1,X[:,0],X[:,1])

ax = plt.axes(projection='3d')
ax.scatter3D(X[:, 0],X[:, 1], y, alpha=0.2)
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
ax.set_zlabel("$y$")
    
plt.savefig("data3D.png")
plt.clf()

plt.scatter(X[:, 0], y, alpha=0.2)
plt.xlabel("$x_0$")
plt.ylabel("$y$")
plt.savefig("data2D_1.png")
plt.clf()

plt.scatter(X[:, 1], y, alpha=0.2)
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.savefig("data2D_2.png")