import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pysr
from sklearn.model_selection import train_test_split
import os
import time
from mdutils.mdutils import MdUtils
from sympy import latex

plt.rcParams['text.usetex'] = True


seed = np.array([1,2,3,4,5,6,7,8,9,10])
n = np.array([100,1000,10000])
error = np.array([0,0.01,0.1,1])
mult = np.array([0.5,1,2,4,10])

def gauss(x,x0,sigma):
    np.exp(- (x - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))

x0 = 1
sigma = 0.3

mdFile = MdUtils(file_name='data/artifitial/Analysis', title='Artifitial Data Analysis')

try:
    os.mkdir("data/artifitial/seed")
except:
    pass

mdFile.new_header(level=1, title='Seed Analysis')

tableseeds= ["Turbo","Denoise","Seed", "Complexity", "Loss", "Score" ,"Best Fit", "Time"]

for seedval in seed:
    np.random.seed(1)
    X = 2 * np.random.rand(1000,2)
    y=np.zeros(1000)
    for index in range(1000):
            y[index] = np.exp(-(X[index,0] - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))*(1 + np.random.uniform(-0.01,0.01))
            #y[index] = gauss(X[index,0],x0,sigma)*(1 + np.random.uniform(-0.01,0.01))
    np.random.seed(seedval)
    model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    equation_file=f"data/artifitial/seed/{seedval}.csv",
    model_selection="score",)

    starttime = time.time()
    model.fit(X, y)
    endtime = time.time()

    if seedval == 1:
        best_idx = model.equations_.query(
            f"loss < {2 * model.equations_.loss.min()}"
        ).score.idxmax()

        # Plotting the data
        fig, ax1 = plt.subplots()
        ax1.plot(X[:,0], y, label = "$Data$" )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2 = ax1.twinx()
        ax2.plot(X[:,0], model.predict(X, index=best_idx),'r', label = "$Fit$" , alpha = 0.8)
        ax2.set_ylabel("")
        ax2.set_ylim(0,2)
        plt.title("Artifitial Data Fit")
        plt.grid(True)
        fig.legend(loc = (0.75,0.70))
        fig.tight_layout()
        plt.savefig("ArtifitialFit.png")
        plt.close()
        plt.clf()
    
    print(X.iloc[:,0].values)

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableseeds.extend(["False", "False", seedval,model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], f"${latex(model.equations_.iloc[index,4])}$", f"{endtime - starttime}"])
    plt.plot(seedval,endtime - starttime,'o',color='black')

    np.random.seed(seedval)
    model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    equation_file=f"data/artifitial/seed/{seedval}turbo.csv",
    model_selection="score",
    turbo=True,)

    starttime = time.time()
    model.fit(X, y)
    endtime = time.time()

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i
    

    tableseeds.extend(["True", "False", seedval,model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], f"${latex(model.equations_.iloc[index,4])}$", f"{endtime - starttime}"])
    plt.plot(seedval,endtime - starttime,'o',color='red',label="Turbo")

    np.random.seed(seedval)
    model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    equation_file=f"data/artifitial/seed/{seedval}denoise.csv",
    model_selection="score",
    turbo=True,)

    starttime = time.time()
    model.fit(X, y)
    endtime = time.time()

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableseeds.extend(["False", "True", seedval,model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], f"${latex(model.equations_.iloc[index,4])}$", f"{endtime - starttime}"])
    plt.plot(seedval,endtime - starttime,'o',color='green',label="Denoise")

mdFile.new_line()
mdFile.new_paragraph("The standard parameter used are: 50 iterations, 12 populations, 1000 data points and error 0.01")
mdFile.new_table(columns=8, rows=(len(seed)+1)*3, text=tableseeds, text_align='center')
plt.xlabel("Seed number")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Time needed to fit the model for different seed values")
plt.show()
plt.savefig("data/artifitial/seed/time_seed.png")
plt.close()


try:
    os.mkdir("data/artifitial/error")
except:
    pass

mdFile.new_header(level=1, title='Error Analysis')

tableerrors= ["Error", "Complexity", "Loss", "Score" ,"Best Fit", "Time"]

for errorval in error:
    starttime = time.time()
    np.random.seed(1)
    X = 2 * np.random.rand(1000,2)
    y=np.zeros(1000)
    for index in range(1000):
            y[index] = np.exp(-(X[index,0] - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))*(1 + np.random.uniform(-errorval,errorval))
            #y[index] = gauss(X[index,0],x0,sigma)*(1 + np.random.uniform(-errorval,errorval))
    
    model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    equation_file=f"data/artifitial/error/{errorval}.csv",
    model_selection="score",
    denoise=True,)

    model.fit(X, y)

    endtime = time.time()

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableerrors.extend([errorval,model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], f"${latex(model.equations_.iloc[index,4])}$", f"{endtime - starttime}"])
    plt.plot(errorval,endtime - starttime,'o',color='black')

mdFile.new_line()
mdFile.new_paragraph("The standard parameter used are: 50 iterations, 12 populations, 1000 data points, seed 1 and denoise True")
mdFile.new_table(columns=6, rows=len(error)+1, text=tableerrors, text_align='center')

plt.xscale('log')
plt.xlabel("Error")
plt.ylabel("Time (s)")
plt.title("Time needed to fit the model for different error values")
plt.show()
plt.savefig("data/artifitial/error/time_error.png")
plt.close()


try:
    os.mkdir("data/artifitial/regression")
except:
    pass

mdFile.new_header(level=1, title='Regression Analysis')

tableregression= ["50 Iterations, 12 Populations", "Complexity", "Loss", "Score", "Best Fit", "Time"]


for multval in mult:
    starttime = time.time()
    np.random.seed(1)
    X = 2 * np.random.rand(1000,2)
    y=np.zeros(1000)
    for index in range(1000):
            y[index] = np.exp(-(X[index,0] - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))*(1 + np.random.uniform(-0.01,0.01))
            #y[index] = gauss(X[index,0],x0,sigma)*(1 + np.random.uniform(-0.01,0.01))
    model = pysr.PySRRegressor(
    niterations=50*multval,
    populations=12*multval,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    equation_file=f"data/artifitial/regression/x{multval}.csv",
    model_selection="score",
    turbo=True,)

    model.fit(X, y)

    endtime = time.time()

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableregression.extend([f"x{multval}",model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], f"${latex(model.equations_.iloc[index,4])}$", f"{endtime - starttime}"])
    plt.plot(multval,endtime - starttime,'o',color='black')

mdFile.new_line()
mdFile.new_table(columns=6, rows=len(mult)+1, text=tableregression, text_align='center')
plt.xlabel("Multiplication number")
plt.ylabel("Time (s)")
plt.title("Time needed to fit the model for different multiplication values\nof 50 iterations and 12 populations")
plt.show()
plt.savefig("data/artifitial/regression/time_mult.png")
plt.close()


try:
    os.mkdir("data/artifitial/datapoints")
except:
    pass

mdFile.new_header(level=1, title='Number of Data Points Analysis')

tabledatapoints= ["Number of Data Points", "Complexity", "Loss", "Score", "Best Fit", "Time"]
     
for ndp in n:
    starttime = time.time()
    np.random.seed(1)
    X = 2 * np.random.rand(ndp,2)
    y=np.zeros(ndp)
    for index in range(ndp):
            y[index] = np.exp(-(X[index,0] - x0) ** 2 / (2 * sigma ** 2))/(sigma*np.sqrt(2*np.pi))*(1 + np.random.uniform(-0.01,0.01))
            #y[index] = gauss(X[index,0],x0,sigma)*(1 + np.random.uniform(-0.01,0.01))
    model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    equation_file=f"data/artifitial/datapoints/{ndp}.csv",
    model_selection="score",
    turbo=True,)
    
    model.fit(X, y)
    
    endtime = time.time()

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tabledatapoints.extend([ndp,model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], f"${latex(model.equations_.iloc[index,4])}$", f"{endtime - starttime}"])
    plt.plot(ndp,endtime - starttime,'o',color='black')

mdFile.new_line()
mdFile.new_table(columns=6, rows=len(n)+1, text=tabledatapoints, text_align='center')
plt.xscale('log')
plt.xlabel("Number of data points")
plt.ylabel("Time (s)")
plt.title("Time needed to fit the model for different number of data points")
plt.show()
plt.savefig("data/artifitial/datapoints/time_ndp.png")
plt.close()


mdFile.new_table_of_contents(table_title='Contents', depth=2)
mdFile.create_md_file()