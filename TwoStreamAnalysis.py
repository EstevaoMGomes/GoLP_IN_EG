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

# Reading TwoStream data - 100 lines
inputs100 = pd.read_csv("/home/estevao/GoLP/datasets/inputs.csv", header=None, skiprows=[0], nrows=100)
inputs100.columns = ["F","E_f","v","x","F_x","F_xx","E_x","E_xx","F_u","F_uu"]
targets100 = pd.read_csv("/home/estevao/GoLP/datasets/target.csv", header=None, nrows=100)

# Reading TwoStream data - 10 lines
inputs10 = pd.read_csv("/home/estevao/GoLP/datasets/inputs.csv", header=None, skiprows=[0], nrows=10)
inputs10.columns = ["F","E_f","v","x","F_x","F_xx","E_x","E_xx","F_u","F_uu"]
targets10 = pd.read_csv("/home/estevao/GoLP/datasets/target.csv", header=None, nrows=10)

# Setting the variation parameters
seed = np.array([1,2,3,4,5,6,7,8,9,10])
n = np.array([10,100,1000])
mult = np.array([1,2,3,4])


mdFile = MdUtils(file_name='data/real/twostream/Analysis', title='TwoStream Data Analysis')

try:
    os.mkdir("data/real/twostream/seed")
except:
    pass

mdFile.new_header(level=1, title='Seed Analysis')

tableseeds= ["Seed", "Complexity", "Loss", "Score" ,"Best Fit", "Time"]

for seedval in seed:
    starttime = time.time()
    np.random.seed(seedval)

    model = pysr.PySRRegressor(
        niterations=50,
        populations=12,
        binary_operators=["+", "*","-", "/"],
        delete_tempfiles=True,
        equation_file=f"data/real/twostream/seed/{seedval}.csv",
        turbo=True,
        model_selection="score",
    )

    model.fit(inputs, targets)

    endtime = time.time()

    maxscore=0
    for i, score in enumerate(model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableseeds.extend([seedval,model.equations_.iloc[index,0], model.equations_.iloc[index,1], model.equations_.iloc[index,2], latex(model.equations_.iloc[index,4]), f"{endtime - starttime}"])
    plt.plot(seedval,endtime - starttime,'o',color='black')

mdFile.new_line()
mdFile.new_table(columns=6, rows=len(seed)+1, text=tableseeds, text_align='center')
plt.xlabel("Seed number")
plt.ylabel("Time (s)")
plt.title("Time needed to fit the model for different seed values")
plt.show()
plt.savefig("data/real/twostream/seed/time_seed.png")
plt.close()


try:
    os.mkdir("data/real/twostream/regression")
except:
    pass

mdFile.new_header(level=1, title='Regression Analysis')

tableregression= ["50 Iterations", "12 Populations", "Best Fit", "Loss", "Time"]

for multval in mult:
    starttime = time.time()
    np.random.seed(1)
    
    model = pysr.PySRRegressor(
        niterations=50*multval,
        populations=12*multval,
        binary_operators=["+", "*","-", "/"],
        delete_tempfiles=True,
        equation_file=f"data/real/twostream/regression/x{multval}.csv",
    )

    model.fit(inputs, targets)

    endtime = time.time()
    tableregression.extend([f"x{multval}", f"x{multval}", f"${model.latex()}$", f"-", f"{endtime - starttime}"])
    plt.plot(multval,endtime - starttime,'o',color='black')

mdFile.new_line()
mdFile.new_table(columns=5, rows=len(mult)+1, text=tableregression, text_align='center')
plt.xlabel("Multiplication number")
plt.ylabel("Time (s)")
plt.title("Time needed to fit the model for different multiplication values\nof 50 iterations and 12 populations")
plt.show()
plt.savefig("data/real/twostream/regression/time_mult.png")
plt.close()


try:
    os.mkdir("data/real/twostream/datapoints")
except:
    pass

mdFile.new_header(level=1, title='Number of Data Points Analysis')

tabledatapoints= ["Number of Data Points", "Best Fit", "Loss", "Time"]
     
for ndp in n:
    starttime = time.time()
    np.random.seed(1)
    
    model = pysr.PySRRegressor(
        niterations=50,
        populations=12,
        binary_operators=["+", "*","-", "/"],
        delete_tempfiles=True,
        equation_file=f"data/real/twostream/datapoints/{ndp}.csv",    
    )
    
    if ndp == 10:
        model.fit(inputs10, targets10)
    elif ndp == 100:
        model.fit(inputs100, targets100)
    else:
        model.fit(inputs, targets)
    
    endtime = time.time()
    tabledatapoints.extend([f"{ndp}", f"${model.latex()}$", f"-", f"{endtime - starttime}"])
    plt.plot(ndp,endtime - starttime,'o',color='black')

mdFile.new_line()
mdFile.new_table(columns=4, rows=len(n)+1, text=tabledatapoints, text_align='center')
plt.xscale('log')
plt.xlabel("Number of data points")
plt.ylabel("Time (s)")
plt.title("Time needed to fit the model for different number of data points")
plt.show()
plt.savefig("data/real/twostream/datapoints/time_ndp.png")
plt.close()


try:
    os.mkdir("data/real/twostream/seedinvariant")
except:
    pass

mdFile.new_header(level=1, title='Seed Invariance Analysis')

for multval in mult:
    try:
        os.mkdir(f"data/real/twostream/seedinvariant/x{multval}")
    except:
        pass
    mdFile.new_header(level=2, title=f"populations = {12*multval}, iterations = {50*multval}")
    tableseeds= ["Seed", "Best Fit", "Complexity", "Loss", "Time"]
    for seedval in seed:
        starttime = time.time()
        np.random.seed(seedval)

        model = pysr.PySRRegressor(
            niterations=50*multval,
            populations=12*multval,
            binary_operators=["+", "*","-", "/"],
            delete_tempfiles=True,
            equation_file=f"data/real/twostream/seedinvariant/x{multval}/{seedval}.csv",
        )

        model.fit(inputs, targets)

        endtime = time.time()
        tableseeds.extend([f"{seedval}", f"${model.latex()}$", f"-", f"-", f"{endtime - starttime}"])
        plt.plot(seedval,endtime - starttime,'o',color='black')

    mdFile.new_line()
    mdFile.new_table(columns=5, rows=len(seed)+1, text=tableseeds, text_align='center')
    plt.xlabel("Seed number")
    plt.ylabel("Time (s)")
    plt.title(f"Time needed to fit the model for different seed values\npopulations = {12*multval}, iterations = {50*multval}")
    plt.show()
    plt.savefig(f"data/real/twostream/seedinvariant/x{multval}/time_seed_x{multval}.png")
    plt.close()



mdFile.new_table_of_contents(table_title='Contents', depth=2)
mdFile.create_md_file()