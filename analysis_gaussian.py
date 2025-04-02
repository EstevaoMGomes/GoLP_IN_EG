import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pysr
from sklearn.model_selection import train_test_split
import os
from time import time
from mdutils.mdutils import MdUtils
from sympy import latex

plt.rcParams['text.usetex'] = True

# Set the seed, number of data points, error, and regression multiplication factors for the analysis
seed = np.array([1,2,3,4])
n = np.array([10, 100, 1000, 10000])
error = np.array([0,0.01,0.1,1])
mult = np.array([0.5,1,2,4])

# Set the parameters for the Gaussian function
def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
mu = 0.5
sigma = 0.1

# Create a folder for results
folder = "results/gaussian/"
path = folder + "seed/"
os.makedirs(path, exist_ok=True)

# Create a new Markdown file
mdFile = MdUtils(file_name= folder + 'analysis', title='Gaussian Data Analysis')
mdFile.new_header(level=1, title='Seed Analysis')
tableseeds= ["Turbo", "Denoise", "Seed", "Complexity", "Loss", "Score" ,"Best Fit", "Time"]

# Creating data
np.random.seed(42) # Set the random seed for data reproducibility
data = np.linspace(0, 1, 1000)

# Creating models
normal_model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    output_directory = path + "normal/",
    model_selection="best",)

turbo_model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    output_directory = path + "turbo/",
    model_selection="best",
    turbo=True,)

denoise_model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    output_directory = path + "denoise/",
    model_selection="best",
    denoise=True,)

plt.figure(figsize=(7, 4))

for seed_idx, seed_val in enumerate(seed):
    expected = gauss(data, mu, sigma) + np.random.uniform(-0.1, 0.1, size=1000)
    
    # Set the random seed for PySR
    np.random.seed(seed_val)

    # Fit the model to the data before measuring time to compile first
    if seed_idx == 0: normal_model.fit(data[:, np.newaxis], expected)
    # Measure the time taken to fit the model
    start = time()
    normal_model.fit(data[:, np.newaxis], expected)
    end = time()

    if seed_idx == 0:
        best_idx = normal_model.equations_.query(
            f"loss < {2 * normal_model.equations_.loss.min()}"
        ).score.idxmax()

        # Plotting the data
        plt.figure(figsize=(7, 4))
        plt.plot(data, expected, label = "$Data$" )
        plt.plot(data, normal_model.predict(data[:, np.newaxis], index=best_idx), 'r', label = "Fit" , alpha = 0.8)
        plt.title("Gaussian Data Fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(path + f"seed_fit.pdf")
        plt.close()

    maxscore=0
    for i, score in enumerate(normal_model.equations_.score):
        if score > maxscore:
            maxscore = score
            index = i

    # Append the results to markdown table
    tableseeds.extend(["False", "False", seed_val, normal_model.equations_.iloc[index,0], normal_model.equations_.iloc[index,1], normal_model.equations_.iloc[index,2], f"${latex(normal_model.equations_.iloc[index,4])}$", f"{end - start}"])
    
    # Plotting the data
    if seed_idx == 0:
        plt.bar(seed_val - 0.2 , end - start, width=0.2, color='black', label="Normal", edgecolor='black')
    plt.bar(seed_val - 0.2 , end - start, width=0.2, color='black', edgecolor='black')

    # Fit the model to the data before measuring time to compile first
    if seed_idx == 0: turbo_model.fit(data[:, np.newaxis], expected)
    # Measure the time taken to fit the model
    start = time()
    turbo_model.fit(data[:, np.newaxis], expected)
    end = time()

    if seed_idx == 0:
        best_idx = turbo_model.equations_.query(
            f"loss < {2 * turbo_model.equations_.loss.min()}"
        ).score.idxmax()

        # Plotting the data
        plt.figure(figsize=(7, 4))
        plt.plot(data, expected, label = "$Data$" )
        plt.plot(data, turbo_model.predict(data[:, np.newaxis], index=best_idx), 'r', label = "Fit" , alpha = 0.8)
        plt.title("Gaussian Data Fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(path + f"seed_turbo_fit.pdf")
        plt.close()

    maxscore=0
    for i, score in enumerate(turbo_model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i
    
    # Append the results to markdown table
    tableseeds.extend(["True", "False", seed_val, turbo_model.equations_.iloc[index,0], turbo_model.equations_.iloc[index,1], turbo_model.equations_.iloc[index,2], f"${latex(turbo_model.equations_.iloc[index,4])}$", f"{end - start}"])
    
    if seed_idx == 0:
        plt.bar(seed_val , end - start, width=0.2, color='red', label="Turbo", edgecolor='black')
    plt.bar(seed_val , end - start, width=0.2, color='red', edgecolor='black')

    # Measure the time taken to fit the model
    start = time()
    denoise_model.fit(data[:, np.newaxis], expected)
    end = time()

    if seed_idx == 0:
        best_idx = denoise_model.equations_.query(
            f"loss < {2 * denoise_model.equations_.loss.min()}"
        ).score.idxmax()

        # Plotting the data
        plt.figure(figsize=(7, 4))
        plt.plot(data, expected, label = "$Data$" )
        plt.plot(data, denoise_model.predict(data[:, np.newaxis], index=best_idx), 'r', label = "Fit" , alpha = 0.8)
        plt.title("Gaussian Data Fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(path + f"seed_denoise_fit.pdf")
        plt.close()
    
    maxscore=0
    for i, score in enumerate(denoise_model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableseeds.extend(["False", "True", seed_val, denoise_model.equations_.iloc[index,0], denoise_model.equations_.iloc[index,1], denoise_model.equations_.iloc[index,2], f"${latex(denoise_model.equations_.iloc[index,4])}$", f"{end - start}"])
    
    if seed_idx == 0:
        plt.bar(seed_val+0.2 , end - start, width=0.2, color='green', label="Denoise", edgecolor='black')
    plt.bar(seed_val+0.2 , end - start, width=0.2, color='green', edgecolor='black')

mdFile.new_line()
mdFile.new_paragraph("The standard parameter used are: 50 iterations, 12 populations, 1000 data points and error 0.01")
mdFile.new_table(columns=8, rows=len(seed)*3+1, text=tableseeds, text_align='center')

plt.xlabel("Seed number")
plt.ylabel("Time (s)")
plt.legend()
plt.xticks(seed, seed)
plt.title("Time needed to fit the model for different seed values")
plt.savefig(path + "seed_time.pdf")
plt.close()

###################################  Analysis of the error  ###################################

# Create a folder for error analysis
path = folder + "error/"
os.makedirs(path, exist_ok=True)

# Create a new paragraph in the Markdown file
mdFile.new_header(level=1, title='Error Analysis')
tableerrors= ["Error", "Complexity", "Loss", "Score", "Best Fit", "Time"]

# Set the random seed for PySR
np.random.seed(42)

# Creating model
turbo_model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    output_directory = path,
    model_selection="best",
    turbo=True,)

plt.figure(figsize=(7, 4))

for error_idx, error_val in enumerate(error):
    expected = gauss(data, mu, sigma) + np.random.uniform(-error_val, error_val, size=1000)

    start = time()
    turbo_model.fit(data[:, np.newaxis], expected)
    end = time()

    best_idx = turbo_model.equations_.query(
        f"loss < {2 * turbo_model.equations_.loss.min()}"
    ).score.idxmax()
    
    # Plotting the data
    plt.figure(figsize=(7, 4))
    plt.plot(data, expected, label = "$Data$")
    plt.plot(data, turbo_model.predict(data[:, np.newaxis], index=best_idx), 'r', label = "Fit" , alpha = 0.8)
    plt.title("Gaussian Data Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(path + f"error_{error_val}_fit.pdf")
    plt.close()

    maxscore=0
    for i, score in enumerate(turbo_model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableerrors.extend([error_val, turbo_model.equations_.iloc[index,0], turbo_model.equations_.iloc[index,1], turbo_model.equations_.iloc[index,2], f"${latex(turbo_model.equations_.iloc[index,4])}$", f"{end - start}"])
    plt.bar(error_idx, end - start, color='red', edgecolor='black')

mdFile.new_line()
mdFile.new_paragraph("The standard parameter used are: 50 iterations, 12 populations, 1000 data points, seed 1 and denoise True")
mdFile.new_table(columns=6, rows=len(error)+1, text=tableerrors, text_align='center')

plt.xlabel("Error")
plt.ylabel("Time (s)")
plt.xticks(range(len(error)), error)
plt.title("Time needed to fit the model for different error values")
plt.savefig(path + "error_time.pdf")
plt.close()

###################################  Analysis of the regression  ###################################

# Create a folder for error analysis
path = folder + "regression/"
os.makedirs(path, exist_ok=True)

# Create a new paragraph in the Markdown file
mdFile.new_header(level=1, title='Regression Analysis')
tableregression= ["50 Iterations, 12 Populations", "Complexity", "Loss", "Score", "Best Fit", "Time"]

# Set the random seed for PySR
np.random.seed(42)

plt.figure(figsize=(7, 4))

for mult_idx, mult_val in enumerate(mult):
    expected = gauss(data, mu, sigma) + np.random.uniform(-0.1, 0.1, size=1000)
    
    # Creating model
    turbo_model = pysr.PySRRegressor(
        niterations=50*mult_val,
        populations=12*mult_val,
        binary_operators=["+", "*","-", "/"],
        unary_operators=["exp"],
        output_directory = path,
        model_selection="best",
        turbo=True,)

    start = time()
    turbo_model.fit(data[:, np.newaxis], expected)
    end = time()

    best_idx = turbo_model.equations_.query(
        f"loss < {2 * turbo_model.equations_.loss.min()}"
    ).score.idxmax()
    
    # Plotting the data
    plt.figure(figsize=(7, 4))
    plt.plot(data, expected, label = "$Data$")
    plt.plot(data, turbo_model.predict(data[:, np.newaxis], index=best_idx), 'r', label = "Fit" , alpha = 0.8)
    plt.title("Gaussian Data Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(path + f"mult_{mult_val}_fit.pdf")
    plt.close()

    maxscore=0
    for i, score in enumerate(turbo_model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tableregression.extend([f"x{mult_val}", turbo_model.equations_.iloc[index,0], turbo_model.equations_.iloc[index,1], turbo_model.equations_.iloc[index,2], f"${latex(turbo_model.equations_.iloc[index,4])}$", f"{end - start}"])
    plt.bar(mult_idx, end - start, color='red', edgecolor='black')

mdFile.new_line()
mdFile.new_table(columns=6, rows=len(mult)+1, text=tableregression, text_align='center')
plt.xlabel("Multiplication number")
plt.ylabel("Time (s)")
plt.xticks(range(len(mult)), mult)
plt.title("Time needed to fit the model for different multiplication values\nof 50 iterations and 12 populations")
plt.savefig(path + "mult_time.pdf")
plt.close()

###################################  Analysis of the data points  ###################################

# Create a folder for error analysis
path = folder + "data_points/"
os.makedirs(path, exist_ok=True)

# Create a new paragraph in the Markdown file
mdFile.new_header(level=1, title='Number of Data Points Analysis')
tabledatapoints= ["Number of Data Points", "Complexity", "Loss", "Score", "Best Fit", "Time"]

# Set the random seed for PySR
np.random.seed(42)

# Creating model
turbo_model = pysr.PySRRegressor(
    niterations=50,
    populations=12,
    binary_operators=["+", "*","-", "/"],
    unary_operators=["exp"],
    output_directory = path,
    model_selection="best",
    turbo=True,)

plt.figure(figsize=(7, 4))

for ndp_idx, ndp in enumerate(n):
    data = np.linspace(0, 1, ndp)
    expected = gauss(data, mu, sigma) + np.random.uniform(-0.1, 0.1, size=ndp)
    
    start = time()
    turbo_model.fit(data[:, np.newaxis], expected)
    end = time()

    best_idx = turbo_model.equations_.query(
        f"loss < {2 * turbo_model.equations_.loss.min()}"
    ).score.idxmax()
    
    # Plotting the data
    plt.figure(figsize=(7, 4))
    plt.plot(data, expected, label = "$Data$")
    plt.plot(data, turbo_model.predict(data[:, np.newaxis], index=best_idx), 'r', label = "Fit" , alpha = 0.8)
    plt.title("Gaussian Data Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(path + f"ndp_{ndp}_fit.pdf")
    plt.close()

    maxscore=0
    for i, score in enumerate(turbo_model.equations_.score):
        if score > maxscore:
            maxscore= score
            index = i

    tabledatapoints.extend([ndp, turbo_model.equations_.iloc[index,0], turbo_model.equations_.iloc[index,1], turbo_model.equations_.iloc[index,2], f"${latex(turbo_model.equations_.iloc[index,4])}$", f"{end - start}"])
    plt.bar(ndp_idx, end - start, color='red', edgecolor='black')

mdFile.new_line()
mdFile.new_table(columns=6, rows=len(n)+1, text=tabledatapoints, text_align='center')

plt.xlabel("Number of data points")
plt.ylabel("Time (s)")
plt.xticks(range(len(n)), n)
plt.title("Time needed to fit the model for different number of data points")
plt.savefig(path + "ndp_time.pdf")
plt.close()


mdFile.new_table_of_contents(table_title='Contents', depth=2)
mdFile.create_md_file()