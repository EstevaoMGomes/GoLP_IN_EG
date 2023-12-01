
Artifitial Data Analysis
========================

Contents
========

* [Seed Analysis](#seed-analysis)
* [Error Analysis](#error-analysis)
* [Regression Analysis](#regression-analysis)
* [Number of Data Points Analysis](#number-of-data-points-analysis)

# Seed Analysis
  

|Seed|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: |
|1|$0.384 e^{\left(8.13 - 5.51 x_{0}\right) \left(x_{0} - 0.526\right)}$|-|90.38273239135742|
|2|$e^{3.61 \left(1.88 - 1.54 x_{0}\right) \left(x_{0} - 0.774\right)}$|-|77.76671004295349|
|3|$x_{0}^{2} e^{\left(x_{0} - 1.11\right) \left(- e^{x_{0}} - 0.570 + 2.16 e^{- x_{0}^{2}}\right)}$|-|74.25499248504639|
|4|$\frac{2.76}{-0.666 + \frac{e^{x_{0}^{3}}}{x_{0}^{3}}}$|-|93.29622793197632|

# Error Analysis
  

|Error|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: |
|0.0|$e^{5.56 \left(1.23 - x_{0}\right) \left(x_{0} - 0.774\right)}$|-|79.29691767692566|
|0.01|$x_{0}^{2} e^{\left(1.12 - x_{0}\right) \left(x_{0} \left(x_{0} + 0.584\right) + 2 x_{0} - 1.12\right)}$|-|78.03501510620117|
|0.1|$x_{0}^{2} e^{- 2.34 x_{0}^{2} \left(x_{0} - 1.12\right)}$|-|77.0858371257782|
|1.0|$x_{0} \left(1.83 - x_{0}\right)$|-|79.8245460987091|

# Regression Analysis
  

|20 Iterations|50 Populations|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: | :---: |
|x1|x1|$x_{0}^{3} e^{- \left(x_{0} - 1.08\right) \left(x_{0} + e^{x_{0}} - 0.428\right)}$|-|79.01841616630554|
|x2|x2|$e^{\left(0.774 - x_{0}\right) \left(5.55 x_{0} - 6.81\right)}$|-|327.9216182231903|
|x4|x4|$e^{5.55 \left(0.774 - x_{0}\right) \left(x_{0} - 1.23\right)}$|-|1243.5722739696503|

# Number of Data Points Analysis
  

|Number of Data Points|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: |
|100|$e^{\left(1.23 - x_{0}\right) \left(5.56 x_{0} - 4.30\right)}$|-|ordem de 10|
|1000|$e^{\left(0.591 - 0.482 x_{0}\right) \left(11.5 x_{0} - 8.90\right)}$|-|ordem de 100|
|10000|$e^{\left(6.81 - 5.56 x_{0}\right) \left(x_{0} - 0.774\right)}$|-|ordem de 1000|
|100000|$x_{0} e^{\left(1.15 - x_{0}\right) \left(4.91 x_{0} - 3.14\right)}$|-|cerca de 6000|
