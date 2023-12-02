
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
|1|$e^{\left(1.23 - x_{0}\right) \left(5.55 x_{0} - 4.30\right)}$|-|88.84700584411621|
|2|$\frac{2.69 x_{0}}{- x_{0} + 0.372 + \frac{e^{x_{0}^{3}} - 0.0886}{x_{0}^{2}}}$|-|79.3480908870697|
|3|$e^{5.61 \left(1.23 - x_{0}\right) \left(x_{0} - 0.774\right)}$|-|85.29582595825195|
|4|$x_{0}^{2} e^{- \left(x_{0} - 1.12\right) \left(x_{0} \left(x_{0} + 2.65\right) - 1.17\right)}$|-|86.13774394989014|
|5|$x_{0}^{2} e^{- \frac{x_{0}}{0.0639 + \frac{0.379}{0.894 x_{0}^{2} - x_{0}}}}$|-|75.06784462928772|
|6|$e^{\left(1.23 - x_{0}\right) \left(5.51 x_{0} - 4.26\right)}$|-|78.20820736885071|
|7|$x_{0}^{3} e^{- \left(x_{0} - 1.08\right) \left(x_{0} + e^{x_{0}} - 0.444\right)}$|-|79.83965635299683|
|8|$e^{2.06 \left(1.21 - 0.986 x_{0}\right) \left(2.75 x_{0} - 2.12\right)}$|-|84.11719369888306|
|9|$x_{0}^{2} e^{x_{0}^{2} \left(1.12 - x_{0}\right) \left(2.73 - 0.322 x_{0}\right)}$|-|77.0909194946289|
|10|$e^{5.50 \left(0.774 - x_{0}\right) \left(x_{0} - 1.23\right)}$|-|77.13196039199829|

# Error Analysis
  

|Error|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: |
|0.0|$e^{5.56 \left(1.23 - x_{0}\right) \left(x_{0} - 0.773\right)}$|-|77.49297881126404|
|0.01|$e^{\left(1.23 - x_{0}\right) \left(5.56 x_{0} - 4.30\right)}$|-|78.55957293510437|
|0.1|$e^{11.4 \left(x_{0} e^{- x_{0}} - 0.359\right) e^{x_{0}}}$|-|96.48492169380188|
|1.0|$x_{0} \left(1.89 - x_{0}\right)$|-|74.03535962104797|

# Regression Analysis
  

|20 Iterations|50 Populations|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: | :---: |
|x1|x1|$e^{\left(6.81 - 5.55 x_{0}\right) \left(x_{0} - 0.774\right)}$|-|82.55933904647827|
|x2|x2|$e^{5.56 \left(0.774 - x_{0}\right) \left(x_{0} - 1.23\right)}$|-|311.8865373134613|
|x3|x3|$e^{5.56 \left(1.23 - x_{0}\right) \left(x_{0} - 0.774\right)}$|-|693.7757701873779|
|x4|x4|$e^{\left(1.23 - x_{0}\right) \left(5.55 x_{0} - 4.30\right)}$|-|1205.6800231933594|

# Number of Data Points Analysis
  

|Number of Data Points|Best Fit|Loss|Time|
| :---: | :---: | :---: | :---: |
|100|$e^{\left(1.23 - x_{0}\right) \left(5.56 x_{0} - 4.30\right)}$|-|30.106420516967773|
|1000|$e^{\left(2.53 - 2.06 x_{0}\right) \left(2.69 x_{0} - 2.08\right)}$|-|93.54706931114197|
|10000|$x_{0}^{3} e^{\left(1.08 - x_{0}\right) \left(x_{0} + e^{x_{0}} - 0.419\right)}$|-|721.9279661178589|
