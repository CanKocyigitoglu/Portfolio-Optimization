# Portfolio Optimisation Using CAPM and Quadratic Programming
This project applies the Capital Asset Pricing Model (CAPM) and quadratic programming to optimise a stock portfolio based on expected returns and risk. The analysis was completed as part of a Machine Learning for Finance coursework.

# Project Overview
The project includes collecting 5 years of historical stock data for more than 10 assets using yFinance. It calculates daily returns and applies linear regression based on the CAPM formula to estimate alpha and beta values. These values help form the covariance matrix by combining systematic and idiosyncratic risks. Using CVXOPT, the notebook solves a quadratic programming problem to find the best portfolio weights for different return targets. It also visualises the efficient frontier and analyses how portfolio weights shift with different risk-free rates and levels of expected return.

# Technologies and Libraries Used
This project is implemented in Python and uses libraries such as NumPy, Pandas, Matplotlib, yFinance, Statsmodels, and CVXOPT.

# How to Run
1) Clone the repository

2) Install the required libraries (pip install -r requirements.txt)

3) Run the Jupyter Notebook Portfolio-Optimisation.ipynb step by step
