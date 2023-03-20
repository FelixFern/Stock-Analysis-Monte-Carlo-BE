# Import modules
# Extract stock market data
from func import fetch_data, check_columns, check_values, expand_data, final_data, get_return, \
    simulation, generate_decision_sequence, trading_algo, trading_sim, validate_decision, stock_var
import yfinance as yf

# Importing and transforming file
import pandas as pd

# Data manipulation
import numpy as np
import datetime as dt  # Datetime manipulation

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# ------------------------------------------------------------------------------------------------- #
# Main Features

# Start and End Date
start_date = '2021-01-01'
end_date = dt.datetime.now().strftime('%Y-%m-%d')

# Stock
stock = fetch_data(stock='ANTM.JK',
                   start_date=start_date,
                   end_date=end_date)

stock = final_data(data=stock,
                   start_date=start_date,
                   end_date=end_date,
                   criteria='Adj Close')

# Overview
stock = get_return(data=stock, 
                   method='log')

# Simulation
n_sim = 1e4
days = 30
conf_level = 0.95

# Simulation
price = simulation(data=stock,
                   days=days,
                   n_sim=n_sim)

# Trading
money = 1e8
decision = trading_algo(mrx=price)
optimal_decision_sequence = generate_decision_sequence(data=decision, 
                                                       days=days,
                                                       n_sim=n_sim,
                                                       threshold=0.5)
final_sim = trading_sim(price=price,
                        decision=optimal_decision_sequence,
                        money=money)
var_value = stock_var(final_sim, conf_level)

print(final_sim)
print(f'Risiko maksimum pada taraf kepercayaan {conf_level}: {var_value}')