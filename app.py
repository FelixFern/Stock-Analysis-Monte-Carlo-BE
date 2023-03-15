from fastapi import FastAPI
from pydantic import BaseModel

# Extract stock market data
from func import fetch_data, check_columns, check_values, expand_data, final_data, get_return, \
    simulation, generate_decision_sequence, trading_algo, trading_sim, validate_decision, stock_var
import yfinance as yf

# Importing and transforming file
import pandas as pd

# Data manipulation
import numpy as np
import datetime as dt  # Datetime manipulation

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class GetDataPayload(BaseModel):
    stock: str
    criteria: str
    start_date: str


class SimulatePayload(BaseModel):
    stock: str
    criteria: str
    threshold = float
    start_date: str
    n_sim: int
    days: int


class TradePayload(BaseModel):
    stock: str
    criteria: str
    start_date: str
    n_sim: int
    days: int
    optimal_trading_sequence: list
    money: int
    conf_level: float


@app.get('/get')
async def getData(req: GetDataPayload):
    stock_name = req.stock
    criteria = req.criteria
    start_date = req.start_date

    if start_date == '':
        start_date = '2021-01-01'

    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    stock = fetch_data(stock=stock_name,
                       start_date=start_date,
                       end_date=end_date)

    stock = final_data(data=stock,
                       start_date=start_date,
                       end_date=end_date,
                       criteria=criteria)

    stock = stock.reset_index()

    dct = {'date': list(stock.iloc[:, 0]),
           'values': list(stock.iloc[:, 1])}

    return dct


@app.post('/simulate')
def simulateStock(req: SimulatePayload):
    stock_name = req.stock
    criteria = req.criteria
    start_date = req.start_date
    threshold = req.threshold
    n_sim = req.n_sim
    days = req.days

    if start_date == '':
        start_date = '2021-01-01'

    # if threshold == '':
    #     threshold = float(0.5)

    end_date = dt.datetime.now().strftime('%Y-%m-%d')

    stock = fetch_data(stock=stock_name,
                       start_date=start_date,
                       end_date=end_date)

    stock = final_data(data=stock,
                       start_date=start_date,
                       end_date=end_date,
                       criteria=criteria)

    stock = get_return(stock)

    # print(stock)
    price = simulation(data=stock,
                       days=days,
                       n_sim=n_sim)
    decisions = trading_algo(mrx=price)
    decision_sequence = generate_decision_sequence(decisions, n_sim, days)

    optimal_trading_sequence = []
    data = []

    stock = stock.reset_index()
    stock["Date"] = stock["Date"].astype(str)
    for day in range(days):
        decision = decision_sequence['DECISION'].iloc[day]

        if decision == 0:
            conf = decision_sequence['HOLD_CONF'].iloc[day]

        elif decision == 1:
            conf = decision_sequence['SELL_CONF'].iloc[day]

        else:
            conf = decision_sequence['BUY_CONF'].iloc[day]

        optimal_trading_sequence.append({'conf': conf,
                                         'decision': decision})


    dates = pd.date_range(dt.datetime.today(), periods = days).to_pydatetime().tolist()
    for i in range(len(dates)): 
        dates[i] = dates[i].strftime("%Y-%m-%d")

    for sim in range(n_sim):
        data.append({'simulation': sim + 1,
                     'data': {'date': list(dates),
                              'price': list(price[sim])},
                     'trading_sequence': list(decisions[sim])})
        
    dct = {'optimal_trading_sequence': optimal_trading_sequence,
           'data': data}

    return dct


@app.post('/trade')
async def simulateTrade(req: TradePayload):
    stock_name = req.stock
    criteria = req.criteria
    start_date = req.start_date
    n_sim = req.n_sim
    days = req.days
    optimal_trading_sequence = req.optimal_trading_sequence
    money = req.money
    conf_level = req.conf_level

    if start_date == '':
        start_date = '2021-01-01'

    if conf_level == '':
        conf_level == float(99)

    end_date = dt.datetime.now().strftime('%Y-%m-%d')

    stock = fetch_data(stock=stock_name,
                       start_date=start_date,
                       end_date=end_date)

    stock = final_data(data=stock,
                       start_date=start_date,
                       end_date=end_date,
                       criteria=criteria)

    stock = get_return(stock)

    # Kalau pakai data baru
    price = simulation(data=stock, 
                       days=days, 
                       n_sim=n_sim)


    final_sim = trading_sim(price=price, 
                            decision=optimal_trading_sequence, 
                            money=money)
    
    data = []
    for i in range(n_sim):
        profit = final_sim['PROFIT/LOSS'].iloc[i]
        final_balance = final_sim['FINAL_BALANCE'].iloc[i]
        starting_balance = final_sim['STARTING_BALANCE'].iloc[i]

        data.append({'profit': profit,
                     'final_balance': final_balance,
                     'starting_balance': starting_balance})

    var_value = stock_var(data=final_sim, conf_level=conf_level)
    dct = {'data': data,
           'var': {'conf_level': conf_level,
                   'var': var_value}
                   }

    return dct