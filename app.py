from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


# Extract stock market data
from func import fetch_data, check_columns, check_values, expand_data, final_data, get_return, \
    simulation, generate_decision_sequence, trading_algo, trading_sim
import yfinance as yf

# Importing and transforming file
import pandas as pd

# Data manipulation
import numpy as np
import datetime as dt  # Datetime manipulation


# Flask App
app = Flask(__name__, template_folder='template')
CORS(app)


# Routing
@app.route('/getData', methods=['POST'])
def getData():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        req = request.get_json()
        stock_name = req['stock']
        criteria = req['criteria']
        start_date = req['start_date']

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

        return jsonify(dct)

    else:
        return 'Invalid.'


@app.route('/simulateStock', methods=['POST'])
def simulateStock():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        req = request.get_json()
        stock_name = req['stock']
        criteria = req['criteria']
        start_date = req['start_date']
        n_sim = req['n_sim']
        days = req['days']

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

        stock = get_return(stock)

        price = simulation(data=stock,
                           days=days,
                           n_sim=n_sim)

        decisions = trading_algo(mrx=price)
        decision_sequence = generate_decision_sequence(decisions, n_sim, days)

        estimated_sequence_trading = []
        data = []
        for idx in range(len(decision_sequence)):
            decision = decision_sequence['DECISION'].iloc[idx]

            if decision == 0:
                conf = decision_sequence['HOLD_CONF'].iloc[idx]

            elif decision == 1:
                conf = decision_sequence['SELL_CONF'].iloc[idx]

            else:
                conf = decision_sequence['BUY_CONF'].iloc[idx]

            estimated_sequence_trading.append({'conf': conf,
                                               'decision': decision})

            data.append({'simulation': idx + 1,
                         'data': {'date': list(stock.reset_index().iloc[:, 0]),
                                  'price': list(price[idx])},
                         'trading_sequence': decisions[idx]})

        dct = {'estimated_sequence_trading': estimated_sequence_trading,
               'data': data}

        return jsonify(dct)

    else:
        return 'Invalid.'


if __name__ == '__main__':
    app.debug = True
    app.run()
