from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Extract stock market data
from func import fetch_data, check_columns, check_values, expand_data, final_data, get_return, simulation, trading_algo, trading_sim
import yfinance as yf

# Importing and transforming file
import pandas as pd

# Data manipulation
import numpy as np
import re  # Cleaning texts
import time
import datetime as dt  # Datetime manipulation


# Flask App
app = Flask(__name__, template_folder = 'template')
CORS(app)


# Routing
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getData', methods = ['POST'])
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
        stock = fetch_data(stock = stock_name,
                           start_date = start_date,
                           end_date = end_date)
        
        stock = final_data(data = stock, 
                           start_date = start_date,
                           end_date = end_date, 
                           criteria = criteria)

        dct = {'date': list(stock.index.values),
               'values': list(stock.iloc[:, 0])}
        
        return jsonify(dct)

    else:
        return 'Invalid.'


# @app.route('/stockSim', methods = ['POST'])
# def stockSim():
#     content_type = request.headers.get('Content-Type')
#     if (content_type == 'application/json'):
#         req = request.get_json()
#         stocks_id = req['StockCode']
#         stocks_description = req['Description']

#         update_users_stocks(id = stocks_id, 
#                             name = 'stock', 
#                             description = stocks_description)
#         return jsonify(req)

#     else:
#         return 'Invalid.'


if __name__ == '__main__':
    app.debug = True
    app.run()
    