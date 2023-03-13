# Import modules
# Extract stock market data
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
# Fetch Data


def fetch_data(stock, start_date, end_date):
    data = yf.download(stock,
                       start=start_date,
                       end=end_date,
                       progress=False)

    data = data.reset_index()
    return data

# ------------------------------------------------------------------------------------------------- #
# Data Manipulation


def check_columns(data):
    for col in data.columns:
        if col == 'Date':
            if data[col].dtype == 'str':
                data[col] = pd.to_datetime(data[col]).date()

        else:
            data[col] = data[col].astype('float64')

    return data


def check_values(data):
    cols = [col for col in data.columns if data[col].dtype == 'float64']

    # Null values
    data = data.dropna(subset=cols)

    # Negative values
    data = data[(data[cols] > 0).all(1)].reset_index(drop=True)

    return data


def expand_data(data, start_date, end_date):
    # Create temporary dataframe
    temp = pd.DataFrame()
    temp['Date'] = pd.date_range(start=start_date,
                                 end=end_date)

    # Merge the data
    res = pd.merge(temp, data, how='left', on='Date')

    # Interpolate the missing values
    # IDR/USD dataset
    a = res.rolling(3).mean()
    b = res.iloc[::-1].rolling(3).mean()

    c = a.fillna(b).fillna(res).interpolate(method='nearest').ffill().bfill()

    res = res.fillna(c)

    return res

# ------------------------------------------------------------------------------------------------- #
# Final Data Preparation


def final_data(data, start_date, end_date, criteria):
    # Check columns type
    data = check_columns(data=data)

    # Check null and negative values
    data = check_values(data=data)

    # Expand and interpolate
    data = expand_data(data=data,
                       start_date=start_date,
                       end_date=end_date)

    data = data.set_index('Date')

    return data[[criteria]]


def get_return(data):
    data['Daily Return'] = (
        data.iloc[:, 0] - data.iloc[:, 0].shift(1)) / data.iloc[:, 0].shift(1)

    data = data.dropna()

    # sns.displot(data['Daily Return'].dropna(),
    #             bins = 50,
    #             color = 'blue',
    #             kde = True)
    # plt.title('Daily return distribution')
    # plt.show()

    return data

# ------------------------------------------------------------------------------------------------- #
# Monte Carlo Simulation


def simulation(data, days, n_sim):
    start_price = data.iloc[-1, 0]
    sim = np.zeros(n_sim)
    table = np.zeros((n_sim, days))

    delta = 1 / days
    mu = data['Daily Return'].mean()
    sigma = data['Daily Return'].std()

    def monte_carlo(start_price, days, mu=mu, sigma=sigma):
        price = np.zeros(days)
        price[0] = start_price

        shock = np.zeros(days)
        drift = np.zeros(days)

        for x in range(1, days):
            shock[x] = np.random.normal(
                loc=mu * delta, scale=sigma * np.sqrt(delta))
            drift[x] = mu * delta

            price[x] = price[x - 1] + (price[x - 1] * (drift[x] + shock[x]))

        return price

    for i in range(n_sim):
        result = monte_carlo(start_price, days)
        table[i] = result
        sim[i] = result[days - 1]
        plt.plot(result)

    return table

# ------------------------------------------------------------------------------------------------- #
# Trading Algo and Simulation


def trading_algo(mrx):
    '''
    0: Hold
    1: Sell
    2: Buy
    '''
    res = np.zeros((mrx.shape[0], mrx.shape[1]))
    n_sim = mrx.shape[0]

    for i in range(n_sim):
        j = 1
        while j != mrx.shape[1] - 1:
            start = j
            cond = True

            # Check increasing
            if mrx[i, j] > mrx[i, j - 1]:
                while cond:

                    # Still increasing and not at the end
                    if (mrx[i, j] > mrx[i, j - 1]) & (j != mrx.shape[1] - 1):
                        j = j + 1

                    # At the end or price dropped
                    else:
                        # Still increasing at the end, sell
                        if mrx[i, j] > mrx[i, j - 1]:
                            res[i, j] = 1
                            res[i, start: j] = 0
                            cond = False

                        # Price dropped, sell at prev price
                        else:
                            res[i, j - 1] = 1
                            res[i, start: j - 1] = 0
                            cond = False

            # Check decreasing
            else:
                while cond:

                    # Still decreasing and not at the end
                    if (mrx[i, j] < mrx[i, j - 1]) & (j != mrx.shape[1] - 1):
                        j = j + 1

                    # At the end or price increased
                    else:
                        # Still decreasing at the end, buy
                        if mrx[i, j] < mrx[i, j - 1]:
                            res[i, j] = 2
                            res[i, start: j] = 0
                            cond = False

                        # Price increased, buy at prev price
                        else:
                            res[i, j - 1] = 2
                            res[i, start: j - 1] = 0
                            cond = False

    return res


def trading_sim(price, decision, money):
    '''
    0: Hold
    1: Sell
    2: Buy
    '''
    res = np.zeros(price.shape[0])
    n_sim = price.shape[0]
    days = price.shape[1]
    result_df = pd.DataFrame(columns=['SIMULATION', 'PROFIT/LOSS', 'FINAL_BALANCE', 'STARTING_BALANCE'],
                             index=[i for i in range(n_sim)])

    for i in range(n_sim):
        # Initial stock at hand
        stock = 0
        curr_money = money

        for j in range(days):
            # Hold
            if decision['DECISION'].iloc[j] == 0:
                pass

            # Sell
            elif decision['DECISION'].iloc[j] == 1:
                # Proportion of sell decision at given day
                factor = decision['SELL_CONF'].iloc[j] / \
                    (decision['SELL_CONF'].iloc[j] +
                     decision['BUY_CONF'].iloc[j])
                curr_price = price[i, j]
                curr_money += stock * factor * curr_price
                stock -= stock * factor

            # Buy
            else:
                # Proportion of buy decision at given day
                factor = decision['BUY_CONF'].iloc[j] / \
                    (decision['SELL_CONF'].iloc[j] +
                     decision['BUY_CONF'].iloc[j])
                curr_price = price[i, j]
                stock += factor * curr_money / curr_price
                curr_money -= factor * curr_money

        # Final day result
        res[i] = curr_money + stock * price[i, -1]

        result_df['SIMULATION'].iloc[i] = i + 1
        result_df['FINAL_BALANCE'].iloc[i] = curr_money + stock * price[i, -1]
        result_df['STARTING_BALANCE'].iloc[i] = money

    pct_return = 100 * (res - money) / money
    result_df['PROFIT/LOSS'] = pct_return

    return result_df

# ------------------------------------------------------------------------------------------------- #


def generate_decision_sequence(data, n_sim, days):
    decision_df = pd.DataFrame(columns=['DAY', 'DECISION', 'HOLD_CONF', 'BUY_CONF', 'SELL_CONF'],
                               index=[i for i in range(days)])

    for i in range(days):
        decision_df['DAY'].iloc[i] = i + 1
        buy = 0
        sell = 0
        hold = 0

        for j in range(n_sim):
            if data[j, i] == 0:
                hold += 1

            elif data[j, i] == 1:
                sell += 1

            else:
                buy += 1

        if max(buy, sell, hold) == hold and (hold / n_sim) > 0.5:
            decision_df['DECISION'].iloc[i] = 0

        elif max(buy, sell) == sell:
            decision_df['DECISION'].iloc[i] = 1

        else:
            decision_df['DECISION'].iloc[i] = 2

        decision_df['HOLD_CONF'].iloc[i] = hold / n_sim
        decision_df['BUY_CONF'].iloc[i] = buy / n_sim
        decision_df['SELL_CONF'].iloc[i] = sell / n_sim

    return decision_df
