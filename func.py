# Import modules
# Extract stock market data
import yfinance as yf

# Importing and transforming file
import pandas as pd

# Data manipulation
import numpy as np
import datetime as dt  # Datetime manipulation

# Chatbot
import random
import nltk
from nltk.stem import WordNetLemmatizer

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


def get_return(data, method):
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

        shock = np.array([np.random.normal(loc=0, scale=1)
                         for _ in range(days)])
        drift = np.array([(mu - 1 / 2 * sigma ** 2) for _ in range(days)])

        for x in range(1, days):
            price[x] = price[0] * \
                np.exp(drift[x] * x + sigma * np.sum(shock[:x]))

        return price

    for i in range(n_sim):
        result = monte_carlo(start_price, days)
        table[i] = result
        sim[i] = result[days - 1]

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
        print("Simulation : ", i)

        while j != mrx.shape[1] - 1:
            print(j)
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


def generate_decision_sequence(data, n_sim, days, threshold=0.5):
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

        if max(buy, sell, hold) == hold and (hold / n_sim) > threshold:
            decision_df['DECISION'].iloc[i] = 0

        elif max(buy, sell) == sell:
            decision_df['DECISION'].iloc[i] = 1

        else:
            decision_df['DECISION'].iloc[i] = 2

        decision_df['HOLD_CONF'].iloc[i] = hold / n_sim
        decision_df['BUY_CONF'].iloc[i] = buy / n_sim
        decision_df['SELL_CONF'].iloc[i] = sell / n_sim

    return decision_df


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
            if decision[j]["decision"] == 0:
                pass

            # Sell
            elif decision[j]["decision"] == 1:
                # Proportion of sell decision at given day
                factor = decision[j]["conf"]
                curr_price = price[i, j]
                curr_money += stock * factor * curr_price
                stock -= stock * factor

            # Buy
            else:
                # Proportion of buy decision at given day
                factor = decision[j]["conf"]
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


def validate_decision(data, days, decision, money, n_sim, n_valid):
    df = pd.DataFrame(columns=['SIMULATION', 'WINNING_PERC', 'MAX_RETURN', 'MIN_RETURN'],
                      index=[i for i in range(n_valid)])
    for i in range(n_valid):
        validate = simulation(data=data, days=days, n_sim=n_sim)
        validate_sim = trading_sim(
            price=validate, decision=decision, money=money)

        winning = len(
            validate_sim.loc[validate_sim['PROFIT/LOSS'] >= 0]) / len(validate_sim) * 100
        max_return = np.max(validate_sim['PROFIT/LOSS'])
        min_return = np.min(validate_sim['PROFIT/LOSS'])

        df['SIMULATION'].iloc[i] = i + 1
        df['WINNING_PERC'].iloc[i] = winning
        df['MAX_RETURN'].iloc[i] = max_return
        df['MIN_RETURN'].iloc[i] = min_return

    return df


def stock_var(data, conf_level):
    res = data['PROFIT/LOSS'] / 100
    money = data['STARTING_BALANCE'][0]
    var = np.percentile(res, 100 - conf_level)

    return var * money

# ------------------------------------------------------------------------------------------------- #
# Chatbot


def lemmatize_sentence(sentence):
    # Break the sentence into parts
    lemmatizer = WordNetLemmatizer()
    word_pattern = nltk.word_tokenize(sentence)
    word_patterns = [lemmatizer.lemmatize(word) for word in word_pattern]

    return word_patterns


def words_bag(sentence, words):
    word_patterns = lemmatize_sentence(sentence)
    bag = [0 for _ in range(len(words))]

    # Match the sentence given with existing 'dictionary' of words
    for pattern in word_patterns:
        for idx, word in enumerate(words):
            if word == pattern:
                bag[idx] = 1

    return bag


def predict_tag(model, sentence, words, tags):
    # Lemmatize the sentence and make it a 2D numpy array
    bag = words_bag(sentence, words)
    data = np.array([bag])

    # Predict the intent
    result = model.predict(data)[0]

    # Error threshold, only fetch the result if the probability exceeds the threshold
    threshold = 0.25
    results = [[res, prob]
               for res, prob in enumerate(result) if prob > threshold]

    # If there is an answer with prob above the threshold
    if len(results) != 0:
        results.sort(key=lambda x: x[1], reverse=True)

        # Store the possible intents from the most possible one,
        # indicated by highest probability from the previous line
        intents_lst = []
        for result in results:
            intents_lst.append({
                'intent': tags[result[0]],
                'prob': result[1]
            })

    # If not
    else:
        intents_lst = [{
            'intent': 'unknown',
            'prob': 1
        }]

    return intents_lst


def get_response(intents_lst, intents_json):
    # Get the highest score at top
    tag = intents_lst[0]['intent']
    intents = intents_json['intents']

    response = ''
    for intent in intents:
        if intent['tag'] == tag:
            # Randomly pick the responses from intents.json file
            # which corresponds to specific tag
            response = random.choice(intent['responses'])
            break

    return response
