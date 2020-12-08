import os
import robin_stocks as rs
import pandas as pd
import numpy as np
from datetime import datetime
from examples import moving_average

targets = ['MSFT', 'SPY', 'AMZN', 'AAPL', 'TCEHY', 'AMD', 'GOOGL', 'NET', 'ENPH', 'ICLN', 'JKS', 'FCEL', 'GEVO']

def main():
    user = os.environ.get('RH_USER')
    password = os.environ.get('RH_PASS')
    rs.login(username=user,
             password=password,
             expiresIn=86400,
             by_sms=True)
    results = {}
    for ticker in targets:
        # historical = rs.stocks.get_stock_historicals(ticker, interval='5minute', span='day')
        historical = rs.stocks.get_stock_historicals(ticker, interval='hour', span='month')
        close = np.array([float(row['close_price']) for row in historical])
        trader = moving_average.Trader(close)
        predictions = trader.predict()
        difference = close[-1] - predictions[-1]
        pct_diff = difference / close[-1]
        results[ticker] = pct_diff
    inorder = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
    # negative numbers mean the pred was higher than reality, meaning you should buy
    print(inorder)
    

if __name__ == '__main__':
    main()
