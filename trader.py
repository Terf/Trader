#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import mysql.connector as mysql
import robin_stocks as rs
from datetime import datetime
from methods import moving_average
from methods import lstm

# targets = ['MSFT', 'SPY', 'AMZN', 'AAPL', 'TCEHY', 'AMD', 'GOOGL', 'NET', 'ENPH', 'ICLN', 'JKS', 'FCEL', 'GEVO']
targets = ['SPY']

def train_lstm():
    window_size = 2
    epochs = 400
    neurons = 1000
    lr = 0.001
    db = mysql.connect(
        host=os.environ.get("MYSQL_HOST"),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DATABASE"),
        port=os.environ.get("MYSQL_PORT")
    )
    conn = db.cursor()
    conn.execute("SELECT ticker, close_price FROM data ORDER BY ticker ASC, begins_at ASC")
    last_ticker = None
    data_for_ticker = []
    for row in conn.fetchall():
        ticker = row[0]
        if last_ticker is not None and last_ticker != ticker:
            trader = lstm.Trader()
            trader.prices = data_for_ticker
            trader.window_size = window_size
            trader.epochs = epochs
            trader.neurons = neurons
            trader.lr = lr
            trader.model_path = "/var/keras/%s_win-%d_epoch-%d_neuron-%d_lr-%f.h5" % (ticker, window_size, epochs, neurons, lr)
            trader.train()
            data_for_ticker = []
        data_for_ticker.append(float(row[1]))
        last_ticker = ticker


def trade(trader):
    window_size = 2
    epochs = 400
    neurons = 1000
    lr = 0.001
    for ticker in targets:
        historical = rs.stocks.get_stock_historicals(ticker, interval='5minute', span='day')
        close = np.array([float(row['close_price']) for row in historical])
        trader.prices = close
        trader.window_size = window_size
        trader.epochs = epochs
        trader.neurons = neurons
        trader.lr = lr
        trader.model_path = "/var/keras/%s_win-%d_epoch-%d_neuron-%d_lr-%f.h5" % (ticker, window_size, epochs, neurons, lr)
        trader.predict(True)

def main():
    user = os.environ.get('RH_USER')
    password = os.environ.get('RH_PASS')
    rs.login(username=user,
             password=password,
             expiresIn=86400,
             by_sms=True)
    trader = lstm.Trader()
    trade(trader)

if __name__ == '__main__':
    main()
