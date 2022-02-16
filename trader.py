#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import mysql.connector as mysql
from datetime import datetime
from methods import moving_average
from methods import lstm
from methods import ta
from providers import rh


def db_conn():
    db = mysql.connect(
        host=os.environ.get("MYSQL_HOST"),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DATABASE"),
        port=os.environ.get("MYSQL_PORT"),
        autocommit=True
    )
    return db

def train_lstm():
    window_size = 2
    epochs = 400
    neurons = 1000
    lr = 0.001
    conn = db_conn()
    conn.execute(
        "SELECT ticker, close_price FROM data ORDER BY ticker ASC, begins_at ASC")
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
            trader.model_path = "/opt/methods/models/%s_win-%d_epoch-%d_neuron-%d_lr-%f.h5" % (
                ticker, window_size, epochs, neurons, lr)
            trader.train()
            data_for_ticker = []
        data_for_ticker.append(float(row[1]))
        last_ticker = ticker


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def trade(trader, cache = {}, targets = ['MSFT', 'AMZN', 'AAPL', 'ENPH', 'ICLN', 'JKS'], backtest = False):
    # if the cache is filled, use those as targets
    if cache:
        targets = cache.keys()
    best_ticker = None
    best_slope = 0
    for ticker in targets:
        rh.setup_trader(trader, ticker, cache)
        predictions = trader.predict()
        last_pred = trader.last_prediction(predictions)
        m = slope(0, last_pred[0], 1, last_pred[-1])
        if best_slope < m:
            best_slope = m
            best_ticker = ticker
    if best_slope > 0:
        # print('BEST SLOPE', best_slope)
        conn = db_conn()
        return best_ticker if backtest else rh.simulate_trade(best_ticker, conn)
    # else:
    #     print("NO BEST SLOPE FOUND")
    return None


def main():
    # user = os.environ.get('RH_USER')
    # password = os.environ.get('RH_PASS')
    # trader = ta.Trader()
    # trade(trader)
    trader = lstm.Trader()
    trade(trader)


if __name__ == '__main__':
    main()
