#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import robin_stocks as rs
import numpy as np
import pandas as pd


def setup_trader(trader, ticker, cache):
    if ticker not in cache:
        historical = rs.stocks.get_stock_historicals(
            ticker, interval="5minute", span="day"
        )
        cache[ticker] = historical
    else:
        historical = cache[ticker]
    close = np.array([float(row["close_price"]) for row in historical])
    if str(trader) == "LSTM":
        trader.prices = close
        trader.window_size = 2
        trader.epochs = 400
        trader.neurons = 1000
        trader.lr = 0.001
        trader.model_path = (
            "/opt/methods/models/%s_win-%d_epoch-%d_neuron-%d_lr-%f.h5"
            % (ticker, 2, 400, 1000, 0.001)
        )
    elif str(trader) == "TA":
        open = np.array([float(row["open_price"]) for row in historical])
        high = np.array([float(row["high_price"]) for row in historical])
        low = np.array([float(row["low_price"]) for row in historical])
        volume = np.array([int(row["volume"]) for row in historical])
        trader.data = pd.DataFrame(
            {"close": close, "open": open, "high": high, "low": low, "volume": volume}
        )
    elif str(trader) == "MA":
        trader.prices = close
    else:
        raise ValueError("Unrecognized trader: %s" % (trader))


def simulate_trade(ticker, conn, at="CURRENT_TIMESTAMP"):
    conn.execute(
        "SELECT ticker, shares, price FROM `simulation` ORDER BY `simulation`.`executed_at` DESC LIMIT 1"
    )
    last_trade = conn.fetchone()
    trade_sql = (
        "INSERT INTO `simulation` (`id`, `executed_at`, `action`, `price`, `shares`, `ticker`) VALUES (NULL, "
        + at
        + ", %s, %s, %s, %s)"
    )
    if last_trade == None:
        money = 1000
    elif last_trade[0] != ticker:
        money = float(last_trade[1]) * float(last_trade[2])
        cur_price = float(rs.stocks.get_quotes(last_trade[0], "bid_price")[0])
        conn.execute(trade_sql, ("sell", cur_price, last_trade[1], last_trade[0]))
        print("Selling", last_trade[1])
    else:
        print("holding position with", ticker)
        return

    print("Buying", ticker)
    cur_price = float(rs.stocks.get_quotes(ticker, "ask_price")[0])
    conn.execute(
        trade_sql,
        (
            "buy",
            cur_price,
            money / cur_price,
            ticker,
        ),
    )

def login(user, password):
    rs.login(username=user, password=password, expiresIn=86400, by_sms=True)
