import ta
import pandas as pd

class Trader:
    def __init__(self):
        self.data = None

    def __repr__(self):
        return "TA"

    def predict(self):
        df = ta.add_all_ta_features(
            self.data, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        print(df)
        df.to_csv('out.csv')
        return 0
        