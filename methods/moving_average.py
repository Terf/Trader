import numpy as np
from scipy.ndimage.filters import uniform_filter1d

class Trader:
    def __init__(self):
        self.prices = []

    def __repr__(self):
        return "MA"

    def running_mean(self, n):
        # https://stackoverflow.com/a/43200476/2624391
        # cumsum = np.cumsum(np.insert(self.prices, 0, 0))
        # return (cumsum[n:] - cumsum[:-n]) / float(n)
        return uniform_filter1d(self.prices, size=n, mode = 'nearest')

    def predict(self):
        return self.running_mean(3)

    def last_prediction(self, predictions):
        # return the last 2 points
        return predictions[-2:]
        