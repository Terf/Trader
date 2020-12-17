import numpy as np
from scipy.ndimage.filters import uniform_filter1d

class Trader:
    def __init__(self, prices):
        self.prices = prices

    def running_mean(self, n):
        # https://stackoverflow.com/a/43200476/2624391
        # cumsum = np.cumsum(np.insert(self.prices, 0, 0))
        # return (cumsum[n:] - cumsum[:-n]) / float(n)
        return uniform_filter1d(self.prices, size=n, mode = 'nearest')

    def predict(self):
        return self.running_mean(3)
        