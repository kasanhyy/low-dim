import numpy as np


def infdist(v, w):
    return max(np.abs(v - w))

def checkW1(v, data):
    return min([infdist(v, w) for w in data])
