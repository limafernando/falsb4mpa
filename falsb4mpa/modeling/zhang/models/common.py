from tensorflow._api.v2.math import abs, log
from tensorflow import subtract


def logit(t):
    t = abs(t)
    return subtract(log(t), log(1 - t))
