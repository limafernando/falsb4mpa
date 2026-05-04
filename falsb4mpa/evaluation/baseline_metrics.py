import numpy as np
import tensorflow as tf

eps = 1e-12


def main():
    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Ypred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.2, 0.3, 0.8, 0.9])
    A = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

    assert pos(Y) == 5
    assert neg(Y) == 5
    assert pos(Ypred) == 5
    assert neg(Ypred) == 5

    assert TP(Y, Ypred) == 3
    assert FP(Y, Ypred) == 2
    assert TN(Y, Ypred) == 3
    assert FN(Y, Ypred) == 2

    assert np.isclose(TPR(Y, Ypred), 0.6)
    assert np.isclose(FPR(Y, Ypred), 0.4)
    assert np.isclose(TNR(Y, Ypred), 0.6)
    assert np.isclose(FNR(Y, Ypred), 0.4)
    assert np.isclose(calibPosRate(Y, Ypred), 0.6)
    assert np.isclose(calibNegRate(Y, Ypred), 0.6)
    assert np.isclose(errRate(Y, Ypred), 0.4)
    assert np.isclose(accuracy(Y, Ypred), 0.6)


def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)


def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)


def PR(Y):  # pos rate
    return pos(Y) / (pos(Y) + neg(Y))


def NR(Y):  # neg rate
    return neg(Y) / (pos(Y) + neg(Y))


def TP(Y, Ypred):  # true pos
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)


def FP(Y, Ypred):  # false pos
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)


def TN(Y, Ypred):  # true neg
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(
        np.float32
    )


def FN(Y, Ypred):  # false neg
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)


def FP_soft(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), Ypred)).astype(np.float32)


def FN_soft(Y, Ypred):
    return np.sum(np.multiply(Y, 1 - Ypred)).astype(np.float32)


# note: TPR + FNR = 1; TNR + FPR = 1
def TPR(Y, Ypred):  # TP rate
    return 1 - FNR(Y, Ypred)


def FPR(Y, Ypred):  # FP rate
    negY = neg(Y)
    if negY == 0:
        return 0
    else:
        return FP(Y, Ypred) / negY


def TNR(Y, Ypred):  # TN rate
    return TN(Y, Ypred) / neg(Y)


def FNR(Y, Ypred):  # FP rate
    posY = pos(Y)
    if posY == 0:
        return 0
    else:
        return FN(Y, Ypred) / posY


def FPR_soft(Y, Ypred):
    return FP_soft(Y, Ypred) / neg(Y)


def FNR_soft(Y, Ypred):
    return FN_soft(Y, Ypred) / pos(Y)


def calibPosRate(Y, Ypred):
    return TP(Y, Ypred) / pos(Ypred)


def calibNegRate(Y, Ypred):
    return TN(Y, Ypred) / neg(Ypred)


def errRate(Y, Ypred):
    return (FP(Y, Ypred) + FN(Y, Ypred)) / float(Y.shape[0])


"""def accuracy(Y, Ypred):
    return 1 - errRate(Y, Ypred)"""


def accuracy(Y, Ypred):

    if len(Y.shape) > 1 and Y.shape[1] > 1:
        acc = tf.keras.metrics.CategoricalAccuracy()
    else:
        acc = tf.keras.metrics.BinaryAccuracy()

    acc.update_state(y_true=Y, y_pred=Ypred)
    return acc.result().numpy()


if __name__ == "__main__":
    main()
