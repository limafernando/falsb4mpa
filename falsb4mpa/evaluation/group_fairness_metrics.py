import numpy as np
import tensorflow as tf

from falsb4mpa.evaluation.baseline_metrics import FNR, FPR, PR, TPR, FNR_soft, FPR_soft
from falsb4mpa.evaluation.grouping_functions import (
    wc_categorical_subgroup,
    categorical_subgroup,
    subgroup,
)

eps = 1e-12


def main():
    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Ypred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.2, 0.3, 0.8, 0.9])
    A = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    assert np.isclose(DI_FP(Y, Ypred, A), abs(1.0 / 6))
    assert np.isclose(DI_FP(Y, Ypred, 1 - A), abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, A), abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, 1 - A), abs(1.0 / 6))


def DI_FP(Y, Ypred, A, adim, wc_scenario=True):
    # print('call di fp')
    if adim == 1:
        fpr1 = subgroup(FPR, A, Y, Ypred)
        fpr0 = subgroup(FPR, 1 - A, Y, Ypred)
        return abs(fpr1 - fpr0)

    if wc_scenario:
        group_difference = wc_categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)
    else:
        group_difference = categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)

    return abs(group_difference)


def DI_TP(Y, Ypred, A, adim):
    # print('call di tp')
    if adim == 1:
        tpr1 = subgroup(TPR, A, Y, Ypred)
        tpr0 = subgroup(TPR, 1 - A, Y, Ypred)
        return abs(tpr1 - tpr0)

    group_difference = wc_categorical_subgroup(fn=TPR, Amask=A, adim=adim, Y=Y, Ypred=Ypred)

    return abs(group_difference)


def DI_FN(Y, Ypred, A):
    # print('call di fn')
    fnr1 = subgroup(FNR, A, Y, Ypred)
    fnr0 = subgroup(FNR, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)


def DI_FP_soft(Y, Ypred, A):
    fpr1 = subgroup(FPR_soft, A, Y, Ypred)
    fpr0 = subgroup(FPR_soft, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)


def DI_FN_soft(Y, Ypred, A):
    fnr1 = subgroup(FNR_soft, A, Y, Ypred)
    fnr0 = subgroup(FNR_soft, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)


"""
SHOULD CONSIDER TRUE POSITIVES AND FALSE POSITIVES
def DI(Y, Ypred, A):
    #print('call di')
    return (DI_FN(Y, Ypred, A) + DI_FP(Y, Ypred, A)) * 0.5
"""


def DEqOdds(Y, Ypred, A, adim):  # deltaEOdds
    # print('call di')
    return 1 - ((DI_TP(Y, Ypred, A, adim) + DI_FP(Y, Ypred, A, adim)) * 0.5)


""" CONSIDER THE TRUE POSITIVE FOR BOTH GROUPS
def DI_soft(Y, Ypred, A): #deltaEOpp
    return (DI_FN_soft(Y, Ypred, A) + DI_FP_soft(Y, Ypred, A)) * 0.5"""


def DEqOpp(Y, Ypred, A, adim, wc_scenario=True):  # deltaEOpp

    if adim == 1:
        tpr1 = subgroup(TPR, A, Y, Ypred)
        tpr0 = subgroup(TPR, 1 - A, Y, Ypred)
        return 1 - (abs(tpr1 - tpr0))

    if wc_scenario:
        group_difference = wc_categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)
    else:
        group_difference = categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)

    return 1 - (abs(group_difference))


def DP(Ypred, A, adim, wc_scenario=True):  # deltaDP
    if adim == 1:
        return 1 - (abs(subgroup(PR, A, Ypred) - subgroup(PR, 1 - A, Ypred)))

    if wc_scenario:
        group_difference = wc_categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)
    else:
        group_difference = categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)

    return 1 - (abs(group_difference))


def NLL(Y, Ypred, eps=eps):
    return -np.mean(
        np.multiply(Y, np.log(Ypred + eps)) + np.multiply(1.0 - Y, np.log(1 - Ypred + eps))
    )


if __name__ == "__main__":
    main()
