import numpy as np
import tensorflow as tf


def subgroup(fn, mask, Y, Ypred=None):
    # print('call subgroup')
    m = np.greater(mask, 0.5).flatten()
    # print(m[:5])
    flatten_Y = tf.reshape(Y, [-1])
    if not Ypred is None:  # two-argument functions
        flatten_Ypred = tf.reshape(Ypred, [-1])
        # print('call function {}'.format(fn))
        return fn(
            flatten_Y[m], flatten_Ypred[m]
        )  # access the indexes that are True (the True check subgroup)
    else:  # one-argument functions
        return fn(flatten_Y[m])


def intersectional_subgroup(fn, mask1, mask2, Y, Ypred=None):
    m1 = np.greater(mask1, 0.5).flatten()
    m2 = np.greater(mask2, 0.5).flatten()

    m = m1 & m2

    flatten_Y = tf.reshape(Y, [-1])

    if not Ypred is None:  # two-argument functions
        flatten_Ypred = tf.reshape(Ypred, [-1])
        return fn(
            flatten_Y[m], flatten_Ypred[m]
        )  # access the indexes that are True (the True check subgroup)

    else:  # one-argument functions
        return fn(flatten_Y[m])


def categorical_subgroup(fn, Amask, adim, Y, Ypred=None, priviliged_idx=-1):
    # in our data prep for adult dataset
    # the race order id 'race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other','race_ White'
    # we want to know the difference between 'race_ White' and the others

    groups_difference = []

    priviliged_result = subgroup(fn, Amask[:, priviliged_idx], Y, Ypred)

    if priviliged_idx == -1:
        idx_list = [idx for idx in range(adim)][:-1]
    else:
        idx_list = [idx for idx in range(adim) if idx != priviliged_idx]

    for group_idx in idx_list:
        group_result = subgroup(fn, Amask[:, group_idx], Y, Ypred)
        groups_difference.append(abs(priviliged_result - group_result))
    # print(groups_difference)
    reduce_mean = sum(groups_difference) / len(groups_difference)

    return reduce_mean

    # previous categorical_subgroup implementation
    # group_difference = []
    # for group_idx in range(adim):
    #     if group_difference:
    #         group_difference -= subgroup(fn, Amask[:, group_idx], Y, Ypred)
    #     else:
    #         group_difference = subgroup(fn, Amask[:, group_idx], Y, Ypred)

    # return group_difference
