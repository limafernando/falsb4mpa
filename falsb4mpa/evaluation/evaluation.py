import tensorflow as tf
import numpy as np

from falsb4mpa.evaluation import metrics
from falsb4mpa.modeling.zhang.models.multi_adv import ZhangMultAdv


def fair_evaluation(model: ZhangMultAdv, data):
    Y_hat = None
    A1_hat = None
    A2_hat = None
    Y_real = None
    A1_real = None
    A2_real = None
    batch_count = 1

    for X, Y, A1, A2 in data:

        model(X, Y, A1, A2)

        if batch_count == 1:
            Y_hat = model.Y_hat
            A1_hat = model.A1_hat
            A2_hat = model.A2_hat
            Y_real = Y
            A1_real = A1
            A2_real = A2
            batch_count += 1
        else:
            Y_hat = tf.concat([Y_hat, model.Y_hat], 0)
            A1_hat = tf.concat([A1_hat, model.A1_hat], 0)
            A2_hat = tf.concat([A2_hat, model.A2_hat], 0)
            Y_real = tf.concat([Y_real, Y], 0)
            A1_real = tf.concat([A1_real, A1], 0)
            A2_real = tf.concat([A2_real, A2], 0)

    return Y_real, A1_real, A2_real, Y_hat, A1_hat, A2_hat


def compute_metrics(Y, A, Y_hat, A_hat=None, adim=1):
    print("> Evaluation")
    # Y = Y.numpy()
    # A = A.numpy()

    Y_hat = tf.math.round(Y_hat)
    clas_acc = metrics.accuracy(Y, Y_hat)
    print("> Class Acc = {}".format(clas_acc))

    if A_hat is not None:
        A_hat = tf.math.round(A_hat)
        adv_acc = metrics.accuracy(A, A_hat)
        print("> Adv Acc = {}".format(clas_acc, adv_acc))

    dp = metrics.DP(Y_hat.numpy(), A, adim)
    deqodds = metrics.DEqOdds(Y, Y_hat.numpy(), A, adim)
    deqopp = metrics.DEqOpp(Y, Y_hat.numpy(), A, adim)

    print("> DP | DEqOdds | DEqOpp")
    print("> {} | {} | {}".format(dp, deqodds, deqopp))

    tp = metrics.TP(Y, Y_hat.numpy())
    tn = metrics.TN(Y, Y_hat.numpy())
    fp = metrics.FP(Y, Y_hat.numpy())
    fn = metrics.FN(Y, Y_hat.numpy())

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    print(
        "> Confusion Matrix \n"
        + "TN: {} | FP: {} \n".format(tn, fp)
        + "FN: {} | TP: {}".format(fn, tp)
    )

    if adim == 1:
        metrics_g0, metrics_g1 = group_confusion_matrix(A, Y, Y_hat)
        return clas_acc, dp, deqodds, deqopp, confusion_matrix, metrics_g0, metrics_g1

    return clas_acc, dp, deqodds, deqopp, confusion_matrix  # , metrics_g0, metrics_g1


def evaluation(model, data):
    Y_hat = None
    Y_real = None
    A_real = None
    batch_count = 1

    for X, Y, A in data:

        model(X, Y, A)

        if batch_count == 1:
            Y_hat = model.Y_hat
            Y_real = Y
            A_real = A
            batch_count += 1
        else:
            Y_hat = tf.concat([Y_hat, model.Y_hat], 0)

            Y_real = tf.concat([Y_real, Y], 0)
            A_real = tf.concat([A_real, A], 0)

    return Y_real, A_real, Y_hat


def compute_tradeoff(performance_metric, fairness_metric):
    tradeoff = 2 * (performance_metric * fairness_metric) / (performance_metric + fairness_metric)
    return tradeoff


def group_confusion_matrix(A, Y, Y_hat):
    fn_metrics = [metrics.TN, metrics.FP, metrics.FN, metrics.TP]
    # if adim == 1:
    metrics_a0 = [0, 0, 0, 0]
    metrics_a1 = [0, 0, 0, 0]
    for i in range(len(fn_metrics)):
        metrics_a0[i] = metrics.subgroup(fn_metrics[i], A, Y, Y_hat.numpy())
        metrics_a1[i] = metrics.subgroup(fn_metrics[i], 1 - A, Y, Y_hat.numpy())

    print(
        "> Confusion Matrix for A = 0 \n"
        + "TN: {} | FP: {} \n".format(metrics_a0[0], metrics_a0[1])
        + "FN: {} | TP: {}".format(metrics_a0[2], metrics_a0[3])
    )

    print(
        "> Confusion Matrix for A = 1 \n"
        + "TN: {} | FP: {} \n".format(metrics_a1[0], metrics_a1[1])
        + "FN: {} | TP: {}".format(metrics_a1[2], metrics_a1[3])
    )

    # for i in range(len(fn_metrics)):
    #     metrics_a0[i] = metrics.categorical_subgroup(fn_metrics[i], A, Y, Y_hat.numpy())
    #     metrics_a1[i] = metrics.subgroup(fn_metrics[i], 1 - A, Y, Y_hat.numpy())

    #     print('> Confusion Matrix for A = 0 \n' +
    #             'TN: {} | FP: {} \n'.format(metrics_a0[0], metrics_a0[1]) +
    #             'FN: {} | TP: {}'.format(metrics_a0[2], metrics_a0[3]))

    return metrics_a0, metrics_a1
