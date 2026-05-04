import tensorflow as tf
import numpy as np

from falsb4mpa.evaluation import (
    baseline_metrics,
    intersectional_fairness_metrics,
    group_fairness_metrics,
    grouping_functions,
)
from falsb4mpa.evaluation.predictive_metrics import f1_macro, f1_micro
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


def compute_predictive_metrics(Y, Y_hat):
    print("> Predictive Performance Evaluation")
    # Y = Y.numpy()
    # A = A.numpy()

    Y_hat = tf.math.round(Y_hat)
    clas_acc = baseline_metrics.accuracy(Y, Y_hat)
    print("> Class Acc = {}".format(clas_acc))

    clas_f1_micro = f1_micro(Y, Y_hat)
    clas_f1_macro = f1_macro(Y, Y_hat)

    tp = baseline_metrics.TP(Y, Y_hat.numpy())
    tn = baseline_metrics.TN(Y, Y_hat.numpy())
    fp = baseline_metrics.FP(Y, Y_hat.numpy())
    fn = baseline_metrics.FN(Y, Y_hat.numpy())

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    print(
        "> Confusion Matrix \n"
        + "TN: {} | FP: {} \n".format(tn, fp)
        + "FN: {} | TP: {}".format(fn, tp)
    )

    return clas_acc, clas_f1_micro, clas_f1_macro, confusion_matrix


def compute_adv_metrics(A, A_hat=None):
    if A_hat is not None:
        A_hat = tf.math.round(A_hat)
        adv_acc = baseline_metrics.accuracy(A, A_hat)
        print("> Adv Acc = {}".format(adv_acc))

    return adv_acc


def compute_fair_metrics(Y, A, Y_hat, adim=1):
    print("> Fairness Evaluation")
    # Y = Y.numpy()
    # A = A.numpy()

    Y_hat = tf.math.round(Y_hat)

    dp = group_fairness_metrics.DP(Y_hat.numpy(), A, adim)
    deqodds = group_fairness_metrics.DEqOdds(Y, Y_hat.numpy(), A, adim)
    deqopp = group_fairness_metrics.DEqOpp(Y, Y_hat.numpy(), A, adim)

    print("> DP | DEqOdds | DEqOpp")
    print("> {} | {} | {}".format(dp, deqodds, deqopp))

    if adim == 1:
        metrics_g0, metrics_g1 = group_confusion_matrix(A, Y, Y_hat)
        return dp, deqodds, deqopp, metrics_g0, metrics_g1

    return dp, deqodds, deqopp  # , metrics_g0, metrics_g1


def compute_intersectional_fair_metrics(Y, A1, A2, Y_hat, a1dim=1, a2dim=1):
    print("> Intersectional Fairness Evaluation")

    Y_hat = tf.math.round(Y_hat)

    wc_spd = intersectional_fairness_metrics.wc_spd(Y_hat, A1, a1dim, A2, a2dim)
    wc_aod = intersectional_fairness_metrics.wc_aod(Y, Y_hat, A1, a1dim, A2, a2dim)
    wc_eod = intersectional_fairness_metrics.wc_eod(Y, Y_hat, A1, a1dim, A2, a2dim)

    wc_spd = intersectional_fairness_metrics.opt_wc_spd(wc_spd)
    wc_aod = intersectional_fairness_metrics.opt_wc_aod(wc_aod)
    wc_eod = intersectional_fairness_metrics.opt_wc_eod(wc_eod)

    print("> WC_SPD | WC_AOD | WC_EOD")
    print("> {} | {} | {}".format(wc_spd, wc_aod, wc_eod))

    return wc_spd, wc_aod, wc_eod


def evaluation(model, data):
    Y_hat = None
    Y_real = None
    A_real = None
    batch_count = 1

    for X, Y, A1, A2 in data:

        model(X, Y, A1, A2)

        if batch_count == 1:
            Y_hat = model.Y_hat
            Y_real = Y
            A1_real = A1
            A2_real = A2
            batch_count += 1
        else:
            Y_hat = tf.concat([Y_hat, model.Y_hat], 0)

            Y_real = tf.concat([Y_real, Y], 0)
            A1_real = tf.concat([A1_real, A1], 0)
            A2_real = tf.concat([A2_real, A2], 0)

    return Y_real, A1_real, A2_real, Y_hat


def compute_tradeoff(performance_metric, fairness_metric):
    tradeoff = 2 * (performance_metric * fairness_metric) / (performance_metric + fairness_metric)
    return tradeoff


def group_confusion_matrix(A, Y, Y_hat):
    fn_metrics = [
        baseline_metrics.TN,
        baseline_metrics.FP,
        baseline_metrics.FN,
        baseline_metrics.TP,
    ]
    # if adim == 1:
    metrics_a0 = [0, 0, 0, 0]
    metrics_a1 = [0, 0, 0, 0]
    for i in range(len(fn_metrics)):
        metrics_a0[i] = grouping_functions.subgroup(fn_metrics[i], A, Y, Y_hat.numpy())
        metrics_a1[i] = grouping_functions.subgroup(fn_metrics[i], 1 - A, Y, Y_hat.numpy())

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
