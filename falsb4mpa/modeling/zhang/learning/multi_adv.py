from copy import deepcopy
from math import isnan


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


from falsb4mpa.evaluation import baseline_metrics
from falsb4mpa.modeling.zhang.models.multi_adv import ZhangMultAdv


def projection(V, A):
    if not np.all(V.numpy()):
        shape = [1, A.shape[1]]
        zeros = tf.zeros(shape, dtype="float32")
        return zeros

    else:
        P = tf.multiply(V, A)
        P = tf.multiply(P, V)
        P = tf.divide(P, tf.norm(V))
        return P


def train(model: ZhangMultAdv, X, Y, A1, A2, optimizer, alpha=1):
    adv_vars = [model.adv1.U, model.adv1.c, model.adv2.U, model.adv2.c, model.b]
    clf_vars = [model.clf.W, model.b]

    clf_opt = deepcopy(optimizer)
    adv1_opt = deepcopy(optimizer)
    adv2_opt = deepcopy(optimizer)

    with (
        tf.GradientTape() as adv1_tape,
        tf.GradientTape() as adv2_tape,
        tf.GradientTape(persistent=True) as clf_tape,
    ):

        model(X, Y, A1, A2)  # to compute the foward
        adv1_loss = model.adv1_loss  # current adversarial loss
        adv2_loss = model.adv2_loss  # current adversarial loss
        clf_loss = model.clf_loss  # current classifier loss
        model_loss = model.model_loss

    if isnan(adv1_loss) or isnan(adv2_loss) or isnan(clf_loss):
        print("any loss is NaN")
        return True

    dULa1 = adv1_tape.gradient(adv1_loss, adv_vars)  # adv_grads
    adv1_opt.apply_gradients(zip(dULa1, adv_vars))

    dULa2 = adv2_tape.gradient(adv2_loss, adv_vars)  # adv_grads
    adv2_opt.apply_gradients(zip(dULa2, adv_vars))

    dWLp = clf_tape.gradient(clf_loss, clf_vars)  # regular grads for classifier

    dWLa1 = clf_tape.gradient(adv1_loss, clf_vars)  # grads for W with the adversarial loss
    dWLa2 = clf_tape.gradient(adv2_loss, clf_vars)  # grads for W with the adversarial loss

    ######

    proj_dWLa_dWLp1 = (
        []
    )  # prevents the classifier from moving in a direction that helps the adversary decrease its loss
    for i in range(len(dWLa1)):
        proj_dWLa_dWLp1.append(projection(dWLa1[i], dWLp[i]))

    max_adv_loss = []  # terms that attemps to increase adv loss
    for i in range(len(dWLa1)):
        max_adv_loss.append(tf.math.multiply(alpha, dWLa1[i]))

    proj_minus_max_adv_loss = []
    for i in range(len(max_adv_loss)):
        proj_minus_max_adv_loss.append(tf.subtract(proj_dWLa_dWLp1[i], max_adv_loss[i]))

    clas_grads = []
    for i in range(len(dWLa1)):
        clas_grads.append(tf.subtract(dWLp[i], proj_minus_max_adv_loss[i]))

    clf_opt.apply_gradients(zip(clas_grads, clf_vars))  # For adv1

    proj_dWLa_dWLp2 = (
        []
    )  # prevents the classifier from moving in a direction that helps the adversary decrease its loss
    for i in range(len(dWLa2)):
        proj_dWLa_dWLp2.append(projection(dWLa2[i], dWLp[i]))

    max_adv_loss = []  # terms that attemps to increase adv loss
    for i in range(len(dWLa2)):
        max_adv_loss.append(tf.math.multiply(alpha, dWLa2[i]))

    proj_minus_max_adv_loss = []
    for i in range(len(max_adv_loss)):
        proj_minus_max_adv_loss.append(tf.subtract(proj_dWLa_dWLp2[i], max_adv_loss[i]))

    clas_grads = []
    for i in range(len(dWLa2)):
        clas_grads.append(tf.subtract(dWLp[i], proj_minus_max_adv_loss[i]))

    clf_opt.apply_gradients(zip(clas_grads, clf_vars))  # For adv2

    model(X, Y, A1, A2)  # to compute the foward
    return False


def train_loop(model: ZhangMultAdv, raw_data, train_dataset, epochs, opt=None):

    # x_train, y_train, a_train = raw_data
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()

    if opt is not None:
        optimizer = opt
        decay4epoch = False
    else:
        decay4epoch = True

    for epoch in range(epochs):
        Y_hat = None
        A1_hat = None
        A2_hat = None

        clf_acc = 0
        adv1_acc = 0
        adv2_acc = 0

        alpha = 1 / (epoch + 1)  # sqrt(epoch+1)

        if decay4epoch:
            lr = 0.001 / (epoch + 1)
            optimizer = Adam(learning_rate=lr)

        for X, Y, A1, A2 in train_dataset:

            r = train(model, X, Y, A1, A2, optimizer, alpha)

            if r:
                print("broken loss")
                print(model.clf_loss, model.adv1_loss, model.adv2_loss)
                break

            Y_hat = model.Y_hat
            A1_hat = model.A1_hat
            A2_hat = model.A2_hat
            clf_acc += baseline_metrics.accuracy(Y, tf.math.round(Y_hat))
            adv1_acc += baseline_metrics.accuracy(A1, tf.math.round(A1_hat))
            adv2_acc += baseline_metrics.accuracy(A2, tf.math.round(A2_hat))

        clf_loss = model.clf_loss
        adv1_loss = model.adv1_loss
        adv2_loss = model.adv2_loss
        clf_acc = clf_acc / dataset_size
        adv1_acc = adv1_acc / dataset_size
        adv2_acc = adv2_acc / dataset_size

        print(
            "> Epoch: {} | Clf loss/acc {:.2f}/{:.2f} | Adv1 loss/acc {:.2f}/{:.2f} | Adv2 loss/acc {:.2f}/{:.2f}".format(
                epoch + 1, clf_loss, clf_acc, adv1_loss, adv1_acc, adv2_loss, adv2_acc
            )
        )
