import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.python.keras.initializers import GlorotNormal, RandomNormal
from math import isnan

EPS = 7e-8


class Classifier:
    def __init__(self, xdim, ydim, initializer=GlorotNormal):
        self.ydim = ydim
        # self.ini = initializer()
        # self.ini = RandomNormal(mean=0.0, stddev=1.5)

        # self.W = tf.Variable(self.ini(shape=(1, xdim)), name='W')
        self.W = tf.Variable(tf.zeros([1, xdim]), name="W")
        # self.W = tf.Variable(tf.ones([1, xdim]), name='W')

    def __call__(self, X, b):
        if self.ydim == 1:
            self.Y_hat = tf.math.sigmoid(tf.add(tf.matmul(X, tf.transpose(self.W)), b))

        else:
            self.Y_hat = tf.math.softmax(tf.add(tf.matmul(X, tf.transpose(self.W)), b))

        return self.Y_hat

    def get_loss(self, Y, Y_hat):
        Y_hat = tf.clip_by_value(Y_hat, 1e-9, 1.0)
        if self.ydim == 1:
            bce = BinaryCrossentropy(from_logits=False)
            return bce(Y, Y_hat)
        else:
            cce = CategoricalCrossentropy(from_logits=False)
            return cce(Y, Y_hat)
