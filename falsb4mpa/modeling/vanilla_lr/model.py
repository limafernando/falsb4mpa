import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.initializers import GlorotNormal, RandomNormal


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


##########################################################################################################################


class VanillaLogisticRegression:
    def __init__(self, xdim, ydim, batch_size, initializer=GlorotNormal):

        self.ini = initializer()
        self.batch_size = batch_size
        self.clas = Classifier(xdim, ydim)

        self.b = tf.Variable(tf.ones([self.batch_size, 1]), name="b")
        # self.b = tf.Variable(self.ini(shape=(self.batch_size, 1)), name='b')
        # self.b = tf.Variable(tf.zeros([1,1]), name='b')

    def __call__(self, X, Y, A1, A2):

        # ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A1 = tf.dtypes.cast(A1, tf.float32)
        self.A2 = tf.dtypes.cast(A2, tf.float32)

        self.Y_hat = self.clas(self.X, self.b)
        self.clas_loss = self.clas.get_loss(self.Y, self.Y_hat)
        self.model_loss = self.clas_loss
