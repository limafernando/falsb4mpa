import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.python.keras.initializers import GlorotNormal, RandomNormal
from falsb4mpa.modeling.zhang.models.common import logit

EPS = 7e-8


class Adversarial:
    def __init__(self, initializer=GlorotNormal):

        # self.ini = initializer()
        self.ini = RandomNormal(mean=0.0, stddev=1.5)
        self.U = tf.Variable(1.0, name="U")  # only to initialize
        # self.U = tf.Variable(self.ini(shape=(1, 3)), name='U')
        self.is_built = False
        # self.c = tf.Variable(self.ini(shape=(1,1)), name='c')
        self.c = tf.Variable(tf.ones([1, 1]), name="c")

    def __call__(self):
        pass

    def build(self, shape):
        return tf.Variable(self.ini(shape=shape), name="U")  # check shape

    def get_loss(self, A, A_hat):
        A_hat = tf.clip_by_value(A_hat, 1e-9, 1.0)
        if self.adim == 1:
            bce = BinaryCrossentropy(from_logits=False)
            return bce(A, A_hat)
        else:
            cce = CategoricalCrossentropy(from_logits=False)
            return cce(A, A_hat)

    """
    def get_filtered_inputs(self, W, Y, Y_hat, A):
        col = tf.TensorShape([1])
        lines = None
        shape = None
        
        mask = tf.math.equal(Y, 1.)

        #W_filtered = tf.Variable(W[mask])
        #lines = W_filtered.shape
        #shape = col.concatenate(lines)
        #W_filtered = tf.reshape(W_filtered, shape)
        #print('W_fil ', W_filtered.shape)

        Y_filtered = tf.Variable(Y[mask])
        lines = Y_filtered.shape
        shape = lines.concatenate(col)
        Y_filtered = tf.reshape(Y_filtered, shape)

        Y_hat_filtered = tf.Variable(Y_hat[mask])
        lines = Y_hat_filtered.shape
        shape = lines.concatenate(col)
        Y_hat_filtered = tf.reshape(Y_hat_filtered, shape)
        
        A_filtered = tf.Variable(A[mask])
        lines = A_filtered.shape
        shape = lines.concatenate(col)
        A_filtered = tf.reshape(A_filtered, shape)

        return Y_filtered, Y_hat_filtered, A_filtered#, W_filtered
        """


class AdversarialDemPar(Adversarial):
    def __init__(self, adim):
        super(AdversarialDemPar, self).__init__()
        self.adim = adim
        # self.U = tf.Variable(self.ini(shape=(1, 1)), name='U')
        self.U = tf.Variable(tf.zeros([self.adim, 1]), name="U")

    def __call__(self, Y, Y_hat, b):

        self.S = tf.math.sigmoid(
            tf.multiply(
                (1 + tf.math.abs(self.c)),
                logit(Y_hat - EPS),  # here we add EPS to ensure we wont get a NaN
            )
        )

        # self.S = Y_hat

        """if not self.is_built:
            U_shape = (1, self.S.shape[1])
            self.U = self.build(U_shape)
            #print(self.U)
            self.is_built = True #check how to improve this build of U"""
        if self.adim == 1:
            self.A_hat = tf.math.sigmoid(tf.add(tf.matmul(self.S, tf.transpose(self.U)), b))
        else:
            self.A_hat = tf.math.softmax(tf.add(tf.matmul(self.S, tf.transpose(self.U)), b))

        return self.A_hat


class AdversarialEqOdds(Adversarial):
    def __init__(self, adim):
        super(AdversarialEqOdds, self).__init__()
        self.adim = adim
        # self.U = tf.Variable(self.ini(shape=(1, 3)), name='U')
        self.U = tf.Variable(tf.zeros([self.adim, 3]), name="U")

    def __call__(self, Y, Y_hat, b):

        self.S = tf.math.sigmoid(
            tf.multiply(
                (1 + tf.math.abs(self.c)),
                logit(Y_hat - EPS),  # here we add EPS to ensure we wont get a NaN
            )
        )
        # print('S ', self.S.shape)
        concatenation = tf.concat(
            [self.S, tf.multiply(self.S, Y), tf.multiply(self.S, 1 - Y)], axis=1
        )
        # print('conc ', concatenation.shape)
        """if not self.is_built:
            U_shape = (1, concatenation.shape[1])
            self.U = self.build(U_shape)
            #print(self.U)
            self.is_built = True #check how to improve this build of U - getting a warning
        #print('U ', self.U.shape)"""

        if self.adim == 1:
            self.A_hat = tf.math.sigmoid(tf.add(tf.matmul(concatenation, tf.transpose(self.U)), b))
        else:
            self.A_hat = tf.math.softmax(tf.add(tf.matmul(concatenation, tf.transpose(self.U)), b))

        # print('A_hat ', self.A_hat.shape)
        return self.A_hat


class AdversarialEqOpp(AdversarialEqOdds):
    def __init__(self, adim):
        super(AdversarialEqOpp, self).__init__(adim)

    def filter_As(self, A, A_hat, Y):
        pass