import tensorflow as tf
from tensorflow.python.keras.initializers import GlorotNormal
from falsb4mpa.modeling.madras_laftr.models import Classifier
from falsb4mpa.modeling.zhang.models.adv import (
    AdversarialDemPar,
    AdversarialEqOdds,
    AdversarialEqOpp,
)


class ZhangMultAdv:
    def __init__(self, xdim, ydim, adim, batch_size, fairdef="EqOdds", initializer=GlorotNormal):

        self.ini = initializer()
        self.batch_size = batch_size
        self.fairdef = fairdef
        self.clas = Classifier(xdim, ydim)
        adv = self.get_adv_model(self.fairdef)
        self.adv = adv(adim)  # initializing the adversarial object

        self.b = tf.Variable(tf.ones([self.batch_size, 1]), name="b")
        # self.b = tf.Variable(self.ini(shape=(self.batch_size, 1)), name='b')
        # self.b = tf.Variable(tf.zeros([1,1]), name='b')

    def __call__(self, X, Y, A):

        # ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        self.Y_hat = self.clas(self.X, self.b)

        if self.fairdef == "EqOpp":  # trying the filter
            """
            Y_filtered, Y_hat_filtered, A_filtered = self.adv.get_filtered_inputs(self.clas.W, self.Y, self.Y_hat, self.A)
            self.A_hat = self.adv(Y_filtered, Y_hat_filtered, self.b)
            print(self.A_hat.shape, A_filtered.shape, Y_filtered.shape, Y_hat_filtered.shape)
            self.adv_loss = self.adv.get_loss(A_filtered, self.A_hat)
            """

            self.A_hat = self.adv(self.Y, self.Y_hat, self.b)

            mask = tf.math.equal(self.Y, 1.0).numpy()  # to consider only where Y = 1
            mask = mask.reshape(
                mask.shape[0],
            )

            col = tf.TensorShape([1])
            lines = None
            shape = None

            A_filtered = tf.boolean_mask(self.A, mask)  # tf.Variable(self.A[mask])

            lines = A_filtered.shape
            shape = lines.concatenate(col)

            A_filtered = tf.reshape(A_filtered, shape)

            A_hat_filtered = tf.Variable(self.A_hat[mask])

            lines = A_hat_filtered.shape
            shape = lines.concatenate(col)

            A_hat_filtered = tf.reshape(A_hat_filtered, shape)

            # self.adv_loss = self.adv.get_loss(A_filtered, A_hat_filtered)
            self.adv_loss = self.adv.get_loss(
                tf.math.multiply(self.Y, self.A), tf.math.multiply(self.Y, self.A_hat)
            )

        else:
            self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
            self.adv_loss = self.adv.get_loss(self.A, self.A_hat)

        # self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
        # self.adv_loss = self.adv.get_loss(self.A, self.A_hat)

        self.clas_loss = self.clas.get_loss(self.Y, self.Y_hat)
        self.model_loss = self.clas_loss - self.adv_loss

    def get_adv_model(self, fairdef):
        if fairdef == "DemPar":
            return AdversarialDemPar
        elif fairdef == "EqOdds":
            return AdversarialEqOdds
        elif fairdef == "EqOpp":
            return AdversarialEqOpp
        else:
            print("Not a valid fairness definition! Setting to EqOdds!")
            return AdversarialEqOdds
