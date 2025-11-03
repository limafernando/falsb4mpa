import tensorflow as tf
from tensorflow.python.keras.initializers import GlorotNormal
from falsb4mpa.modeling.zhang.models.classifier import Classifier
from falsb4mpa.modeling.zhang.models.adv import (
    AdversarialDemPar,
    AdversarialEqOdds,
    AdversarialEqOpp,
)


class ZhangMultAdv:
    def __init__(
        self, xdim, ydim, a1dim, a2dim, batch_size, fairdef="EqOdds", initializer=GlorotNormal
    ):

        self.ini = initializer()
        self.batch_size = batch_size
        self.fairdef = fairdef

        self.clf = Classifier(xdim, ydim)

        adv = self.get_adv_model(self.fairdef)  # TODO: check if need double instantiation
        self.adv1 = adv(a1dim)  # initializing the adversarial-1 object
        self.adv2 = adv(a2dim)  # initializing the adversarial-2 object

        self.b = tf.Variable(tf.ones([self.batch_size, 1]), name="b")
        # self.b = tf.Variable(self.ini(shape=(self.batch_size, 1)), name='b')
        # self.b = tf.Variable(tf.zeros([1,1]), name='b')

    def __call__(self, X, Y, A1, A2):

        # ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A1 = tf.dtypes.cast(A1, tf.float32)
        self.A2 = tf.dtypes.cast(A2, tf.float32)

        self.Y_hat = self.clf(self.X, self.b)
        self.A1_hat = self.adv1(self.Y, self.Y_hat, self.b)
        self.A2_hat = self.adv2(self.Y, self.Y_hat, self.b)

        if self.fairdef == "EqOpp":  # trying the filter
            self.adv1_loss = self.__adjust_adv_loss_for_EqOpp__(self.adv1, self.A1, self.A1_hat)
            self.adv2_loss = self.__adjust_adv_loss_for_EqOpp__(self.adv2, self.A2, self.A2_hat)

        else:
            self.adv1_loss = self.adv1.get_loss(self.A1, self.A1_hat)
            self.adv2_loss = self.adv2.get_loss(self.A2, self.A2_hat)

        # self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
        # self.adv_loss = self.adv.get_loss(self.A, self.A_hat)

        self.clf_loss = self.clf.get_loss(self.Y, self.Y_hat)
        self.model_loss = self.clf_loss - (self.adv1_loss + self.adv2_loss)

    def __adjust_adv_loss_for_EqOpp__(
        self,
        adv: AdversarialDemPar | AdversarialEqOdds | AdversarialEqOpp,
        A_real: tf.TensorArray,
        A_predicted: tf.TensorArray,
    ):
        """
        Y_filtered, Y_hat_filtered, A_filtered = self.adv.get_filtered_inputs(self.clf.W, self.Y, self.Y_hat, self.A)
        self.A_hat = self.adv(Y_filtered, Y_hat_filtered, self.b)
        print(self.A_hat.shape, A_filtered.shape, Y_filtered.shape, Y_hat_filtered.shape)
        self.adv_loss = self.adv.get_loss(A_filtered, self.A_hat)
        """

        mask = tf.math.equal(self.Y, 1.0).numpy()  # to consider only where Y = 1
        mask = mask.reshape(
            mask.shape[0],
        )

        col = tf.TensorShape([1])
        lines = None
        shape = None

        A_filtered = tf.boolean_mask(A_real, mask)  # tf.Variable(self.A[mask])

        lines = A_filtered.shape
        shape = lines.concatenate(col)

        A_filtered = tf.reshape(A_filtered, shape)

        A_hat_filtered = tf.Variable(A_predicted[mask])

        lines = A_hat_filtered.shape
        shape = lines.concatenate(col)

        A_hat_filtered = tf.reshape(A_hat_filtered, shape)

        #  adv_loss = adv.get_loss(
        #     tf.math.multiply(self.Y, A_filtered), tf.math.multiply(self.Y, A_hat_filtered)
        # )
        adv_loss = adv.get_loss(
            tf.math.multiply(self.Y, A_real), tf.math.multiply(self.Y, A_predicted)
        )

        return adv_loss

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
