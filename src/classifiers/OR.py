"""
Implementation of logistic ordinal regression (aka proportional odds) model in tensorflow
Fitting is done my minimizing the negative log likelihood according to [1]
With an additional logarithmic barrier penalty to enforce monotonicity of the thresholds

References
----------
[1] Agresti, Alan. Categorical data analysis. Vol. 482. John Wiley & Sons, 2003.
[2] http://fa.bianp.net/blog/2013/logistic-ordinal-regression/
[3] http://fa.bianp.net/blog/2013/loss-functions-for-ordinal-regression/
[4] https://github.com/fabianp/minirank/blob/master/minirank/logistic.py (source code)
[5] https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

Author: Stijn Decubber
"""

import tensorflow as tf
import numpy as np
from keras import backend as K


class ProportionalOdds_TF(object):
    def __init__(self, n_features, n_classes, g, q):
        self.n_features = n_features
        self.n_classes = n_classes
        self.dtype = tf.float32  # dtype has to be float32 for interaction with keras for some reason
        self.g = g  # regularization strength for monotonicity of thresholds
        self.q = q  # L2 shrinkage hyperparameter
        self.build()

    def compute_NLL(self):

        # Initialize weight and bias placeholders
        w0 = np.random.normal(size=(self.n_features, 1))
        # We need to estimate K-1 biases; initialize them in ascending order
        b0 = np.sort(np.random.normal(size=(self.n_classes - 1)))

        self.w = tf.Variable(initial_value=w0, trainable=True, name='weights', dtype=self.dtype)
        self.b = tf.Variable(initial_value=b0, trainable=True, name='biases', dtype=self.dtype)

        def tf_diff(a):
            return a[1:] - a[:-1]

        epsilon = 1e-12  # avoid tf.log(0)

        xW = tf.matmul(self.features, self.w)
        # xW = tf.nn.dropout(xW, keep_prob=1.)
        LL = 0
        LL1 = tf.log(tf.sigmoid(self.b[0] + xW) + epsilon) * \
            tf.to_float(tf.equal(self.y, tf.constant(0., dtype=self.dtype)))

        # This can be written in a more elegant way (vectorize it)
        for i in np.arange(0, self.n_classes - 2, 1):
            LL += tf.multiply(tf.log(tf.sigmoid(self.b[i + 1] + xW) - tf.sigmoid(self.b[i] + xW) + epsilon),
                              tf.to_float(tf.equal(self.y, tf.constant(i + 1, dtype=self.dtype))))

        LL2 = tf.multiply(tf.log(1 - tf.sigmoid(self.b[-1] + xW) + epsilon),
                          tf.to_float(tf.equal(self.y, tf.constant(self.n_classes - 1, dtype=self.dtype))))

        NLL = -1.0 * tf.reduce_mean(LL1 + LL + LL2)

        # Add monotonicity constraints
        diff = tf_diff(self.b)
        c_1 = tf.reduce_sum(-tf.log(diff))

        # Add shrinking constraint (l2 regularization) on the weights
        c_2 = tf.reduce_sum(tf.square(self.w))

        total_loss = NLL + self.g * c_1 + self.q * c_2
        return total_loss, xW

    def build(self):

        self.features = tf.placeholder(
            self.dtype, [None, self.n_features], name='feature_placeholder')
        self.y = tf.placeholder(self.dtype, [None, 1], name='label_placeholder')
        # Construct the NLL that has to be minimized
        self.total_loss, self.xW = self.compute_NLL()
