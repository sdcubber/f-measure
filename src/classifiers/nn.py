"""
Neural networks to estimate marginal probabilities and probabilities required for the GFM algorithm
"""

import math
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, Concatenate, Add, Subtract
from keras.layers import BatchNormalization, Dropout, Activation
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Multiply, Dot
from keras.engine.topology import Layer
from keras.models import Model
from keras.applications import VGG16
import keras.backend as K
from keras import metrics


class BR_classifier(object):

    def __init__(self, n_features,  n_labels, c):
        """
        Classifier for multi-label image classification with binary relevance

        Inputs:
        - keep_prob: dropout rate
        - n_labels: int, number of labels
        - max_s: required for GFM. Max size of sy.
        - GFM: GFM mode or not
        """
        self.n_features = n_features
        self.n_labels = n_labels
        self.c = c
        # build model on init
        self.build()

    def build(self):
        # Define input
        self.x = Input(shape=(self.n_features,))
        self.x_drop = Dropout(self.c)(self.x)
        # Output layers: n_classes output nodes for binary relevance
        self.y = Dense(self.n_labels, activation='sigmoid')(self.x_drop)

        self.model = Model(inputs=self.x, outputs=self.y)


class GFM_labelwise_classifier(BR_classifier):
    def __init__(self, n_features, n_labels, c):
        super().__init__(n_features, n_labels, c)

        # Overwrite output layer
        self.x_drop = Dropout(self.c)(self.x)
        self.y = Dense(self.n_labels, activation='softmax')(self.x_drop)
        self.model = Model(inputs=self.x, outputs=self.y)


class GFM_classifier(BR_classifier):

    def __init__(self, n_features, n_labels, max_s, c):
        """
        CNN for multi-label image classification with GFM

        Inputs:
        - n_labels: int, number of labels
        - max_s: maximum number of positive labels for a single instance
        """
        super().__init__(n_features, n_labels, c)
        self.max_s = max_s
        # build model on init

        # Output layers: n_classes output nodes for binary relevance
        # n_labels * max_s output nodes for GFM
        self.y = Dense(self.n_labels * (self.max_s + 1))(self.x_drop)
        # Reshape should be a reshape operation!
        # Custom reshapes: wrap in a Lambda layer
        # See https://github.com/fchollet/keras/issues/6263
        self.y = Reshape((self.n_labels, self.max_s + 1))(self.y)
        self.model = Model(inputs=self.x, outputs=self.y)
