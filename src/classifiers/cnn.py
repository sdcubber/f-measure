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
from keras.layers import BatchNormalization
from keras.engine.topology import Layer
from keras.models import Model
from keras.applications import VGG16
import keras.backend as K
from keras import metrics


class BR_CNN_classifier(object):

    def __init__(self, im_size,  n_labels):
        """
        CNN for multi-label image classification with binary relevance

        Inputs:
        - keep_prob: dropout rate
        - n_labels: int, number of labels
        - max_s: required for GFM. Max size of sy.
        - GFM: GFM mode or not
        """
        self.im_size = im_size
        self.n_labels = n_labels
        self.dropout_rate = 0.50
        self.n_neurons = 128  # Number of neurons in dense layers
        # build model on init
        self.build()

    def build(self):
        # Define input
        self.x = Input(shape=(self.im_size, self.im_size, 3))

        # Convolutional layers
        conv_1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(self.x)
        conv_1 = MaxPooling2D(padding='same')(conv_1)
        conv_2 = Conv2D(32, kernel_size=(3, 3),
                        padding='same', activation='relu')(conv_1)
        conv_2 = MaxPooling2D(padding='same')(conv_2)

        # Flatten
        conv_flat = Flatten()(conv_2)
        # Fully connected layers
        fc_1 = Dense(self.n_neurons, activation='relu')(conv_flat)
        fc_1 = Dropout(self.dropout_rate)(fc_1)
        fc_2 = Dense(self.n_neurons, activation='relu')(fc_1)
        self.fc_2 = Dropout(self.dropout_rate)(fc_2)

        # Output layers: n_classes output nodes for binary relevance
        self.y = Dense(self.n_labels, activation='sigmoid')(self.fc_2)

        self.model = Model(inputs=self.x, outputs=self.y)


class GFM_CNN_classifier(BR_CNN_classifier):

    def __init__(self, im_size, n_labels, max_s):
        """
        CNN for multi-label image classification with GFM

        Inputs:
        - n_labels: int, number of labels
        - max_s: maximum number of positive labels for a single instance
        """
        super().__init__(im_size, n_labels)
        self.max_s = max_s
        # build model on init
        self.build()

        # Output layers: n_classes output nodes for binary relevance
        # n_labels * max_s output nodes for GFM
        self.y = Dense(self.n_labels * (self.max_s + 1))(self.fc_2)
        # Reshape should be a reshape operation!
        # Custom reshapes: wrap in a Lambda layer
        # See https://github.com/fchollet/keras/issues/6263
        self.y = Reshape((self.n_labels, self.max_s + 1))(self.y)
        self.model = Model(inputs=self.x, outputs=self.y)


class VGG_classifier(object):
    def __init__(self, im_size, n_labels, n_neurons, imagenet):
        self.im_size = im_size
        self.n_labels = n_labels
        self.n_channels = 3
        self.n_neurons = n_neurons
        self.imagenet = imagenet
        self.build()

    def build(self):
        # Define input
        self.x = Input(shape=(self.im_size, self.im_size, 3))

        # Pretrained VGG16
        weights = None
        if self.imagenet:
            weights = 'imagenet'
        VGGmodel = VGG16(include_top=False, weights=weights,
                         input_tensor=self.x,
                         input_shape=(self.im_size, self.im_size, self.n_channels),
                         pooling='max')

        # VGG_out = Flatten()(VGGmodel.output) # in case of no pooling
        VGG_out = VGGmodel.output
        VGG_out = Dropout(0.4)(VGG_out)
        VGG_out = BatchNormalization()(VGG_out)

        # batchnorm + dense layers
        fc_1 = Dense(self.n_neurons, activation='relu')(VGG_out)
        self.fc_1 = Dropout(0.4)(fc_1)
        fc_2 = Dense(self.n_neurons, activation='relu')(self.fc_1)
        self.fc_2 = Dropout(0.4)(fc_2)
        self.y = Dense(self.n_labels, activation='sigmoid')(self.fc_2)
        self.model = Model(inputs=self.x, outputs=self.y)


class GFM_VGG_classifier(VGG_classifier):
    """GFM classifier on top of VGG16"""

    def __init__(self, im_size, n_labels, n_neurons, imagenet, max_s):
        super().__init__(im_size, n_labels, n_neurons, imagenet)
        self.max_s = max_s
        self.build()
        # Overwite final layer
        self.y = Dense(self.n_labels * (self.max_s + 1))(self.fc_2)
        self.y = Reshape((self.n_labels, self.max_s + 1))(self.y)
        self.model = Model(inputs=self.x, outputs=self.y)
