import os
import sys
import ast
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import fbeta_score
# Data augmentation tools
from utils.image_ml_ext import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
# Keras imports
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

# References:
# https://stanford.edu/~shervine/blog/keras-generator-multiprocessing.html
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


def randomHorizontalFlip(image, p=0.5):
    """Do a random horizontal flip with probability p"""
    if np.random.random() < p:
        image = np.fliplr(image)
    return image


def randomVerticalFlip(image, p=0.5):
    """Do a random vertical flip with probability p"""
    if np.random.random() < p:
        image = np.flipud(image)
    return image


class DataGenerator(object):
    """Custom generator"""

    def __init__(self, df, n_labels, im_size, batch_size, shuffle, mode='train', pretrained=False, features=None, augmentation=False):
        self.df = df  # path to csv file that contains mapping from filenames to GFM labels
        self.im_size = im_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.n_labels = n_labels
        self.pretrained = pretrained  # use pretrained weights
        self.features = features
        self.augmentation = augmentation  # use data augmentation

    def get_instance_indexes(self):
        indexes = list(self.df.index)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def get_batch_features(self, indexes):
        # Empty containers for features
        if self.pretrained:
            self.n_features = self.features.shape[1]

            batch_features = np.zeros((len(indexes), self.n_features))
            for i, ix in enumerate(indexes):
                batch_features[i] = self.features[ix, :]
            return batch_features

        else:
            batch_features = np.zeros((len(indexes), self.im_size, self.im_size, 3))

            # Fill up container
            for i, ix in enumerate(indexes):

                im = load_img(self.df['full_path'][ix], target_size=(self.im_size, self.im_size))
                im = img_to_array(im)
                im = preprocess_input(im)

                if self.augmentation:
                    #im = random_rotation(im, rg=15)
                    #im = random_shift(im, 0.05, 0.05)
                    im = randomHorizontalFlip(im)
                    im = randomVerticalFlip(im)
                    # im = random_zoom(im, (0.1, 0.1))
                    # im = random_shear(im, 0.1)
                    # im = random_channel_shift(im, 0.1)

                batch_features[i] = im

            return batch_features

    def get_batch_labels(self, indexes):
        # Empty containers for labels
        batch_labels = np.zeros((len(indexes), self.n_labels))

        if self.mode == 'test':
            return None
        else:
            # Fill up container
            for i, ix in enumerate(indexes):
                batch_labels[i] = np.array(ast.literal_eval(
                    self.df['marginal_labels'][ix]), dtype=int)
            return batch_labels

    def generate(self):
        while True:
            indexes = self.get_instance_indexes()
            num_batches = int(np.ceil(len(self.df) / self.batch_size))
            for i in range(num_batches):
                if i == (num_batches - 1):
                    batch_indexes = indexes[i * self.batch_size:]
                else:
                    batch_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                X = self.get_batch_features(batch_indexes)
                y = self.get_batch_labels(batch_indexes)
                yield (X, y)


class DataGenerator_gfm_MC(DataGenerator):

    """Override get_batch_labels to yield arrays required for multiclass GFM"""

    def __init__(self, df, n_labels, im_size, batch_size, shuffle, mode, pretrained, max_s, features=None, augmentation=False):
        super().__init__(df, n_labels, im_size, batch_size, shuffle, mode, pretrained, features)
        self.max_s = max_s
        self.pretrained = pretrained

    def get_batch_labels(self, indexes):
        batch_labels = np.zeros((len(indexes), self.n_labels, self.max_s + 1))

        if self.mode == 'test':
            return None

        else:
            # Fill up container
            # For multiclass classification: return as matrix and include first column (P(label not present)). Each row corresponds to a label
            for i, ix in enumerate(indexes):
                batch_labels[i] = np.array(ast.literal_eval(
                    self.df['gfm_labels'][ix]), dtype=int)
            return batch_labels
