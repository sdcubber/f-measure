# Extract features from an image dataset with a network pretrained on ImageNet
# Pretrain a network to predict the marginals, extract features and store them

import sys
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# TF - Keras imports
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.applications import VGG16

# Custom modules
import utils.generators as gn
from utils import csv_helpers

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 1. Generate features
batch_size = 32
im_size = 224
epochs = 5
n_labels = 20
debug = False

dataset = sys.argv[1]

csv_path_train = '../data/{}/TRAIN.csv'.format(dataset)
csv_path_validation = '../data/{}/VALIDATION.csv'.format(dataset)
csv_path_test = '../data/{}/TEST.csv'.format(dataset)

df_train = pd.read_csv(csv_path_train).iloc[:]
df_validation = pd.read_csv(csv_path_validation)
df_test = pd.read_csv(csv_path_test).iloc[:]

if debug:
    df_train = df_train.iloc[:100]
    df_validation = df_validation.iloc[:100]
    df_test = df_test.iloc[:100]
    epochs = 2

train_steps = np.ceil(len(df_train) / batch_size)
validation_steps = np.ceil(len(df_validation) / batch_size)
test_steps = np.ceil(len(df_test) / batch_size)

model = VGG16(include_top=False, weights='imagenet',
              input_shape=(im_size, im_size, 3), pooling='max')

# Data generators for inference
train_gen_i = gn.DataGenerator(df=df_train, n_labels=n_labels,
                               im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()
validation_gen_i = gn.DataGenerator(df=df_validation, n_labels=n_labels,
                                    im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()
test_gen_i = gn.DataGenerator(df=df_test, n_labels=n_labels,
                              im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()

features_train = model.predict_generator(train_gen_i, train_steps, verbose=1)
features_validation = model.predict_generator(validation_gen_i, validation_steps, verbose=1)
features_test = model.predict_generator(test_gen_i, test_steps, verbose=1)

# Flatten
features_train = np.array([f.flatten() for f in features_train])
features_validation = np.array([f.flatten() for f in features_validation])
features_test = np.array([f.flatten() for f in features_test])

# Store
if debug:
    np.save('../data/{}/features/features_train_db'.format(dataset), features_train)
    np.save('../data/{}/features/features_validation_db'.format(dataset), features_validation)
    np.save('../data/{}/features/features_test_db'.format(dataset), features_test)

else:
    np.save('../data/{}/features/features_train_max'.format(dataset), features_train)
    np.save('../data/{}/features/features_validation_max'.format(dataset), features_validation)
    np.save('../data/{}/features/features_test_max'.format(dataset), features_test)
