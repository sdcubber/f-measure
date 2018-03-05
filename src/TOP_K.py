"""
Perform GFM MLC with top k labels from Planet data
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# TF - Keras imports
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

# Custom modules
import utils.generators as gn
from utils import csv_helpers
import classifiers.thresholding as th
from classifiers.F_score import compute_F_score
from classifiers.ye_et_al import QuadraticTimeAlgorithm
from sklearn.preprocessing import OneHotEncoder
from classifiers.nn import GFM_classifier
from classifiers.gfm import GeneralFMaximizer, complete_matrix_columns_with_zeros

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset = sys.argv[1]
k = int(sys.argv[2])  # no of labels
pretrained = True

csv_path_train = '../data/{}/TRAIN.csv'.format(dataset)
csv_path_validation = '../data/{}/VALIDATION.csv'.format(dataset)
csv_path_test = '../data/{}/TEST.csv'.format(dataset)

df_train = pd.read_csv(csv_path_train)
df_validation = pd.read_csv(csv_path_validation)
df_test = pd.read_csv(csv_path_test)

# Extract ground truth labels
y_true_train = np.array([np.array(ast.literal_eval(l)) for l in df_train['marginal_labels']])
y_true_validation = np.array([np.array(ast.literal_eval(l))
                              for l in df_validation['marginal_labels']])
y_true_test = np.array([np.array(ast.literal_eval(l))
                        for l in df_test['marginal_labels']])

# Load the predicted marginals
y_predicted_train = np.load(
    '../results/BR_predictions_train_{}_pt{}.npy'.format(dataset, int(pretrained)))
y_predicted_validation = np.load(
    '../results/BR_predictions_validation_{}_pt{}.npy'.format(dataset, int(pretrained)))
y_predicted_test = np.load(
    '../results/BR_predictions_test_{}_pt{}.npy'.format(dataset, int(pretrained)))

# Load pretrained features
features_train = np.load('../data/{}/features/features_train_max.npy'.format(dataset))
features_validation = np.load('../data/{}/features/features_validation_max.npy'.format(dataset))
features_test = np.load('../data/{}/features/features_test_max.npy'.format(dataset))
n_features = features_train.shape[1]

# rescale
from sklearn.preprocessing import StandardScaler
featurescaler = StandardScaler().fit(features_train)

features_train = featurescaler.transform(features_train)
features_validation = featurescaler.transform(features_validation)
features_test = featurescaler.transform(features_test)

# Sort on most frequently occuring labels
label_order = np.argsort(np.sum(y_true_train, axis=0))[::-1]

y_true_train_sorted = y_true_train[:, label_order]
y_true_validation_sorted = y_true_validation[:, label_order]
y_true_test_sorted = y_true_test[:, label_order]

y_predicted_train_sorted = y_predicted_train[:, label_order]
y_predicted_validation_sorted = y_predicted_validation[:, label_order]
y_predicted_test_sorted = y_predicted_test[:, label_order]

# Select top k labels, run GFM_MLC


def labels_to_matrix_Y(y):
    """Convert binary label matrix to a matrix Y that is suitable to estimate P(y,s):
    Each entry of the matrix Y_ij is equal to I(y_ij == 1)*np.sum(yi)"""
    row_sums = np.sum(y, axis=1)
    Y = np.multiply(y, np.broadcast_to(row_sums.reshape(-1, 1), y.shape)).astype(int)
    return(Y)


def labelmatrix_to_GFM_matrix(labelmatrix, max_s):
    n_labels = labelmatrix.shape[1]

    multiclass_matrix = labels_to_matrix_Y(labelmatrix)

    encoder = OneHotEncoder(sparse=False)
    outputs_per_label = []
    enc = encoder.fit(np.arange(0, max_s + 1).reshape(-1, 1))
    for i in range(n_labels):
        label_i = enc.transform(multiclass_matrix[:, i].reshape(-1, 1))
        outputs_per_label.append(label_i)

        outputs_per_field = []

    return np.array(outputs_per_label).transpose(1, 0, 2)


y_true_train_selection = y_true_train_sorted[:, :k]
y_true_validation_selection = y_true_validation_sorted[:, :k]
y_true_test_selection = y_true_test_sorted[:, :k]

y_predicted_train_selection = y_predicted_train_sorted[:, :k]
y_predicted_validation_selection = y_predicted_validation_sorted[:, :k]
y_predicted_test_selection = y_predicted_test_sorted[:, :k]

max_s = np.max(np.array([np.max(np.sum(y_true_train_selection, axis=1)),
                         np.max(np.sum(y_true_validation_selection, axis=1)),
                         np.max(np.sum(y_true_test_selection, axis=1))]))
y_gfm_train_selection = labelmatrix_to_GFM_matrix(y_true_train_selection, max_s)
y_gfm_validation_selection = labelmatrix_to_GFM_matrix(y_true_validation_selection, max_s)
y_gfm_test_selection = labelmatrix_to_GFM_matrix(y_true_test_selection, max_s)

n_labels = k
model = GFM_classifier(n_features, n_labels, max_s, c=0.2).model
optimizer = Adam(lr=1e-4)
batch_size = 32
epochs = 200  # Early stopping on validation data

# Compile with specific loss function


def GFM_loss(y_true, y_pred):
    """Custom loss function for the joint estimation of the required parameters for GFM.
    The combined loss is the row-wise sum of categorical losses over the rows of the matrix P
    Where each row corresponds to one label.

    """
    loss = K.constant(0, tf.float32)
    for i in range(n_labels):
        loss += K.categorical_crossentropy(target=y_true[:, i, :],
                                           output=y_pred[:, i, :], from_logits=True)
    return loss


model.compile(loss=GFM_loss, optimizer=optimizer)

print(model.summary())
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0,
                  patience=3, verbose=1, mode='auto')
]

model.fit(x=features_train, y=y_gfm_train_selection,
          batch_size=batch_size, epochs=epochs,
          verbose=1, callbacks=callbacks, validation_data=(features_validation, y_gfm_validation_selection))

# Make predictions
pis_train = model.predict(features_train, verbose=1)
pis_validation = model.predict(features_validation, verbose=1)
pis_test = model.predict(features_test, verbose=1)


def softmax(v):
    """softmax a vector"""
    return(np.exp(v) / np.sum(np.exp(v)))


print('Softmaxing...')
pis_train = np.apply_along_axis(softmax, 2, pis_train)
pis_validation = np.apply_along_axis(softmax, 2, pis_validation)
pis_test = np.apply_along_axis(softmax, 2, pis_test)

print('Filling...')
pis_train_filled = [complete_matrix_columns_with_zeros(
    mat[:, 1:], len=n_labels) for mat in tqdm(pis_train)]
pis_validation_filled = [complete_matrix_columns_with_zeros(
    mat[:, 1:], len=n_labels) for mat in tqdm(pis_validation)]
pis_test_filled = [complete_matrix_columns_with_zeros(
    mat[:, 1:], len=n_labels) for mat in tqdm(pis_test)]
print('Done!')

# Store these
np.save('../matP/{}_top_{}_P_train.npy'.format(dataset, k), pis_train_filled)
np.save('../matP/{}_top_{}_P_validation.npy'.format(dataset, k), pis_validation_filled)
np.save('../matP/{}_top_{}_P_test.npy'.format(dataset, k), pis_test_filled)
