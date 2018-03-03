"""
GFM with multiclass classification for multilabel classification tasks
Both F1 and F2 scores are calculated
"""

import os
import ast
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# TF - Keras imports
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Custom modules
import utils.generators as gn
from utils import csv_helpers
from classifiers.nn import GFM_labelwise_classifier
from classifiers.F_score import compute_F_score
from classifiers.gfm import GeneralFMaximizer, complete_matrix_columns_with_zeros
# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def GFM_MLC(args):

    # Parameters
    batch_size = 32
    dataset = args.dataset
    epochs = 1000  # early stopping on validation data
    verbosity = 0
    sklearn = False
    c = args.c
    print('Amount of regularization= {:.3f}'.format(c))

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

    csv_path_train = '../data/{}/TRAIN.csv'.format(dataset)
    csv_path_validation = '../data/{}/VALIDATION.csv'.format(dataset)
    csv_path_test = '../data/{}/TEST.csv'.format(dataset)

    df_train = pd.read_csv(csv_path_train)
    df_validation = pd.read_csv(csv_path_validation)
    df_test = pd.read_csv(csv_path_test)

    train_steps = np.ceil(len(df_train) / batch_size)
    validation_steps = np.ceil(len(df_validation) / batch_size)
    test_steps = np.ceil(len(df_test) / batch_size)

    # Extract ground truth labels
    y_true_train = np.array([ast.literal_eval(df_train['marginal_labels'][i])
                             for i in range(len(df_train))])
    y_true_validation = np.array([ast.literal_eval(df_validation['marginal_labels'][i])
                                  for i in range(len(df_validation))])
    y_true_test = np.array([ast.literal_eval(df_test['marginal_labels'][i])
                            for i in range(len(df_test))])

    n_labels = y_true_train.shape[1]

    y_gfm_train = np.array([ast.literal_eval(df_train['gfm_labels'][i])
                            for i in range(len(df_train))])
    y_gfm_validation = np.array([ast.literal_eval(df_validation['gfm_labels'][i])
                                 for i in range(len(df_validation))])

    # Compute max_s: the maximum number of positive label for a single instance
    max_s = np.max(np.array([np.max(np.sum(y_true_train, axis=1)),
                             np.max(np.sum(y_true_validation, axis=1)),
                             np.max(np.sum(y_true_test, axis=1))]))

    # Containers
    GFM_train_entries = []
    GFM_validation_entries = []
    GFM_test_entries = []

    for label in range(n_labels):
        # print('Label {} of {}...'.format(label, n_labels))
        # extract one multinomial regression problem
        if sklearn:
            y_label_train = np.argmax(y_gfm_train[:, label, :], axis=1)
            y_label_validation = np.argmax(y_gfm_validation[:, label, :], axis=1)
        else:
            y_label_train = y_gfm_train[:, label, :]
            y_label_validation = y_gfm_validation[:, label, :]
        # print(y_label_train.shape)

        if sklearn:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(multi_class='ovr', solver='lbfgs', C=c)

        else:
            model = GFM_labelwise_classifier(n_features, max_s + 1, c).model
            optimizer = Adam()
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            callbacks = [
                EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=3, verbose=verbosity, mode='auto'),
                ModelCheckpoint('../models/GFMMLC_labelwise_{}.h5'.format(dataset),
                                monitor='val_loss', save_best_only=True, verbose=verbosity)
            ]

            model.fit(x=features_train, y=y_label_train,
                      batch_size=batch_size, epochs=epochs,
                      verbose=verbosity, callbacks=callbacks, validation_data=(features_validation, y_label_validation))
            # Load best model
            model.load_weights('../models/GFMMLC_labelwise_{}.h5'.format(dataset))
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        pis_train = model.predict(features_train)
        pis_validation = model.predict(features_validation)
        pis_test = model.predict(features_test)

        if sklearn:
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder()
            enc.fit(np.argmax(np.argmax(y_gfm_train[:, :, :], axis=1), axis=1).reshape(-1, 1))
            pis_train = enc.transform(pis_train.reshape(-1, 1)).toarray()
            pis_validation = enc.transform(pis_validation.reshape(-1, 1)).toarray()
            pis_test = enc.transform(pis_test.reshape(-1, 1)).toarray()

        GFM_train_entries.append(pis_train)
        GFM_validation_entries.append(pis_validation)
        GFM_test_entries.append(pis_test)

    # Combine all the predictonis
    pis_train = np.stack(GFM_train_entries).transpose(1, 0, 2)
    pis_validation = np.stack(GFM_validation_entries).transpose(1, 0, 2)
    pis_test = np.stack(GFM_test_entries).transpose(1, 0, 2)

    pis_train_final = [complete_matrix_columns_with_zeros(
        mat[:, 1:], len=n_labels) for mat in pis_train]
    pis_validation_final = [complete_matrix_columns_with_zeros(
        mat[:, 1:], len=n_labels) for mat in pis_validation]
    pis_test_final = [complete_matrix_columns_with_zeros(
        mat[:, 1:], len=n_labels) for mat in pis_test]

    # Compute optimal predictions for F1
    for beta in [1, 2]:
        GFM = GeneralFMaximizer(beta, n_labels)

        # Run GFM algo on this output
        (optimal_predictions_train, E_F_train) = GFM.get_predictions(predictions=pis_train_final)
        (optimal_predictions_validation, E_F_validation) = GFM.get_predictions(
            predictions=pis_validation_final)
        (optimal_predictions_test, E_F_test) = GFM.get_predictions(predictions=pis_test_final)

        # Evaluate F score
        F_train = compute_F_score(y_true_train, optimal_predictions_train, t=0.5, beta=beta)
        F_validation = compute_F_score(
            y_true_validation, optimal_predictions_validation, t=0.5, beta=beta)
        F_test = compute_F_score(y_true_test, optimal_predictions_test, t=0.5, beta=beta)

        print('GFM_MLC ({})'.format(dataset))
        print('-' * 50)
        # print('F{} score on training data: {:.4f}'.format(beta, F_train))
        # print('F{} score on validation data: {:.4f}'.format(beta, F_validation))
        print('F{} score on test data: {:.4f}'.format(beta, F_test))

        # Store test set predictions to submit to Kaggle
        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(
                os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in optimal_predictions_test]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_GFM_MC_labelwise')


def main():
    parser = argparse.ArgumentParser(
        description='GFM_MLC')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('c', type=float, help='amount of regularization')
    args = parser.parse_args()
    GFM_MLC(args)


if __name__ == "__main__":
    sys.exit(main())
    # main()
