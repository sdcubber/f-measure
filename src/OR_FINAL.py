"""
Implementation of logistic ordinal regression (aka proportional odds) model
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

import os
import ast
import time
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from sklearn import datasets, metrics
from tqdm import tqdm
from classifiers.F_score import compute_F_score
from classifiers.gfm import GeneralFMaximizer, complete_matrix_columns_with_zeros
from classifiers.OR import ProportionalOdds_TF
from utils import csv_helpers
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, Concatenate, Add, Subtract
from keras.layers import BatchNormalization, Dropout, Activation
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Multiply, Dot
from keras.engine.topology import Layer
from keras.models import Model
from tensorflow.contrib.opt import ScipyOptimizerInterface  # constrained optimization
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def OR(args, logger):
    dataset = args.dataset
    g = args.g
    q = args.q
    pretrained = True

    csv_path_train = '../data/{}/TRAIN.csv'.format(dataset)
    csv_path_validation = '../data/{}/VALIDATION.csv'.format(dataset)
    csv_path_test = '../data/{}/TEST.csv'.format(dataset)

    df_train = pd.read_csv(csv_path_train)
    df_validation = pd.read_csv(csv_path_validation)
    df_test = pd.read_csv(csv_path_test)

    features_train = np.load('../data/{}/features/features_train_max.npy'.format(dataset))
    features_validation = np.load('../data/{}/features/features_validation_max.npy'.format(dataset))
    features_test = np.load('../data/{}/features/features_test_max.npy'.format(dataset))

    # rescale
    from sklearn.preprocessing import StandardScaler
    featurescaler = StandardScaler().fit(features_train)

    features_train = featurescaler.transform(features_train)
    features_validation = featurescaler.transform(features_validation)
    features_test = featurescaler.transform(features_test)

    y_gfm_train = np.array([ast.literal_eval(df_train['gfm_labels'][i])
                            for i in range(len(df_train))])
    y_gfm_validation = np.array([ast.literal_eval(df_validation['gfm_labels'][i])
                                 for i in range(len(df_validation))])

    # Extract ground truth labels to compute F scores
    y_true_train = np.array([ast.literal_eval(df_train['marginal_labels'][i])
                             for i in range(len(df_train))])
    y_true_validation = np.array([ast.literal_eval(df_validation['marginal_labels'][i])
                                  for i in range(len(df_validation))])
    y_true_test = np.array([ast.literal_eval(df_test['marginal_labels'][i])
                            for i in range(len(df_test))])
    n_labels = y_true_train.shape[1]

    # Load the predicted marginals from BR method and replace the true test labels with them
    predicted_marginals_train = np.load(
        '../results/BR_predictions_train_{}_pt{}.npy'.format(dataset, int(pretrained)))
    predicted_marginals_validation = np.load(
        '../results/BR_predictions_validation_{}_pt{}.npy'.format(dataset, int(pretrained)))
    predicted_marginals_test = np.load(
        '../results/BR_predictions_test_{}_pt{}.npy'.format(dataset, int(pretrained)))

    # Containers
    GFM_train_entries = []
    GFM_validation_entries = []
    GFM_test_entries = []

    for label in range(n_labels):
        print('Label {} of {}...'.format(label, n_labels))
        # Extract one ordinal regression problem
        boolean_index_train = y_gfm_train[:, label, 0] == 0
        boolean_index_valid = y_gfm_validation[:, label, 0] == 0

        # we don't need the first row (P(y=0))
        y_dummies_train = y_gfm_train[boolean_index_train, label, 1:]
        y_dummies_validation = y_gfm_validation[boolean_index_valid, label, 1:]

        # We need to transform the labels such that they start from zero
        # And such that each class occurs at least once in the dataset (to avoid errors when minimizing the NLL)
        # A backtransform will be required later on

        y_train = np.argmax(y_dummies_train, axis=1)
        y_validation = np.argmax(y_dummies_validation, axis=1)

        y_train_transformed = y_train - y_train.min()
        y_validation_transformed = y_validation - y_train.min()

        x_train = features_train[boolean_index_train, :]
        x_validation = features_validation[boolean_index_valid, :]

        n_classes = len(np.unique(y_train))
        n_features = x_train.shape[1]

        # register TF session with keras
        sess = tf.Session()
        K.set_session(sess)

        # fetch NLL
        proportialoddsmodel = ProportionalOdds_TF(n_features, n_classes, g, q)
        features, y = proportialoddsmodel.features, proportialoddsmodel.y  # placeholders
        total_loss = proportialoddsmodel.total_loss  # loss
        b, w, xW = proportialoddsmodel.b, proportialoddsmodel.w, proportialoddsmodel.xW  # weights and biases

        train_step = tf.train.AdamOptimizer().minimize(total_loss)
        validation_loss_hist = []
        train_loss_hist = []
        patience_counter = 0
        patience = 3
        min_delta = 1e-4
        epochs = 2000  # Early stopping on validation data
        batch_size = 32

        def get_batch(x, y, i):
            n_steps = int(np.ceil(len(y) / batch_size))
            if i == (n_steps - 1):
                batch_x = x[i * batch_size:]
                batch_y = y[i * batch_size:]
                return batch_x, batch_y
            else:
                batch_x = x[i * batch_size:(i + 1) * batch_size]
                batch_y = y[i * batch_size:(i + 1) * batch_size]
                return batch_x, batch_y

        train_steps = int(np.ceil(len(y_train) / batch_size))
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for i in range(epochs + 1):
                # shuffle x and y at beginning of epoch
                x_train_shuffle, y_train_shuffle = shuffle(x_train, y_train_transformed)
                for j in range(train_steps):

                    batch_x, batch_y = get_batch(x_train_shuffle, y_train_shuffle, j)
                    sess.run(train_step, feed_dict={features: batch_x,
                                                    y: batch_y.reshape(-1, 1)})

                train_loss = sess.run(total_loss, feed_dict={features: x_train,
                                                             y: y_train_transformed.reshape(-1, 1)})
                validation_loss = sess.run(total_loss, feed_dict={features: x_validation,
                                                                  y: y_validation_transformed.reshape(-1, 1)})

                train_loss_hist.append(train_loss)
                validation_loss_hist.append(validation_loss)
                # print('Epoch {} - Training loss {:.3f} - Validation loss {:.3f}'.format(i,
            #                                                                            train_loss, validation_loss))

                if np.isnan(train_loss):
                    logger.log('NaN loss!')
                    b_current = b.eval()
                    w_current = w.eval()
                    encoding_current = xW.eval(feed_dict={features: x_train})
                    print('Current biases : {}'.format(b.eval()))
                    print('Current weights: {}'.format(w.eval()))
                    break

                # Control flow for early stopping
                if validation_loss_hist[i] - np.min(validation_loss_hist) > min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter == patience or i == epochs:
                    print('Early stopping... ({} epochs)'.format(i))
                    print('Optimal biases : {}'.format(b.eval()))
                    break

            # Make predictions for the subset of data that was used to train the OR model
            b_opt = b.eval()
            encoding_train = xW.eval(feed_dict={features: x_train})
            encoding_validation = xW.eval(feed_dict={features: x_validation})
            preds_train = np.sum(encoding_train <= b_opt, axis=1)
            preds_validation = np.sum(encoding_validation <= b_opt, axis=1)
            acc_train = metrics.accuracy_score(y_train_transformed, preds_train)
            acc_validation = metrics.accuracy_score(y_validation_transformed, preds_validation)
            print('Training set accuracy: {:.3f}'.format(acc_train))
            print('Validation set accuracy: {:.3f}'.format(acc_validation))

            # Finally, make predictions for all instances (we don't know where the exact marginals at test time)
            encoding_train_full = xW.eval(feed_dict={features: features_train})
            encoding_validation_full = xW.eval(
                feed_dict={features: features_validation})
            encoding_test_full = xW.eval(feed_dict={features: features_test})

        tf.reset_default_graph()

        # Go to probability estimates

        def sigmoid(v):
            return 1. / (1. + np.exp(-v))

        def or_to_probabilities(encoding, biases):
            return sigmoid(encoding + np.hstack([-np.inf, biases, np.inf])[1:]) - sigmoid(encoding + np.hstack([-np.inf, biases, np.inf])[:-1])

        conditionals_train = or_to_probabilities(encoding_train_full, b_opt)
        conditionals_validation = or_to_probabilities(encoding_validation_full, b_opt)
        conditionals_test = or_to_probabilities(encoding_test_full, b_opt)

        # Add columns of zeros for the classes that were not present
        index = [i not in np.unique(y_train_transformed + y_train.min())
                 for i in np.arange(y_dummies_train.shape[1])]
        missing = list(np.sort(np.arange(y_dummies_train.shape[1])[np.array(index)]))

        for m in missing:  # Has to be done in a loop to be correct
            conditionals_train = np.insert(conditionals_train, m, 0, axis=1)
            conditionals_validation = np.insert(conditionals_validation, m, 0, axis=1)
            conditionals_test = np.insert(conditionals_test, m, 0, axis=1)

        # Multiply them with the marginals
        probabilities_train = conditionals_train * \
            predicted_marginals_train[:, label].reshape(-1, 1)
        probabilities_validation = conditionals_validation * \
            predicted_marginals_validation[:, label].reshape(-1, 1)
        probabilities_test = conditionals_test * predicted_marginals_test[:, label].reshape(-1, 1)

        GFM_train_entries.append(probabilities_train)
        GFM_validation_entries.append(probabilities_validation)
        GFM_test_entries.append(probabilities_test)

    GFM_train_entries = np.stack(GFM_train_entries).transpose(1, 0, 2)
    GFM_validation_entries = np.stack(GFM_validation_entries).transpose(1, 0, 2)
    GFM_test_entries = np.stack(GFM_test_entries).transpose(1, 0, 2)

    # Store GFM train entries for debugging
    import pickle
    pickle.dump(GFM_train_entries, open('../notebooks/GFM_train_entries_original.p', 'wb'))

    # Fill them
    train_predictions_filled = [complete_matrix_columns_with_zeros(
        mat[:, :], len=n_labels) for mat in tqdm(GFM_train_entries)]
    validation_predictions_filled = [complete_matrix_columns_with_zeros(
        mat[:, :], len=n_labels) for mat in tqdm(GFM_validation_entries)]
    test_predictions_filled = [complete_matrix_columns_with_zeros(
        mat[:, :], len=n_labels) for mat in tqdm(GFM_test_entries)]

    # Run GFM for F1 and F2

    for beta in [1, 2]:
        GFM = GeneralFMaximizer(beta, n_labels)

        # Run GFM algo on this output
        (optimal_predictions_train, E_F_train) = GFM.get_predictions(
            predictions=train_predictions_filled)
        (optimal_predictions_validation, E_F_validation) = GFM.get_predictions(
            predictions=validation_predictions_filled)
        (optimal_predictions_test, E_F_test) = GFM.get_predictions(predictions=test_predictions_filled)

        # Evaluate F score
        F_train = compute_F_score(y_true_train, optimal_predictions_train, t=0.5, beta=beta)
        F_validation = compute_F_score(
            y_true_validation, optimal_predictions_validation, t=0.5, beta=beta)
        F_test = compute_F_score(y_true_test, optimal_predictions_test, t=0.5, beta=beta)

        logger.log('GFM_OR ({})'.format(dataset))
        logger.log('-' * 50)
        logger.log('F{} score on training data: {:.4f}'.format(beta, F_train))
        logger.log('F{} score on validation data: {:.4f}'.format(beta, F_validation))
        logger.log('F{} score on test data: {:.4f}'.format(beta, F_test))

        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in optimal_predictions_test]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_OR_{}'.format(int(pretrained)))


import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='OR')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('g', type=float, help='monotonicity constraint hyperparam', default=0.1)
    parser.add_argument('q', type=float, help='L2 shrinkage hyperparam', default=0.)
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='Use pretrained VGG features')
    parser.add_argument('-name', type=str, help='name of experiment', default='OR_FINAL')

    args = parser.parse_args()
    args.name = '_'.join((args.name, args.dataset))

    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    logger = loggerClass(args, timestamp)
    OR(args, logger)


if __name__ == "__main__":
    sys.exit(main())
    # main()
