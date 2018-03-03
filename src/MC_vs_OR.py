"""
Compare performance of GFM with multiclass versus GFM with ordinal regression for different fractions of dataset used
"""
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

# Custom modules
import utils.generators as gn
from utils import csv_helpers
import classifiers.thresholding as th
from classifiers.F_score import compute_F_score
from classifiers.ye_et_al import QuadraticTimeAlgorithm
from sklearn.preprocessing import OneHotEncoder
from classifiers.nn import BR_classifier, GFM_classifier
from classifiers.gfm import GeneralFMaximizer, complete_matrix_columns_with_zeros
from classifiers.OR import ProportionalOdds_TF
from sklearn.utils import shuffle
from sklearn import datasets, metrics

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

dataset = sys.argv[1]
beta = sys.argv[2]

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

n_labels = y_true_train.shape[1]

# Load pretrained features
features_train = np.load('../data/{}/features/features_train_max.npy'.format(dataset))
features_validation = np.load('../data/{}/features/features_validation_max.npy'.format(dataset))
features_test = np.load('../data/{}/features/features_test_max.npy'.format(dataset))
# rescale
from sklearn.preprocessing import StandardScaler
featurescaler = StandardScaler().fit(features_train)
features_train = featurescaler.transform(features_train)
features_validation = featurescaler.transform(features_validation)
features_test = featurescaler.transform(features_test)
n_features = features_train.shape[1]

# rescale
from sklearn.preprocessing import StandardScaler
featurescaler = StandardScaler().fit(features_train)

features_train = featurescaler.transform(features_train)
features_validation = featurescaler.transform(features_validation)
features_test = featurescaler.transform(features_test)

# Select a fraction of the data (not stratified, so rare labels will be missing)


def get_fraction_of_training_data(fraction):
    indices = np.array(df_train.sample(int(fraction * len(df_train))).index)
    df_train_f = df_train.iloc[indices, :].reset_index(drop=True)
    y_true_train_f = y_true_train[indices]
    features_train_f = features_train[indices]

    return df_train_f, y_true_train_f, features_train_f


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


# Containers to store scores
F1_train_MC, F1_val_MC, F1_test_MC = [], [], []
F1_train_OR, F1_val_OR, F1_test_OR = [], [], []

for f in tqdm(np.arange(0.1, 1, 0.1)):
    print('Data fraction: {}'.format(f))
    # Leave validation data intact
    df_train_f, y_true_train_f, features_train_f = get_fraction_of_training_data(
        f)

    print('BR...')
    # A. Estimate the marginals
    model = BR_classifier(n_features, n_labels, 0.2).model
    optimizer = Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0,
                      patience=3, verbose=1, mode='auto')
    ]
    model.fit(x=features_train_f, y=y_true_train_f,
              batch_size=32, epochs=200,
              verbose=0, callbacks=callbacks, validation_data=(features_validation, y_true_validation))

    predicted_marginals_train_f = model.predict(features_train_f, verbose=0)
    predicted_marginals_validation = model.predict(features_validation, verbose=0)
    predicted_marginals_test = model.predict(features_test, verbose=0)

    print('GFM...')
    # B. Do GFM MC
    max_s = np.max(np.array([np.max(np.sum(y_true_train_f, axis=1)),
                             np.max(np.sum(y_true_validation, axis=1)),
                             np.max(np.sum(y_true_test, axis=1))]))

    y_gfm_train_f = labelmatrix_to_GFM_matrix(y_true_train_f, max_s)
    y_gfm_validation = labelmatrix_to_GFM_matrix(y_true_validation, max_s)
    y_gfm_test = labelmatrix_to_GFM_matrix(y_true_test, max_s)

    model = GFM_classifier(n_features, n_labels, max_s, 0.2).model
    optimizer = Adam()
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

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0,
                      patience=3, verbose=1, mode='auto')
    ]

    model.fit(x=features_train_f, y=y_gfm_train_f,
              batch_size=batch_size, epochs=epochs,
              verbose=0, callbacks=callbacks, validation_data=(features_validation, y_gfm_validation))

    # Make predictions
    pis_train = model.predict(features_train_f, verbose=1)
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

    # Compute optimal predictions for F1
    GFM = GeneralFMaximizer(beta, n_labels)

    # Run GFM algo on this output
    (optimal_predictions_train, E_F_train) = GFM.get_predictions(predictions=pis_train_filled)
    (optimal_predictions_validation, E_F_validation) = GFM.get_predictions(
        predictions=pis_validation_filled)

    # Evaluate F score
    F_GFM_train = compute_F_score(y_true_train_f,
                                  optimal_predictions_train, t=0.5, beta=beta)
    F_GFM_validation = compute_F_score(
        y_true_validation, optimal_predictions_validation, t=0.5, beta=beta)

    F_GFM_test = 0
    if dataset != 'KAGGLE_PLANET':
        (optimal_predictions_test, E_F_test) = GFM.get_predictions(predictions=pis_test_filled)
        F_GFM_test = compute_F_score(
            y_true_test, optimal_predictions_test, t=0.5, beta=beta)

    if beta == 1:
        F1_train_MC.append(F_GFM_train)
        F1_val_MC.append(F_GFM_validation)
        F1_test_MC.append(F_GFM_test)

    # C. Do OR
    print('OR...')
    # Containers
    GFM_train_entries = []
    GFM_validation_entries = []
    GFM_test_entries = []

    for label in range(n_labels):
        print('Label {} of {}...'.format(label, n_labels))
        # Extract one ordinal regression problem
        boolean_index_train = y_gfm_train_f[:, label, 0] == 0
        boolean_index_valid = y_gfm_validation[:, label, 0] == 0

        # we don't need the first row (P(y=0))
        y_dummies_train = y_gfm_train_f[boolean_index_train, label, 1:]
        y_dummies_validation = y_gfm_validation[boolean_index_valid, label, 1:]

        # We need to transform the labels such that they start from zero
        # And such that each class occurs at least once in the dataset (to avoid errors when minimizing the NLL)
        # A backtransform will be required later on

        y_train = np.argmax(y_dummies_train, axis=1)
        y_validation = np.argmax(y_dummies_validation, axis=1)

        y_train_transformed = y_train - y_train.min()
        y_validation_transformed = y_validation - y_train.min()

        x_train = features_train_f[boolean_index_train, :]
        x_validation = features_validation[boolean_index_valid, :]

        n_classes = len(np.unique(y_train))
        n_features = x_train.shape[1]

        # fetch NLL
        proportialoddsmodel = ProportionalOdds_TF(n_features, n_classes, 0.1, 0.1)
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
        sess = tf.Session()
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
                    print('NaN loss!')
                    b_current = b.eval()
                    w_current = w.eval()
                    encoding_current = xW.eval(feed_dict={features: x_train})
                    print('Current biases : {}'.format(b.eval()))
                    print('Current weights: {}'.format(w.eval()))
                    break

                # Control flow for early stopping
                if validation_loss_hist[i] - np.min(validation_loss_hist) > min_delta:
                    patience_counter += 1
                    # print('Patience {}...'.format(patience_counter))
                else:
                    patience_counter = 0

                if patience_counter == patience or i == epochs:
                    # print('Early stopping... ({} epochs)'.format(i))
                    # print('Optimal weights: {}'.format(w_opt))
                    # print('Optimal biases : {}'.format(b.eval()))
                    break

            # Make predictions for the subset of data that was used to train the OR model
            b_opt = b.eval()
            encoding_train = xW.eval(feed_dict={features: x_train})
            encoding_validation = xW.eval(feed_dict={features: x_validation})
            preds_train = np.sum(encoding_train <= b_opt, axis=1)
            preds_validation = np.sum(encoding_validation <= b_opt, axis=1)
            acc_train = metrics.accuracy_score(y_train_transformed, preds_train)
            acc_validation = metrics.accuracy_score(y_validation_transformed, preds_validation)

            # Finally, make predictions for all instances (we don't know where the exact marginals at test time)
            encoding_train_full = xW.eval(feed_dict={features: features_train_f})
            encoding_validation_full = xW.eval(
                feed_dict={features: features_validation})
            encoding_test_full = xW.eval(feed_dict={features: features_test})

        # tf.reset_default_graph()

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
            predicted_marginals_train_f[:, label].reshape(-1, 1)
        probabilities_validation = conditionals_validation * \
            predicted_marginals_validation[:, label].reshape(-1, 1)
        probabilities_test = conditionals_test * predicted_marginals_test[:, label].reshape(-1, 1)

        GFM_train_entries.append(probabilities_train)
        GFM_validation_entries.append(probabilities_validation)
        GFM_test_entries.append(probabilities_test)

    GFM_train_entries = np.stack(GFM_train_entries).transpose(1, 0, 2)
    GFM_validation_entries = np.stack(GFM_validation_entries).transpose(1, 0, 2)
    GFM_test_entries = np.stack(GFM_test_entries).transpose(1, 0, 2)

    # Fill them
    train_predictions_filled = [complete_matrix_columns_with_zeros(
        mat[:, :], len=n_labels) for mat in tqdm(GFM_train_entries)]
    validation_predictions_filled = [complete_matrix_columns_with_zeros(
        mat[:, :], len=n_labels) for mat in tqdm(GFM_validation_entries)]
    test_predictions_filled = [complete_matrix_columns_with_zeros(
        mat[:, :], len=n_labels) for mat in tqdm(GFM_test_entries)]

    # Run GFM for F1
    GFM = GeneralFMaximizer(beta, n_labels)
    (optimal_predictions_train, E_F_train) = GFM.get_predictions(
        predictions=train_predictions_filled)
    (optimal_predictions_validation, E_F_validation) = GFM.get_predictions(
        predictions=validation_predictions_filled)
    (optimal_predictions_test, E_F_test) = GFM.get_predictions(predictions=test_predictions_filled)

    # Evaluate F score
    F_train = compute_F_score(y_true_train_f, optimal_predictions_train, t=0.5, beta=beta)
    F_validation = compute_F_score(
        y_true_validation, optimal_predictions_validation, t=0.5, beta=beta)
    F_test = compute_F_score(y_true_test, optimal_predictions_test, t=0.5, beta=beta)

    F1_train_OR.append(F_train)
    F1_val_OR.append(F_validation)
    F1_test_OR.append(F_test)


# Store everything
import pickle
pickle.dump(F1_train_MC, open("../results/GFM_vs_OR/F{}_MC_f_{}.p".format(beta, dataset), "wb"))
pickle.dump(F1_val_MC, open("../results/GFM_vs_OR/F{}_MC_f_{}.p".format(beta, dataset), "wb"))
pickle.dump(F1_test_MC, open("../results/GFM_vs_OR/F{}_MC_test_f_{}.p".format(beta, dataset), "wb"))
pickle.dump(F1_train_OR, open("../results/GFM_vs_OR/F{}_OR_f_{}.p".format(beta, dataset), "wb"))
pickle.dump(F1_val_OR, open("../results/GFM_vs_OR/F{}_OR_f_{}.p".format(beta, dataset), "wb"))
pickle.dump(F1_test_OR, open("../results/GFM_vs_OR/F{}_OR_test_f_{}.p".format(beta, dataset), "wb"))
