"""
GFM with multiclass classification for multilabel classification tasks
Both F1 and F2 scores are calculated
"""

import os
import ast
import sys
import time
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
from classifiers.nn import GFM_classifier
from classifiers.cnn import GFM_VGG_classifier
from classifiers.F_score import compute_F_score
from classifiers.gfm import GeneralFMaximizer, complete_matrix_columns_with_zeros
# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def GFM_MLC(args, logger, timestamp):

    # Parameters
    im_size = args.im_size
    batch_size = 32
    dataset = args.dataset
    pretrained = args.pretrained
    epochs = 1000  # early stopping on validation data
    verbosity = 1
    c = args.c
    lr = args.lr
    opt = args.opt
    imagenet = args.imagenet
    n_hidden = args.n_hidden

    logger.log('PRETRAINED: {}'.format(pretrained))
    features_train = None
    features_validation = None
    features_test = None

    if pretrained:
        features_train = np.load('../data/{}/features/features_train_max.npy'.format(dataset))
        features_validation = np.load(
            '../data/{}/features/features_validation_max.npy'.format(dataset))
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

    # extract GFM output in case of training without generator
    if pretrained:
        y_gfm_train = np.array([ast.literal_eval(df_train['gfm_labels'][i])
                                for i in range(len(df_train))])
        y_gfm_validation = np.array([ast.literal_eval(df_validation['gfm_labels'][i])
                                     for i in range(len(df_validation))])

    # Compute max_s: the maximum number of positive label for a single instance
    max_s = np.max(np.array([np.max(np.sum(y_true_train, axis=1)),
                             np.max(np.sum(y_true_validation, axis=1)),
                             np.max(np.sum(y_true_test, axis=1))]))

    # Data generators for training
    train_gen = gn.DataGenerator_gfm_MC(df=df_train, n_labels=n_labels,
                                        im_size=im_size, batch_size=batch_size, shuffle=True, mode='train', pretrained=False, max_s=max_s).generate()

    validation_gen = gn.DataGenerator_gfm_MC(df=df_validation, n_labels=n_labels,
                                             im_size=im_size, batch_size=batch_size, shuffle=False, mode='train', pretrained=False, max_s=max_s).generate()

    # Set up the model
    if pretrained:
        model = GFM_classifier(n_features, n_labels, max_s, c).model
        optimizer = Adam()
    else:
        model = GFM_VGG_classifier(im_size, n_labels, n_hidden, imagenet, max_s).model
        optimizer = Adam(lr=lr)

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
                      patience=3, verbose=verbosity, mode='auto'),
        ModelCheckpoint('../models/GFMMLC_{}_{}_{}.h5'.format(dataset, im_size, int(pretrained)),
                        monitor='val_loss', save_best_only=True, verbose=verbosity)
    ]

    if pretrained:
        model.fit(x=features_train, y=y_gfm_train,
                  batch_size=batch_size, epochs=epochs,
                  verbose=verbosity, callbacks=callbacks, validation_data=(features_validation, y_gfm_validation))
    else:
        history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs, verbose=verbosity,
                                      callbacks=callbacks, validation_data=validation_gen, validation_steps=validation_steps)
        # Store history
        import pickle
        pickle.dump(history.history, open(
            '../results/learningcurves/GFM_MLC_{}_{}.p'.format(dataset, timestamp), 'wb'))

    # Load best model
    model.load_weights('../models/GFMMLC_{}_{}_{}.h5'.format(dataset, im_size, int(pretrained)))
    model.compile(loss=GFM_loss, optimizer=optimizer)

    # Data generators for inference
    train_gen_i = gn.DataGenerator_gfm_MC(df=df_train, n_labels=n_labels,
                                          im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False, max_s=max_s).generate()
    validation_gen_i = gn.DataGenerator_gfm_MC(df=df_validation, n_labels=n_labels,
                                               im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False, max_s=max_s).generate()
    test_gen_i = gn.DataGenerator_gfm_MC(df=df_test, n_labels=n_labels,
                                         im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False, max_s=max_s).generate()

    # Make predictions
    if pretrained:
        pis_train = model.predict(features_train, verbose=1)
        pis_validation = model.predict(features_validation, verbose=1)
        pis_test = model.predict(features_test, verbose=1)
    else:
        pis_train = model.predict_generator(train_gen_i, steps=train_steps, verbose=1)
        pis_validation = model.predict_generator(
            validation_gen_i, steps=validation_steps, verbose=1)
        pis_test = model.predict_generator(test_gen_i, steps=test_steps, verbose=1)

    def softmax(v):
        """softmax a vector
        Adaptation for numerical stability according to
        http://python.usyiyi.cn/documents/effective-tf/12.html
        """
        exp = np.exp(v - np.max(v))
        return(exp / np.sum(exp))

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

    # (Extra: do a postprocessing: constrain the rank of the output matrices) before running GFM
    # Store output of network to this end

    np.save('../results/GFM_MLC_output_train_{}_pt{}'.format(dataset,
                                                             int(pretrained)), np.array(pis_train_filled))
    np.save('../results/GFM_MLC_output_validation_{}_pt{}'.format(dataset, int(pretrained)),
            np.array(pis_validation_filled))
    np.save('../results/GFM_MLC_output_test_{}_pt{}'.format(dataset,
                                                            int(pretrained)), np.array(pis_test_filled))
    # Compute optimal predictions for F1
    for beta in [1, 2]:
        GFM = GeneralFMaximizer(beta, n_labels)

        # Run GFM algo on this output
        (optimal_predictions_train, E_F_train) = GFM.get_predictions(predictions=pis_train_filled)
        (optimal_predictions_validation, E_F_validation) = GFM.get_predictions(
            predictions=pis_validation_filled)
        (optimal_predictions_test, E_F_test) = GFM.get_predictions(predictions=pis_test_filled)

        # Evaluate F score
        F_train = compute_F_score(y_true_train, optimal_predictions_train, t=0.5, beta=beta)
        F_validation = compute_F_score(
            y_true_validation, optimal_predictions_validation, t=0.5, beta=beta)
        F_test = compute_F_score(y_true_test, optimal_predictions_test, t=0.5, beta=beta)

        logger.log('GFM_MLC ({})'.format(dataset))
        logger.log('-' * 50)
        logger.log('F{} score on training data: {:.4f}'.format(beta, F_train))
        logger.log('F{} score on validation data: {:.4f}'.format(beta, F_validation))
        logger.log('F{} score on test data: {:.4f}'.format(beta, F_test))

        # Store test set predictions to submit to Kaggle
        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in optimal_predictions_test]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_GFM_MC_pt{}'.format(int(pretrained)))


def main():
    parser = argparse.ArgumentParser(
        description='GFM_MLC')
    parser.add_argument('im_size', type=int, help='image size')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('c', type=float, help='amount of dropout', default=0.)
    parser.add_argument('lr', type=float, help='learning rate')
    parser.add_argument('opt', type=str, choices=['sgd', 'adam'])
    parser.add_argument('n_hidden', type=int, help='number of neurons in hidden layer')
    parser.add_argument('-im', '--imagenet', action='store_true', help='use imagenet weights')
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='Use pretrained VGG features')
    parser.add_argument('-name', type=str, help='name of experiment', default='GFM_MLC')
    args = parser.parse_args()
    args.name = '_'.join((args.name, args.dataset))

    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    logger = loggerClass(args, timestamp)
    GFM_MLC(args, logger, timestamp)


if __name__ == "__main__":
    sys.exit(main())
    # main()
