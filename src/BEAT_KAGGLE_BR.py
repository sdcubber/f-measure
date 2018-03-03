"""
Obtain a high score on the Kaggle LB with the GFM method
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
from keras.callbacks import EarlyStopping

# Custom modules
import utils.generators as gn
from utils import csv_helpers
from classifiers.nn import GFM_classifier
from classifier.cnn import GFM_CNN_classifier, VGG_classifier
from classifiers.F_score import compute_F_score
from classifiers.gfm import GeneralFMaximizer, complete_matrix_columns_with_zeros
import classifiers.thresholding as th

# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def BR(args):

    # Parameters
    im_size = args.im_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    augmentation = args.augmentation
    epochs = args.epochs
    earlystop = args.earlystop
    name = args.name
    n_neurons = args.n_neurons

    n_labels = 17
    dataset = 'KAGGLE_PLANET'
    csv_path_train = '../data/{}/TRAIN.csv'.format(dataset)
    csv_path_validation = '../data/{}/VALIDATION.csv'.format(dataset)
    csv_path_trainval = '../data/{}/TRAINVAL.csv'.format(dataset)
    csv_path_test = '../data/{}/TEST.csv'.format(dataset)

    df_train = pd.read_csv(csv_path_train)
    df_validation = pd.read_csv(csv_path_validation)
    df_trainval = pd.read_csv(csv_path_trainval)
    df_test = pd.read_csv(csv_path_test)

    train_steps = np.ceil(len(df_train) / batch_size)
    validation_steps = np.ceil(len(df_validation) / batch_size)
    trainval_steps = np.ceil(len(df_trainval) / batch_size)
    test_steps = np.ceil(len(df_test) / batch_size)

    # Extract ground truth labels
    y_true_train = np.array([ast.literal_eval(df_train['marginal_labels'][i])
                             for i in range(len(df_train))])
    y_true_validation = np.array([ast.literal_eval(df_validation['marginal_labels'][i])
                                  for i in range(len(df_validation))])
    y_true_train_val = np.array([ast.literal_eval(df_trainval['marginal_labels'][i])
                                 for i in range(len(df_trainval))])
    y_true_test = np.array([ast.literal_eval(df_test['marginal_labels'][i])
                            for i in range(len(df_test))])

    # Data generators for training
    train_gen = gn.DataGenerator(df=df_train, n_labels=n_labels,
                                 im_size=im_size, batch_size=batch_size,
                                 shuffle=True, mode='train',
                                 pretrained=False, augmentation=augmentation).generate()

    validation_gen = gn.DataGenerator(df=df_validation, n_labels=n_labels,
                                      im_size=im_size, batch_size=batch_size, shuffle=False, mode='train', pretrained=False).generate()

    trainval_gen = gn.DataGenerator(df=df_trainval, n_labels=n_labels,
                                    im_size=im_size, batch_size=batch_size,
                                    shuffle=True, mode='train',
                                    pretrained=False, augmentation=augmentation).generate()

    # Set up the model
    model = VGG_classifier(im_size, n_labels, n_neurons).model
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0,
                      patience=3, verbose=1, mode='auto')
    ]

    if earlystop:
        model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs, verbose=1,
                            callbacks=callbacks, validation_data=validation_gen, validation_steps=validation_steps)
    else:
        model.fit_generator(trainval_gen, steps_per_epoch=trainval_steps, epochs=epochs, verbose=1)

        # Data generators for inference
    train_gen_i = gn.DataGenerator(df=df_train, n_labels=n_labels,
                                   im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()
    validation_gen_i = gn.DataGenerator(df=df_validation, n_labels=n_labels,
                                        im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()
    trainval_gen_i = gn.DataGenerator(df=df_trainval, n_labels=n_labels, im_size=im_size,
                                      batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()
    test_gen_i = gn.DataGenerator(df=df_test, n_labels=n_labels,
                                  im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=False).generate()

    # Make predictions
    if earlystop:
        predicted_marginals_train = model.predict_generator(
            train_gen_i, steps=train_steps, verbose=1)
        predicted_marginals_validation = model.predict_generator(
            validation_gen_i, steps=validation_steps, verbose=1)
    else:
        predicted_marginals_trainval = model.predict_generator(
            trainval_gen_i, steps=trainval_steps, verbose=1)
    predicted_marginals_test = model.predict_generator(test_gen_i, steps=test_steps, verbose=1)

    # Compute optimal predictions for F2
    beta = 2
    # Evaluate F score
    if earlystop:
        F_train = compute_F_score(y_true_train, predicted_marginals_train, t=0.5, beta=beta)
        F_validation = compute_F_score(
            y_true_validation, predicted_marginals_validation, t=0.5, beta=beta)
    else:
        F_train = F_validation = compute_F_score(
            y_true_train_val, predicted_marginals_trainval, t=0.5, beta=beta)
    F_test = compute_F_score(y_true_test, predicted_marginals_test, t=0.5, beta=beta)

    print('BR with threshold 0.5 - ({})'.format(dataset))
    if not earlystop:
        print('-' * 50)
        print('---- No early stopping on validation data ----')
    print('-' * 50)
    print('F{} score on training data: {:.4f}'.format(beta, F_train))
    print('F{} score on validation data: {:.4f}'.format(beta, F_validation))
    print('F{} score on test data: {:.4f}'.format(beta, F_test))

    # Map predictions to filenames
    def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
    test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
    GFM_predictions_mapping = dict(
        zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in (predicted_marginals_test > 0.5).astype(int)]))
    # Create submission file
    csv_helpers.create_submission_file(
        GFM_predictions_mapping, name='BR0.5_KAGGLE_imsize{}_lr{}_ep{}_ag{}_{}'.format(im_size, learning_rate, epochs, int(augmentation), name))

    # Compute global optimal threshold
    # Sort true and predicted row-wise on Hi
    algorithm_2 = th.OptimalGlobalThreshold(beta)
    if earlystop:
        optimal_t_2 = algorithm_2.get_optimal_t(y_true_train, predicted_marginals_train)
    else:
        optimal_t_2 = algorithm_2.get_optimal_t(y_true_train_val, predicted_marginals_trainval)

    # Evaluate F score
    if earlystop:
        F_train = compute_F_score(y_true_train, predicted_marginals_train, t=optimal_t_2, beta=beta)
        F_validation = compute_F_score(
            y_true_validation, predicted_marginals_validation, t=optimal_t_2, beta=beta)
    else:
        F_train = F_validation = compute_F_score(
            y_true_train_val, predicted_marginals_trainval, t=optimal_t_2, beta=beta)
    F_test = compute_F_score(y_true_test, predicted_marginals_test, t=optimal_t_2, beta=beta)

    print('\n')
    print('Results with global optimal threshold {:.2f} - ({})'.format(optimal_t_2, dataset))
    print('--' * 20)
    print('Mean F{}-score with optimal threshold - Train: {:.4f}'.format(beta, F_train))
    print('Mean F{}-score with optimal threshold - Val: {:.4f}'.format(beta, F_validation))
    print('Mean F{}-score with optimal threshold - Test: {:.4f}'.format(beta, F_test))

    # Store test set predictions to submit to Kaggle
    if (dataset == 'KAGGLE_PLANET') and (beta == 2):
        # Map predictions to filenames
        def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
        test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
        GFM_predictions_mapping = dict(
            zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in (predicted_marginals_test > optimal_t_2).astype(int)]))
        # Create submission file
        csv_helpers.create_submission_file(
            GFM_predictions_mapping, name='BR_opt_KAGGLE_imsize{}_lr{}_ep{}_ag{}_nn_{}_{}'.format(im_size, learning_rate, epochs, int(augmentation), n_neurons, name))


def main():
    parser = argparse.ArgumentParser(
        description='GFM_MLC')
    parser.add_argument('im_size', type=int, help='image size')
    parser.add_argument('batch_size', type=int, default=32, help='batch size')
    parser.add_argument('learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('name', type=str, default=' ', help='experiment name')
    parser.add_argument('n_neurons', type=int, default=128,
                        help='number of neurons in hidden layer')
    parser.add_argument('-nes', '--earlystop', action='store_false',
                        help='no early stopping on validation data')
    parser.add_argument('-ag', '--augmentation', action='store_true',
                        help='apply data augmentation')

    args = parser.parse_args()
    BR(args)


if __name__ == "__main__":
    sys.exit(main())
    # main()
