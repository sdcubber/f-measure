# Evaluate performance of different thresholding algorithms
# Thresholding algorithm 1: binary relevance with a single optimized threshold for all instances and all labels.
# The best threshold is computed for all instances in parallell, and the mean is used as the optimal t.


# Author: Stijn Decubber

import os
import ast
import time
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

# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def thresholding(args, logger):

    # Parameters
    dataset = args.dataset
    pretrained = args.pretrained

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

    # Load the predicted marginals
    y_predicted_train = np.load(
        '../results/BR_predictions_train_{}_pt{}.npy'.format(dataset, int(pretrained)))
    y_predicted_validation = np.load(
        '../results/BR_predictions_validation_{}_pt{}.npy'.format(dataset, int(pretrained)))
    y_predicted_test = np.load(
        '../results/BR_predictions_test_{}_pt{}.npy'.format(dataset, int(pretrained)))

    for beta in [1, 2]:
        # Evaluate F score with threshold 0.5
        F_train = compute_F_score(y_true_train, y_predicted_train, t=0.5, beta=beta)
        F_validation = compute_F_score(y_true_validation, y_predicted_validation, t=0.5, beta=beta)
        F_test = compute_F_score(y_true_test, y_predicted_test, t=0.5, beta=beta)

        logger.log('\n')
        logger.log('Binary relevance with threshold 0.5 - ({}) '.format(dataset))
        logger.log('-' * 50)
        logger.log('F{} score on training data: {:.4f}'.format(beta, F_train))
        logger.log('F{} score on validation data: {:.4f}'.format(beta, F_validation))
        logger.log('F{} score on test data: {:.4f}'.format(beta, F_test))

        # Thresholding algorithm 1
        # Sort true and predicted row-wise on Hi
        algorithm_1 = th.OptimalMeanThreshold(beta)
        optimal_t_1 = algorithm_1.get_optimal_t(y_true_train, y_predicted_train)
        # Also get instance-wise threshold for the validation data to use them as labels in algorithm 3
        optimal_t_1_validation = algorithm_1.get_optimal_t(
            y_true_validation, y_predicted_validation)

        # Evaluate F score
        F_train = compute_F_score(y_true_train, y_predicted_train,
                                  t=np.mean(optimal_t_1), beta=beta)
        F_validation = compute_F_score(y_true_validation, y_predicted_validation,
                                       t=np.mean(optimal_t_1), beta=beta)
        F_test = compute_F_score(y_true_test, y_predicted_test, t=np.mean(optimal_t_1), beta=beta)

        logger.log('\n')
        logger.log(
            'Results with mean optimal threshold {:.2f} - ({})'.format(np.mean(optimal_t_1), dataset))
        logger.log('--' * 20)
        logger.log('Mean F{}-score with mean threshold - Train: {:.4f}'.format(beta, F_train))
        logger.log('Mean F{}-score with mean threshol - Val: {:.4f}'.format(beta, F_validation))
        logger.log('Mean F{}-score with mean threshol - Test: {:.4f}'.format(beta, F_test))

        # Store test set predictions to submit to Kaggle
        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in (y_predicted_test > np.mean(optimal_t_1)).astype(int)]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_BR_OptimalMeanThreshold')

        # Thresholding algorithm 2
        # Sort true and predicted row-wise on Hi
        algorithm_2 = th.OptimalGlobalThreshold(beta)
        optimal_t_2 = algorithm_2.get_optimal_t(y_true_train, y_predicted_train)

        # Evaluate F score
        F_train = compute_F_score(y_true_train, y_predicted_train, t=optimal_t_2, beta=beta)
        F_validation = compute_F_score(y_true_validation, y_predicted_validation,
                                       t=optimal_t_2, beta=beta)
        F_test = compute_F_score(y_true_test, y_predicted_test, t=optimal_t_2, beta=beta)

        logger.log('\n')
        logger.log(
            'Results with global optimal threshold {:.2f} - ({})'.format(optimal_t_2, dataset))
        logger.log('--' * 20)
        logger.log('Mean F{}-score with global threshold - Train: {:.4f}'.format(beta, F_train))
        logger.log('Mean F{}-score with global threshold - Val: {:.4f}'.format(beta, F_validation))
        logger.log('Mean F{}-score with global threshold - Test: {:.4f}'.format(beta, F_test))

        # Store test set predictions to submit to Kaggle
        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in (y_predicted_test > optimal_t_2).astype(int)]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_BR_OptimalGlobalThreshold_pt{}'.format(int(pretrained)))


import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='thresholding')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='Load marginals from experiments with pretrained features')
    parser.add_argument('-name', type=str, help='name of experiment', default='THRESH')

    args = parser.parse_args()

    args.name = '_'.join((args.name, args.dataset))
    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    logger = loggerClass(args, timestamp)
    thresholding(args, logger)


if __name__ == "__main__":
    # sys.exit(main())
    main()
