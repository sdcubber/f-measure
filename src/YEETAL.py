# Algorithm of Ye et al (2012) for multi-label classification
#[Optimizing F-measures: A Tale of Two Approaches] - https://arxiv.org/pdf/1206.4625.pdf
#
# Author: Stijn Decubber

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
from keras.callbacks import EarlyStopping


# Custom modules
import utils.generators as gn
from utils import csv_helpers
import classifiers.thresholding as th
from classifiers.F_score import compute_F_score
from classifiers.ye_et_al import QuadraticTimeAlgorithm

# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def ye_et_al(args, logger):

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
        logger.log('Binary relevance with threshold 0.5 - ({})'.format(dataset))
        logger.log('-' * 50)
        logger.log('F{} score on training data: {:.4f}'.format(beta, F_train))
        logger.log('F{} score on validation data: {:.4f}'.format(beta, F_validation))
        logger.log('F{} score on test data: {:.4f}'.format(beta, F_test))

        # Ye et al (2012): plug-in rule algorithm that takes the predicted marginals as input
        algorithm = QuadraticTimeAlgorithm(beta)

        optimal_predictions_train = np.array(
            [algorithm.get_predictions(i) for i in tqdm(y_predicted_train)])
        optimal_predictions_validation = np.array(
            [algorithm.get_predictions(i) for i in tqdm(y_predicted_validation)])
        optimal_predictions_test = np.array(
            [algorithm.get_predictions(i) for i in tqdm(y_predicted_test)])

        F_GFM_MC_train = compute_F_score(y_true_train, optimal_predictions_train, t=0.5, beta=beta)
        F_GFM_MC_validation = compute_F_score(
            y_true_validation, optimal_predictions_validation, t=0.5, beta=beta)
        F_GFM_MC_test = compute_F_score(y_true_test, optimal_predictions_test, t=0.5, beta=beta)
        logger.log('\n')
        logger.log('F{} scores with algorithm of Ye et al (2012) - ({})'.format(beta, dataset))
        logger.log('-' * 50)
        logger.log('F{} score on training data: {:.4f}'.format(beta, F_GFM_MC_train))
        logger.log('F{} score on validation data: {:.4f}'.format(beta, F_GFM_MC_validation))
        logger.log('F{} score on test data: {:.4f}'.format(beta, F_GFM_MC_test))

        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in optimal_predictions_test]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_Yeetal_2012_pt{}'.format(int(pretrained)))


def main():
    parser = argparse.ArgumentParser(
        description='YE2012')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='Use pretrained VGG features')
    parser.add_argument('-name', type=str, help='name of experiment', default='YE2012')

    args = parser.parse_args()
    args.name = '_'.join((args.name, args.dataset))

    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    logger = loggerClass(args, timestamp)
    ye_et_al(args, logger)


if __name__ == "__main__":
    # sys.exit(main())
    main()
