import os
# os.chdir('../src')
import ast
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# custom modules
import utils.generators as gn
from utils import csv_helpers
import classifiers.thresholding as th
from classifiers.F_score import compute_F_score

# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def thresh_stack(args, logger):

        # Parameters
    dataset = args.dataset
    pretrained = args.pretrained
    nonlinear = args.nonlinear

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

    n_labels = y_true_train.shape[1]

    # Thresholding algorithm 1

    for beta in [1, 2]:
        # Sort true and predicted row-wise on Hi
        algorithm_1 = th.OptimalMeanThreshold(beta)
        optimal_t_1 = algorithm_1.get_optimal_t(y_true_train, y_predicted_train)

        # Also get instance-wise threshold for the validation data to use them as labels in algorithm 3
        optimal_t_1_validation = algorithm_1.get_optimal_t(
            y_true_validation, y_predicted_validation)

        #t_train = np.log(optimal_t_1)
        #t_validation = np.log(optimal_t_1_validation)

        # 'temporary' hack: replace marginal labels in dataframe with optimized thresholds
        # So as to avoid having to write another generator
        df_train['marginal_labels'] = [str(t) for t in list(optimal_t_1)]
        df_validation['marginal_labels'] = [str(t) for t in list(optimal_t_1_validation)]

        from sklearn.linear_model import RidgeCV
        from sklearn.ensemble import RandomForestRegressor

        alphas = np.logspace(-5, 5, 100)
        model = RidgeCV(alphas=alphas)
        if nonlinear:
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

        # Rescale
        y_mean = np.mean(y_predicted_train, axis=0)
        y_std = np.std(y_predicted_train, axis=0)

        m_train = (y_predicted_train - y_mean) / y_std
        m_validation = (y_predicted_validation - y_mean) / y_std
        m_test = (y_predicted_test - y_mean) / y_std

        model.fit(X=m_train, y=optimal_t_1)
        if not nonlinear:
            assert (model.alpha_ < alphas[-1]) and (model.alpha_ >
                                                    alphas[0]), 'Increase the search range for lambda'

        # Make prediction
        predictions_train_t = model.predict(m_train)
        predictions_validation_t = model.predict(m_validation)
        predictions_test_t = model.predict(m_test)

        from sklearn.metrics import r2_score
        logger.log('R²: {:.2f}'.format(r2_score(optimal_t_1_validation, predictions_validation_t)))

        # Store the true and predicted thresholds of the validation dataset to make plots
        np.save('../results/INST_THRESH/{}_{}_{}'.format(dataset,
                                                         beta, int(pretrained)), optimal_t_1_validation)
        np.save('../results/INST_THRESH/{}_{}_{}_predicted'.format(dataset,
                                                                   beta, int(pretrained)), predictions_validation_t)

        # Print results
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

        # Evaluate F score with mean threshold
        F_train = compute_F_score(y_true_train, y_predicted_train,
                                  t=np.mean(optimal_t_1), beta=beta)
        F_validation = compute_F_score(y_true_validation, y_predicted_validation,
                                       t=np.mean(optimal_t_1), beta=beta)
        F_test = compute_F_score(y_true_test, y_predicted_test, t=np.mean(optimal_t_1), beta=beta)

        logger.log('\n')
        logger.log(
            'Results with mean optimal threshold {:.2f} - ({})'.format(np.mean(optimal_t_1), dataset))
        logger.log('--' * 20)
        logger.log('Mean F{}-score with optimal thresholds - Train: {:.4f}'.format(beta, F_train))
        logger.log('Mean F{}-score with optimal thresholds - Val: {:.4f}'.format(beta, F_validation))
        logger.log('Mean F{}-score with optimal thresholds - Test: {:.4f}'.format(beta, F_test))

        # Evaluate F score with predicted instance-wise threshold
        from sklearn.metrics import fbeta_score

        def compute_F_score_instancewise_threshold(y_true, predictions, t, beta):
            return(fbeta_score(y_true, np.array([predictions[i, :] > t[i] for i in range(len(y_true))]).astype(int), beta=beta, average='samples'))

        F_train = compute_F_score_instancewise_threshold(
            y_true_train, y_predicted_train, t=predictions_train_t, beta=beta)
        F_validation = compute_F_score_instancewise_threshold(
            y_true_validation, y_predicted_validation, t=predictions_validation_t, beta=beta)
        F_test = compute_F_score_instancewise_threshold(
            y_true_test, y_predicted_test, t=predictions_test_t, beta=beta)

        logger.log('\n')
        logger.log('Results with instance-wise threshold')
        logger.log('--' * 20)
        logger.log('Mean F{}-score with instance-wise threshold - Train: {:.4f}'.format(beta, F_train))
        logger.log('Mean F{}-score with instance-wise threshold - Val: {:.4f}'.format(beta, F_validation))
        logger.log('Mean F{}-score with instance-wise threshold - Test: {:.4f}'.format(beta, F_test))

        # Store test set predictions to submit to Kaggle

        test_predictions = np.array([y_predicted_test[i, :] > predictions_test_t[i]
                                     for i in range(len(predictions_test_t))]).astype(int)

        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in test_predictions]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_BR_InstanceWiseThreshold_{}'.format(int(pretrained)))


def main():
    parser = argparse.ArgumentParser(
        description='Threshold stacking')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='Use marginals predicted on pretrained features')
    parser.add_argument('-name', type=str, help='name of experiment', default='THRESH_stack')
    parser.add_argument('-nl', '--nonlinear', action='store_true', help='use nonlinear model (RF)')

    args = parser.parse_args()
    args.name = ' '.join((args.name, args.dataset))

    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    logger = loggerClass(args, timestamp)
    thresh_stack(args, logger)


if __name__ == "__main__":
    sys.exit(main())
    # main()
