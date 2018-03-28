"""
Binary relevance methods for multi-label classification tasks
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
from classifiers.nn import BR_classifier
from classifiers.cnn import BR_CNN_classifier, VGG_classifier
from classifiers.F_score import compute_F_score

# Import logger
from utils.logger import loggerClass

# Let TF see only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def BR(args, logger, timestamp):
    # Parameters
    im_size = args.im_size
    batch_size = 16
    dataset = args.dataset
    pretrained = args.pretrained
    epochs = 100  # early stopping on validation data
    verbosity = 1
    c = args.c
    lr = args.lr
    opt = args.opt
    imagenet = args.imagenet
    n_hidden = args.n_hidden

    features_train = None
    features_validation = None
    features_test = None
    if (dataset == 'KAGGLE_PLANET') or (dataset == 'MS_COCO'):
        dropout_rates = [0.10, 0.5]
    elif (dataset == 'PASCAL_VOC_2007') or (dataset == 'PASCAL_VOC_2012'):
        dropout_rates = [0.25, 0.75]

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
    y_true_train = np.array([np.array(ast.literal_eval(l)) for l in df_train['marginal_labels']])
    y_true_validation = np.array([np.array(ast.literal_eval(l))
                                  for l in df_validation['marginal_labels']])
    y_true_test = np.array([np.array(ast.literal_eval(l))
                            for l in df_test['marginal_labels']])

    n_labels = y_true_train.shape[1]

    # Data generators for training
    train_gen = gn.DataGenerator(df=df_train, n_labels=n_labels,
                                 im_size=im_size, batch_size=batch_size, shuffle=True, mode='train', pretrained=pretrained, features=features_train).generate()

    validation_gen = gn.DataGenerator(df=df_validation, n_labels=n_labels,
                                      im_size=im_size, batch_size=batch_size, shuffle=False, mode='train', pretrained=pretrained, features=features_validation).generate()

    # Set up the model
    if pretrained:
        model = BR_classifier(n_features, n_labels, c).model
        optimizer = Adam()
    else:
        model = VGG_classifier(im_size, n_labels, n_hidden, imagenet).model
        if opt == 'sgd':
            optimizer = SGD(lr=lr)  # Use smaller lr
        else:
            optimizer = Adam(lr=lr)

    # First, freeze all layers but the final one
    for layer in model.layers[:-6]:
        layer.trainable = False

    # Disable dropout for the pretraining
    model.layers[-2].rate = dropout_rates[0]
    model.layers[-5].rate = dropout_rates[0]

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    print(model.summary())
    print(model.layers[-2].get_config())
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.,
                      patience=3, verbose=1, mode='auto'),
        ModelCheckpoint('../models/BR_{}_{}_{}.h5'.format(dataset, im_size, int(pretrained)),
                        monitor='val_loss', save_best_only=True, verbose=1)
    ]

    if pretrained:
        history = model.fit(x=features_train, y=y_true_train,
                            batch_size=batch_size, epochs=epochs,
                            verbose=verbosity, callbacks=callbacks, validation_data=(features_validation, y_true_validation))
    else:
        history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs, verbose=verbosity,
                                      callbacks=callbacks, validation_data=validation_gen, validation_steps=validation_steps)

        # Store history
        import pickle
        pickle.dump(history.history, open(
            '../results/learningcurves/BR_{}_{}.p'.format(dataset, timestamp), 'wb'))

    # Load best model
    model.load_weights('../models/BR_{}_{}_{}.h5'.format(dataset, im_size, int(pretrained)))

    # Recompile the model, set all layers to trainable, finetune with small lr
    for layer in model.layers:
        layer.trainable = True

    # Increase dropout rate
    model.layers[-2].rate = dropout_rates[1]
    model.layers[-5].rate = dropout_rates[1]
    optimizer = Adam(lr=1e-5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    print(model.summary())
    print(model.layers[-2].get_config())
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0,
                      patience=2, verbose=1, mode='auto'),
        ModelCheckpoint('../models/BR_{}_{}_{}.h5'.format(dataset, im_size, int(pretrained)),
                        monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs, verbose=verbosity,
                        callbacks=callbacks, validation_data=validation_gen, validation_steps=validation_steps)

    model.load_weights('../models/BR_{}_{}_{}.h5'.format(dataset, im_size, int(pretrained)))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Data generators for inference
    train_gen_i = gn.DataGenerator(df=df_train, n_labels=n_labels,
                                   im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=pretrained, features=features_train).generate()
    validation_gen_i = gn.DataGenerator(df=df_validation, n_labels=n_labels,
                                        im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=pretrained, features=features_validation).generate()
    test_gen_i = gn.DataGenerator(df=df_test, n_labels=n_labels,
                                  im_size=im_size, batch_size=batch_size, shuffle=False, mode='test', pretrained=pretrained, features=features_test).generate()

    # Make predictions
    if pretrained:
        BR_predictions_train = model.predict(features_train, verbose=1)
        BR_predictions_validation = model.predict(features_validation, verbose=1)
        BR_predictions_test = model.predict(features_test, verbose=1)
    else:
        BR_predictions_train = model.predict_generator(train_gen_i, steps=train_steps, verbose=1)
        BR_predictions_validation = model.predict_generator(
            validation_gen_i, steps=validation_steps, verbose=1)
        BR_predictions_test = model.predict_generator(test_gen_i, steps=test_steps, verbose=1)

    for beta in [1, 2]:

        F_train = compute_F_score(y_true_train, BR_predictions_train, t=0.5, beta=beta)
        F_validation = compute_F_score(
            y_true_validation, BR_predictions_validation, t=0.5, beta=beta)
        F_test = compute_F_score(y_true_test, BR_predictions_test, t=0.5, beta=beta)

        logger.log('Binary relevance with threshold 0.5 - ({})'.format(dataset))
        logger.log('-' * 50)
        logger.log('F{} score on training data: {:.4f}'.format(beta, F_train))
        logger.log('F{} score on validation data: {:.4f}'.format(beta, F_validation))
        logger.log('F{} score on test data: {:.4f}'.format(beta, F_test))

        # Store test set predictions for the kaggle dataset to submit them to the website
        if (dataset == 'KAGGLE_PLANET') and (beta == 2):
            # Map predictions to filenames
            def filepath_to_filename(s): return os.path.basename(os.path.normpath(s)).split('.')[0]
            test_filenames = [filepath_to_filename(f) for f in df_test['full_path']]
            GFM_predictions_mapping = dict(
                zip(test_filenames, [csv_helpers.decode_label_vector(f) for f in (BR_predictions_test > 0.5).astype(int)]))
            # Create submission file
            csv_helpers.create_submission_file(
                GFM_predictions_mapping, name='Planet_BR_{}'.format(int(pretrained)))

    # Store marginals
    np.save('../results/BR_predictions_train_{}_pt{}'.format(dataset,
                                                             int(pretrained)), BR_predictions_train)
    np.save('../results/BR_predictions_validation_{}_pt{}'.format(dataset, int(pretrained)),
            BR_predictions_validation)
    np.save('../results/BR_predictions_test_{}_pt{}'.format(dataset,
                                                            int(pretrained)), BR_predictions_test)


def main():
    parser = argparse.ArgumentParser(
        description='BR')
    parser.add_argument('im_size', type=int, help='image size')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('c', type=float, help='amount of dropout', default=0.)
    parser.add_argument('lr', type=float, help='learning rate')
    parser.add_argument('opt', type=str, choices=['sgd', 'adam'])
    parser.add_argument('n_hidden', type=int, help='number of neurons in hidden layer')
    parser.add_argument('-im', '--imagenet', action='store_true', help='use imagenet weights')
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='Use pretrained VGG features')
    parser.add_argument('-name', type=str, help='name of experiment', default='BR')
    args = parser.parse_args()
    args.name = '_'.join((args.name, args.dataset, str(args.pretrained)))

    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    logger = loggerClass(args, timestamp)
    BR(args, logger, timestamp)


if __name__ == "__main__":
    sys.exit(main())
    # main()
