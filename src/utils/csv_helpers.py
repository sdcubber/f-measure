import os
import ast
import numpy as np
import pandas as pd


# Read dataframe
df_train = pd.read_csv("../data/KAGGLE_PLANET/train_labels.csv")


def flatten(l): return [item for sublist in l for item in sublist]


# Make label map
labels = sorted(list(set(flatten([l.split(' ') for l in df_train['tags'].values]))))
label_map = {l: i for i, l in enumerate(labels)}


def binarize_label_string(labelstring):
    """Convert label string to binary vector"""
    assert labelstring.__class__ == str, 'Provide a string'
    binary = np.zeros(len(label_map), dtype=int)
    binary[[label_map[t] for t in labelstring.split(' ')]] = 1

    return(list(binary))


def string_to_np_array(list_as_string):
    """convert list contained in a string to numpy array"""
    return np.array(ast.literal_eval(list_as_string))


def decode_label_vector(labelvector):
    """Convert label vector to label strings
    Labelvector should be a np array"""
    return ' '.join(np.array(labels)[labelvector == 1.])


def split_path_byte(filepath_byte):
    return os.path.basename(os.path.normpath(filepath_byte)).decode('utf-8').split('.')[0]


def make_full_path(image_name, path):
    """Convert image name to full image path"""
    return path + image_name + '.jpg'


def create_submission_file(mapping, name='test'):
    df = pd.read_csv('../data/KAGGLE_PLANET/test_labels.csv')
    df['tags'] = df['image_name'].map(mapping)
    df.to_csv('../submission_files/{}.csv'.format(name), index=False)
