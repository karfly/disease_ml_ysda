import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer

DATA_DIR = 'data/'


def load_data(imputer_strategy='most_frequent',
              scale=False,
              ohe=None,
              hashing_trick=False,
              binary=True,
              ngram_range=(1, 1),
              feature_types=('numeric', 'category', 'genes', 'ordered_category')):
    # Loading data
    data = _load_data_from_disk()

    # Replacing missing values
    imputer = Imputer(strategy=imputer_strategy)
    data['numeric']['train'] = imputer.fit_transform(data['numeric']['train'])
    data['numeric']['test'] = imputer.transform(data['numeric']['test'])

    for feature_type in ['category', 'genes', 'ordered_category']:
        data[feature_type]['train'][np.isnan(data[feature_type]['train'])] = -999
        data[feature_type]['test'][np.isnan(data[feature_type]['test'])] = -999
        # imputer = Imputer(strategy=imputer_strategy)
        # data[feature_type]['train'] = imputer.fit_transform(data[feature_type]['train'])
        # data[feature_type]['test'] = imputer.transform(data[feature_type]['test'])

    # OHE
    if ohe is not None:
        for feature_type in ohe:
            train_len = data[feature_type]['train'].shape[0]
            data_all = np.vstack([data[feature_type]['train'], data[feature_type]['test']])
            for i in range(data[feature_type]['train'].shape[1]):
                label_encoder = LabelEncoder()
                data_all[:, i] = label_encoder.fit_transform(data_all[:, i])

            # Splitting back
            data[feature_type]['train'] = data_all[:train_len, :]
            data[feature_type]['test'] = data_all[train_len:, :]

            oh_encoder = OneHotEncoder(handle_unknown='ignore')
            data[feature_type]['train'] = np.array(oh_encoder.fit_transform(data[feature_type]['train']).todense())
            data[feature_type]['test'] = np.array(oh_encoder.transform(data[feature_type]['test']).todense())

    # Hashing trick:
    if hashing_trick:
        data['category']['train'], data['category']['test'] = _hashing_trick(
            data['category']['train'],
            data['category']['test'],
            n_features=5000,
            binary=binary,
            ngram_range=ngram_range)

        data['genes']['train'], data['genes']['test'] = _hashing_trick(
            data['genes']['train'],
            data['genes']['test'],
            n_features=10000,
            binary=binary,
            ngram_range=ngram_range)

        data['ordered_category']['train'], data['ordered_category']['test'] = _hashing_trick(
            data['ordered_category']['train'],
            data['ordered_category']['test'],
            n_features=1000,
            binary=binary,
            ngram_range=ngram_range)

    # Scaling
    if scale:
        for feature_type in ['numeric', 'ordered_category']:
            scaler = StandardScaler()
            data[feature_type]['train'] = scaler.fit_transform(data[feature_type]['train'])
            data[feature_type]['test'] = scaler.transform(data[feature_type]['test'])

    # Joining data
    x_train, y_train, x_test = _join_data(data, feature_types=feature_types)

    return x_train, y_train, x_test


def _load_data_from_disk():
    x_train = pd.read_csv(os.path.join(DATA_DIR, 'X.train.csv')).values
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y.train.csv')).values.squeeze()
    x_test = pd.read_csv(os.path.join(DATA_DIR, 'X.test.csv')).values
    meta_data = pd.read_csv(os.path.join(DATA_DIR, 'MetaData.csv'))

    numeric_indexes = meta_data[meta_data['Column Type'] == 'Numeric'].index.values

    category_indexes = meta_data[meta_data['Column Type'] == 'Category'].index.values
    genes_start_index = 330
    genes_indexes = category_indexes[genes_start_index:]
    category_indexes = category_indexes[:genes_start_index]

    ordered_category_indexes = meta_data[meta_data['Column Type'] == 'Ordered Category'].index.values

    data = {
        'y_train': y_train,

        'numeric': {
            'train': x_train[:, numeric_indexes],
            'test': x_test[:, numeric_indexes]
        },

        'category': {
            'train': x_train[:, category_indexes],
            'test': x_test[:, category_indexes]
        },

        'genes': {
            'train': x_train[:, genes_indexes],
            'test': x_test[:, genes_indexes]
        },

        'ordered_category': {
            'train': x_train[:, ordered_category_indexes],
            'test': x_test[:, ordered_category_indexes]
        }
    }

    return data


def _join_data(data, feature_types):
    y_train = data['y_train']

    x_trains = []
    x_tests = []
    for feature_type in feature_types:
        x_trains.append(data[feature_type]['train'])
        x_tests.append(data[feature_type]['test'])

    x_train = np.hstack(x_trains)
    x_test = np.hstack(x_tests)

    return x_train, y_train, x_test


def _hashing_trick(x_train, x_test, n_features, binary=True, ngram_range=(1, 1)):
    df_train = pd.DataFrame(x_train.astype('str'))
    df_test = pd.DataFrame(x_test.astype('str'))

    for col_i in range(df_train.shape[1]):
        df_train.iloc[:, col_i] = '{}='.format(col_i) + df_train.iloc[:, col_i]
        df_test.iloc[:, col_i] = '{}='.format(col_i) + df_test.iloc[:, col_i]

    texts_train = df_train.apply(lambda row: ' '.join(row), axis=1).values
    texts_test = df_test.apply(lambda row: ' '.join(row), axis=1).values

    hv = HashingVectorizer(n_features=n_features, binary=binary, ngram_range=ngram_range)
    hashed_train = hv.fit_transform(texts_train)
    hashed_test = hv.transform(texts_test)

    hashed_train, hashed_test = np.array(hashed_train.todense()), np.array(hashed_test.todense())
    return hashed_train, hashed_test
