#!/usr/bin/envp python
# coding: utf-8
import os
import operator
import argparse

import numpy as np
import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle

from sklearn import (
    cross_validation,
    decomposition,
    grid_search,
    metrics,
    pipeline,
    preprocessing,
    svm,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Classify handwriten numerals.')
    parser.add_argument('--load', default=False, action='store_true')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--partial', default=True, action='store_true', dest='partial')
    g.add_argument('--full', action='store_false', dest='partial')
    args = parser.parse_args()
    return args


def load_best_model():
    model_name = sorted(
        [(filename, int(filename.lstrip('model-0.').rstrip('.pickle')),) for filename in os.listdir('models')],
        key=operator.itemgetter(1),
        reverse=True,
    )[0][0]

    model = None
    with open(os.path.join('models', model_name), 'rb') as fh:
        model = pickle.load(fh)

    return model


def save_model(model, score):
    with open('models/model-{0:0.3f}.pickle'.format(score), 'wb') as fh:
        pickle.dump(model, fh, protocol=2)


def load_train_data(partial=True):

    filename = 'train_partial.csv' if partial else 'train.csv'
    path = os.path.join('data', filename)

    data = pd.read_csv(path).astype(np.float64)
    X, y = data[[col for col in data.columns if not 'label' in col]], data['label']
    return (X, y)


def load_test_data(partial=True):

    filename = 'test_partial.csv' if partial else 'test.csv'
    path = os.path.join('data', filename)

    data = pd.read_csv(path).astype(np.float64)
    X = data[[col for col in data.columns if not 'label' in col]]
    return X


def train_model(X, y, persist_model=True):
    # Test Train split
    (
        X_train, X_test,
        y_train, y_test,
    ) = cross_validation.train_test_split(X, y)

    estimators = [
        ('scaler', preprocessing.MinMaxScaler()),
        ('pca', decomposition.PCA()),
        ('svm', svm.SVC())
    ]

    pipe = pipeline.Pipeline(estimators)

    parameters = {
        'pca__n_components': [32, 64],
        'svm__C': [10, 100, 1000],
        'svm__kernel': ['rbf', 'poly'],
        'svm__gamma': [0.0, 0.001, 0.01]
    }
    gs = grid_search.GridSearchCV(estimator=pipe, param_grid=parameters, verbose=2, n_jobs=-1)
    gs.fit(X_train, y_train)
    print gs.best_score_
    print gs.best_estimator_

    model = gs.best_estimator_
    prediction = model.predict(X_test)
    print metrics.accuracy_score(y_test, prediction)

    save_model(model, gs.best_score_) if persist_model else None

    return model


def predict(model=None, X=None):

    prediction = model.predict(X).astype(np.int)
    s = pd.Series(data=prediction, index=np.arange(1, len(prediction) + 1), name='Label')
    s.to_csv(path='prediction.csv', index=True, header=True, index_label='ImageId')

# Runs
if __name__ == "__main__":

    args = parse_args()

    model = load_best_model() if args.load else train_model(*load_train_data(partial=args.partial))

    predict(model=model, X=load_test_data())
