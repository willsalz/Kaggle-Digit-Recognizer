#!/usr/bin/envp python
# coding: utf-8
import os
import operator
import numpy as np
import pandas as pd
from StringIO import StringIO

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


# Load Data
try:
    model = load_best_model()
except pickle.PickleError:
    data = pd.read_csv('./data/train.csv').astype(np.float64)
    X, y = data[[col for col in data.columns if not 'label' in col]], data['label']

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

    model = pipeline.Pipeline(estimators)

    parameters = {
        'pca__n_components': [32, 64],
        'svm__C': [10, 100, 1000],
        'svm__kernel': ['rbf', 'poly'],
        'svm__gamma': [0.0, 0.001, 0.01]
    }

    if __name__ == "__main__":
        gs = grid_search.GridSearchCV(estimator=model, param_grid=parameters, verbose=2, n_jobs=-1)
        gs.fit(X_train, y_train)
        print gs.best_score_
        print gs.best_estimator_

        model = gs.best_estimator_
        prediction = model.predict(X_test)
        print metrics.accuracy_score(y_test, prediction)

        with open('models/model-{0:0.3f}.pickle'.format(gs.best_score_), 'wb') as fh:
            pickle.dump(model, fh, protocol=2)
finally:
    data = pd.read_csv('./data/test.csv').astype(np.float64)
    X = data[[col for col in data.columns if not 'label' in col]]
    prediction = model.predict(X).astype(np.int)
    s = pd.Series(data=prediction, index=np.arange(1, len(prediction) + 1), name='Label')
    s.to_csv(path='prediction.csv', index=True, header=True, index_label='ImageId')
