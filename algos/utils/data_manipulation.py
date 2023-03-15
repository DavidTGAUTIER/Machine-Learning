from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split les data en train set et test set """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split le train set / test set en fonction du ratio précisé (test_size)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

