from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys



def mean_squared_error(y_true, y_pred):
    """Retourne l'erreur quadratique moyenne (mse) entre y_true et y_pred"""
    mse = np.mean(np.power(y_true - y_pred), 2)
    return mse



def accuracy_score(y_true, y_pred):
    """ Compare y_true avec y_pred et retourne l'accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy



def calculate_covariance_matrix(X, Y=None):
    """ Calcul de la matrice de covariance pour le dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)



def calculate_correlation_matrix(X, Y=None):
    """ Calcul de la matrice de correlation pour le dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)



def calculate_entropy(y):
    """ Calcul de l'entropy de la target y 
        https://fr.wikipedia.org/wiki/Entropie_de_Shannon"""
    # log2 X est le logarithme binaire
    # logarithme en base d'un entier n
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0

    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)

    return entropy



def calculate_variance(X):
    """ Retourne la variance des features du dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance