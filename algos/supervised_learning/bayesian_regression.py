from __future__ import print_function, division
import numpy as np
from scipy.stats import chi2, multivariate_normal

from algos.utils.data_manipulation import train_test_split, polynomial_features
from algos.utils.data_operation import mean_squared_error
from algos.utils.plot import Plot

"""Rappel:
Dans le contexte du théorème de Bayes:

le "prior" (ou "a priori" = "avant") est la probabilité qu'un événement se produise, avant que toute
information supplémentaire ne soit prise en compte. représente notre croyance initiale sur la probabilité
d'un événement, avant d'avoir examiné les données ou l'information disponible

Le "posterior" (ou "a posteriori" = "après") est la probabilité qu'un événement se produise, après avoir
pris en compte l'information supplémentaire. Il est la probabilité révisée de l'événement, après avoir 
examiné les données ou l'information disponible."""


class BayesianRegression:
    """Modèle de regression Bayesienne. Si l'argument 'poly_degree' est spécifié, 
    les features seront transformées en en fonction polynomiale, ce qui permettra 
    de réaliser la regression polynomiale. Nous partons de l'hypothèse de Normalité
    pour les probabilités connues (prior) des poids, et de l'hypothèse de Chi-deux
    à echelle inversée (scaled inverse chi-squared) pour les probabilités connues
    de la variance des poids.
    Paramètres:
    -----------
    n_draws: float
        Le nombre de tirages simulés (posterior) des paramètres.
    mu0: array
        Les valeurs moyennes de la distribution normale connue(prior) des paramètres.
    oméga0: tableau
        La matrice de précision de la distribution normale connue(prior) des paramètres.
    nu0: float
        Les degrés de liberté de la distribution au chi-deux à echelle inversée (prior)
    sigma_sq0: flotteur
        Paramètre 'scale' de la distribution au chi-deux à echelle inversée (prior)
    poly_degré: int
        Le degré polynomial avec lequel les features doivent être transformées. Permet
        la régression polynomiale.
    cred_int: float
        L'intervalle crédible (ETI dans cet impl.) .95 => 95% d'intervalle crédible du 
        (postérior) des paramètres.
    Référence:
        https://github.com/mattiasvillani/BayesLearnCourse/raw/master/Slides/BayesLearnL5.pdf"""