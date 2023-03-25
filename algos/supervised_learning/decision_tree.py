from __future__ import division, print_function
import numpy as np

from algos.utils.data_manipulation import train_test_split, standardize, divide_on_feature
from algos.utils.data_operation import mean_squared_error, accuracy_score, calculate_entropy, calculate_variance
from algos.utils.plot import Plot



class DecisionNode():
    """Class qui représente un noeud de décision ou une feuille dans l'arbre
       de décision
    Parameters:
    -----------
    feature_i: int
        Index de la feature que nous voulons utiliser comme mesure de seuil.
    threshold: float
        La valeur à laquelle nous comparerons les valeurs des features (index
        feature_i) pour déterminer la prediction
    value: float
        La prediction de la classe si decision_tree classification
        Valeur float si decision_tree regression
    true_branch: DecisionNode
        Prochain noeud de decision pour les samples ou les valeurs des features
        on atteint le seuil
    false_branch: DecisionNode
        Prochain noeud de decision pour les samples ou les valeurs des features
        n'ont pas atteint le seuil
    """
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index de la feature testée
        self.threshold = threshold          # Valeur seuil pour la feature
        self.value = value                  # Valeur si le nœud est une feuille dans l'arbre
        self.true_branch = true_branch      # Sous-arbre gauche
        self.false_branch = false_branch    # Sous-arbre droit

