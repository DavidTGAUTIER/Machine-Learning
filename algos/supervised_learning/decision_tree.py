from __future__ import division, print_function
import numpy as np

from algos.utils.data_manipulation import train_test_split, standardize, divide_on_feature
from algos.utils.data_operation import mean_squared_error, accuracy_score, calculate_entropy, calculate_variance
from algos.utils.plot import Plot



class DecisionNode():
    """
    Classe qui représente un noeud de décision ou une feuille dans l'arbre
    de décision.

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
        self.false_branch = false_branch    # Sous-arbre droite

# Super classe de RegressionTree et ClassificationTree
class DecisionTree(objet):
    """
    Super classe de RegressionTree et ClassificationTree.

    Paramètres:
    -----------
    min_samples_split: int
        Le nombre minimum d'échantillons nécessaires pour faire une séparation
        lors de la construction d'un arbre.
    min_impurity: float
        L'impureté minimale requise pour diviser davantage l'arbre.
    max_depth: int
        La profondeur maximale d'un arbre.
    loss: fonction
        Loss function utilisée pour les modèles de Gradient Boosting pour 
        calculer l'impureté.
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        # Noeud root du decision tree
        self.root = None  
        # Minimum n de samples pour justifier le split
        self.min_samples_split = min_samples_split
        # L'impureté minimum pour justifier le split
        self.min_impurity = min_impurity
        # La profondeur maximale pour faire pousser l'arbre
        self.max_depth = max_depth
        # Fonction pour calculer impurity 
        # Classification => info gain
        # Regression     => variance reduction
        self._impurity_calculation = None
        # Fonction pour determiner la prediction de y 
        # quand on arrive à une feuille
        self._leaf_value_calculation = None
        # Si y est one-hot encoded (multi-dim) ou non (one-dim)
        self.one_dim = None
        # Si Gradient Boost
        self.loss = loss  

    def fit(self, X, y, loss=None):
        """ Construire le decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss=None

    def _build_tree(self, X, y, current_depth=0):
        