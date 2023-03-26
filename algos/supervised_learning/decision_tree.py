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
class DecisionTree(object):
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
        """ Méthode récursive qui construit l'arbre de décision et divise X et y
            respectivement sur la feature de X qui (basé sur l'impureté) sépare 
            le mieux les données"""
        
        largest_impurity = 0
        best_criteria = None    # index de la feature et le seuil
        best_sets = None        # Subsets des données

        # Check si l'expansion de y est nécessaire
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Ajoute y comme dernière colonne de X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calcule impurity pour chaque feature
            for feature_i in range(n_features):
                # Toutes les valeurs de feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Itérer sur toutes les valeurs uniques de la colonne de feature_i
                # et calculer l'impureté
                for threshold in unique_values:
                    # Divise X et y selon si la valeur de la feature de X à l'index feature_i
                    # atteint le seuil
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Sélectionnez les valeurs de y des deux sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calcule l'impurity
                        impurity = self._impurity_calculation(y, y1, y2)
                        
                        # Si ce seuil a entraîné un gain d'information plus élevé qu'auparavant
                        # on sauvegarde la valeur du seuil et l'index de la feature
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X du left subtree
                                "lefty": Xy1[:, n_features:],   # y du left subtree
                                "rightX": Xy2[:, :n_features],  # X du right subtree
                                "righty": Xy2[:, n_features:]   # y du right subtree
                                }
                            
        if largest_impurity > self.min_impurity:
            # On construit des sous-arbres pour les branches droite et gauche
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # Quand on arrive à la feuille => on determine la valeur
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """Fait une recherche recursive descendante de l'arbre et fait une prediction de 
           l'echantillon de donnée par la valeur de la feuille que nous trouvons"""

        if tree is None:
            tree = self.root

        # Si nous avons une valeur (c'est-à-dire que nous sommes à une feuille)
        # => retourne la valeur comme prédiction
        if tree.value is not None:
            return tree.value

        # Choisit la feature que nous testerons
        feature_value = x[tree.feature_i]

        # Détermine si nous suivrons la branche gauche ou droite
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Sous-arbre de test
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classer les échantillons un par un et retourner l'ensemble des labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Print recursivement le decision tree """
        if not tree:
            tree = self.root

        # Si on atteint une feuille = > print le label
        if tree.value is not None:
            print (tree.value)
        # Sinon, on va plus loin dans l'arbre
        else:
            # On print le test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # On print le scenario True
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # On print le scenario False
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class XGBoostRegressionTree(DecisionTree):
    """
    Regression Tree pour XGBoost
    - Reference -
    http://xgboost.readthedocs.io/en/latest/model.html
    """

    def _split(self, y):
        """
        y contient :
            * y_true dans la moitié gauche de la colonne du milieu
            * y_pred dans la moitié droite de la colonne du milieu
        Split et retourne les deux matrices
        """
        col = int(np.shape(y)[1]/2)
        y_true = y[:, :col]
        y_pred = y[:, col:]

        return y_true, y_pred
    

    def _gain(self, y_true, y_pred):
        nominator = np.power((y_true * self.loss_gradient(y_true, y_pred)).sum(), 2)
        denominator = self.loss.hess(y_true, y_pred).sum()

        return 0.5 * (nominator / denominator)