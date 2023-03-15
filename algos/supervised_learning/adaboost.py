
from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


from algos.utils.data_manipulation import train_test_split
from algos.utils.data_operation import accuracy_score
from algos.utils.plot import Plot

# Decision stump (séparation de décision gauche-droite) utilisé comme classifier faible dans cet implémentation d'Adaboost
class DecisionStump():
    def __init__(self):
        # Détermine si le sample doit être classé comme -1 ou 1 en fonction du seuil donné
        self.polarity = 1
        # L'index de la feature utilisée pour effectuer la classification
        self.feature_index = None
        # La valeur seuil à laquelle la feature doit être mesurée
        self.threshold = None
        # Valeur indicative de l'accuracy du classifier
        self.alpha = None



class Adaboost():
    """Méthode de Boosting qui utilise un certain nombre de classifiers faibles par des methodes d'Ensemble 
      (utiliser plusieurs algorithmes d'apprentissage pour obtenir de meilleures performances prédictives) 
      pour créer un classifier fort. 
      Cette mise en œuvre utilise la  décision souches, qui est un arbre de décision à un niveau.
    Parametres:
    -----------
    n_clf: int
        Le nombre de classifiers faibles qui seront utilisés.. 
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Initialiser les poids à 1 / N
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        # Itérer à travers les classificateurs
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Erreur minimale donnée pour l'utilisation d'un certain seuil 
            # de valeur de la feature pour prédire le label du sample
            min_error = float('inf')

            # Iterer sur chaque valeur unique de la feature et voir quelle valeur
            # fait le meilleur seuil pour prédire y
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Essayez chaque valeur de fonctionnalité unique comme seuil
                for threshold in unique_values:
                    p = 1
                    # Définir toutes les predictions sur '1' initialement
                    prediction = np.ones(np.shape(y))
                    # Labeliser les samples dont les valeurs sont inférieures 
                    # au seuil comme '-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    # Error = somme des poids des samples mal classés
                    error = sum(w[y != prediction])
                    
                    # Si error est > 50%, nous renversons la polarité de sorte que
                    # les samples classés 0 soient classés comme 1, et vice versa
                    # Par exemple, error = 0.8 => (1 - error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Si ce seuil a entraîné la plus petite erreur
                    # nous enregistrons la configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # Calculer l'alpha utilisé pour mettre à jour les poids du sample
            # Alpha est également une approximation de la compétence de ce classifier
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # Définir toutes les predictions sur '1' initialement
            predictions = np.ones(np.shape(y))
            # Les indices où les valeurs de sample sont inférieures au seuil
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Labeliser-les comme '-1'
            predictions[negative_idx] = -1
            # Calculer les nouveaux poids
            # Les samples mal classés obtiennent des poids plus importants  
            # et les samples correctement classés des poids plus petits
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalizer à 1
            w /= np.sum(w)

            # Sauvegarder le classifier
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # POur chaque classifier => labeliser les samples
        for clf in self.clfs:
            # Définir toutes les predictions sur '1' initialement
            predictions = np.ones(np.shape(y_pred))
            # Les indices où les valeurs de samples sont inférieures au seuil
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Labeliser-les comme '-1'
            predictions[negative_idx] = -1
            # Ajouter des predictions pondérées par les classifiers alpha
            # (alpha indiquant la compétence du  classifier)
            y_pred += clf.alpha * predictions

        # Retourne le signe de la somme des predictions
        y_pred = np.sign(y_pred).flatten()

        return y_pred


def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # Changer  les labels en {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Classification Adaboost avec les 5 classifiers faibles
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Reduire les dimensions en 2d en utilisant pca and plot les resultats
    Plot().plot_in_2d(X_test, y_pred, title="Adaboost", accuracy=accuracy)


if __name__ == "__main__":
    main()