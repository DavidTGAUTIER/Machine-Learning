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
    

    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        # parametres Prior
        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

        # Permet la simulation de la distribution du chi-deux à echelle inversée
        # On Suppose que la variance est distribuée selon cette distribution.
        # Référence:https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution

    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return sigma_sq


    def fit(self, X, y):

        # Si transformation polynomiale
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        n_samples, n_features = np.shape(X)

        X_X = X.T.dot(X)

        # Approximation des moindres carrés de beta
        beta_hat = np.linalg.pinv(X_X).dot(X.T).dot(y)

        # Les paramètres (postérior) peuvent être déterminés analytiquement puisque nous 
        # supposons conjuguer les priors pour les probabilités.

        # Normal prior / likelihood => Normal posterior
        # prior (avant) Normale / probabilité = posterior (après) Normale
        mu_n = np.linalg.pinv(X_X + self.omega0).dot(X_X.dot(beta_hat)+self.omega0.dot(self.mu0))
        omega_n = X_X + self.omega0
        # Scaled inverse chi-squared prior / likelihood => Scaled inverse chi-squared posterior
        # prior (avant) chi-2 echelle inv. / probabilité = posterior (après) chi-2 echelle inv.
        nu_n = self.nu0 + n_samples
        sigma_sq_n = (1.0/nu_n)*(self.nu0*self.sigma_sq0 + \
            (y.T.dot(y) + self.mu0.T.dot(self.omega0).dot(self.mu0) - mu_n.T.dot(omega_n.dot(mu_n))))

        # Simulation des valeurs des parametres pour n_draws
        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self._draw_scaled_inv_chi_sq(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:,0], cov=sigma_sq*np.linalg.pinv(omega_n))
            # Sauvegardes des parametres draws
            beta_draws[i, :] = beta

        # Selection de la moyenne des variables simulées comme celles utilisées pour faire les predictions
        self.w = np.mean(beta_draws, axis=0)

        # Limite inférieure et supérieure de l'intervalle crédible
        l_eti = 50 - self.cred_int/2
        u_eti = 50 + self.cred_int/2
        self.eti = np.array([[np.percentile(beta_draws[:,i], q=l_eti), np.percentile(beta_draws[:,i], q=u_eti)] \
                                for i in range(n_features)])
        

    def predict(self, X, eti=False):

        # Si transformation polynomiale
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        y_pred = X.dot(self.w)
        # Si les limites inférieure et supérieure pour les 95% (eti=True)
        # alors l'intervalle à queue égale (equal tail interval) doit être renvoyé
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return y_pred, y_lower_pred, y_upper_pred

        return y_pred
