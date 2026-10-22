"""
llre.py -- Paso 5: Lasso-Logistic Regression Ensemble (LLRE)
=============================================================
Implementacion fiel al Algoritmo 2 (Tabla 2) de Wang, Xu & Zhou (2015):

  1. Agrupar (K-means) los ejemplos de la clase MAYORITARIA en k subgrupos.
  2. Conservar L = 0.75 * k subgrupos (los de mayor tamano).
  3. Para cada subgrupo i:
       BS_i  = subgrupo_i  +  muestra bootstrap de la clase minoritaria
       entrenar un Lasso-logistico C_i sobre BS_i
       OBS_i = D - BS_i  (todo el entrenamiento que NO esta en BS_i)
       elegir lambda con el mejor AUC sobre OBS_i y guardar p_i = ese AUC
       w_i = 1 / (1 + exp(-p_i))                                   (Ec. 4)
  4. Prediccion: promedio ponderado de las probabilidades de los C_i;
     etiqueta = argmax (equivale a umbral 0.5, pues los BS estan balanceados).
  5. Importancia por ocurrencia (Ec. 5): # de modelos donde beta_j != 0.

SUPUESTOS de implementacion (el articulo no los fija):
  - El articulo usa glmnet (R). Aqui: sklearn LogisticRegression con
    penalty='l1' y solver='liblinear'; la rejilla de lambda es geometrica en
    C = 1/lambda, desde l1_min_c (equivalente al lambda_max de glmnet), con
    parada cuando el AUC deja de mejorar; tol = 1e-3 por costo de computo.
  - "Conservar L = 0.75k subgrupos": el texto habla de descartar los de
    tamano menor al cuantil 75 % pero tambien de L = 0.75k; se adopta L = ceil(0.75k)
    de los subgrupos mas grandes.
  - Tamano del bootstrap minoritario = n de la clase minoritaria.
  - K-means: MiniBatchKMeans sobre las variables estandarizadas (por velocidad).
"""

import math
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import l1_min_c


warnings.filterwarnings("ignore", message=".*penalty.*")       # sklearn >= 1.8
warnings.filterwarnings("ignore", message=".*l1_ratio.*")


def _ajustar_base(X, y, bs, obs, n_cs, razon, seed):
    """
    Entrena un Lasso-logistico para una trayectoria creciente de C sobre BS y
    devuelve el mejor modelo segun el AUC en OBS.

    La trayectoria parte de c0 = l1_min_c (el menor C con algun coeficiente
    != 0, equivalente al lambda_max de glmnet) y avanza con un paso geometrico
    de razon**(1/(n_cs-1)) por punto (n_cs puntos cubren `razon` veces c0).
    Si el mejor punto queda al final, la trayectoria se prolonga; se detiene
    cuando el AUC no mejora en 2 puntos consecutivos (minimo 5 puntos,
    maximo 3 * n_cs).
    """
    c0 = l1_min_c(X[bs], y[bs], loss="log")
    paso = razon ** (1.0 / (n_cs - 1))
    mejor = (-1.0, None, None, None)
    for j in range(3 * n_cs):
        C = c0 * paso ** j
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, tol=1e-3,
                               max_iter=200, random_state=seed)
        m.fit(X[bs], y[bs])
        auc = roc_auc_score(y[obs], m.decision_function(X[obs]))
        if auc > mejor[0]:
            mejor = (auc, m, C, j)
        if j >= 4 and j - mejor[3] >= 2:
            break
    return mejor


class LLRE:
    def __init__(self, k=14, frac_keep=0.75, n_cs=10, razon=1e4,
                 n_jobs=2, random_state=0):
        self.k, self.frac_keep = k, frac_keep
        self.n_cs, self.razon = n_cs, razon
        self.n_jobs, self.random_state = n_jobs, random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        idx_min, idx_may = np.where(y == 1)[0], np.where(y == 0)[0]

        km = MiniBatchKMeans(n_clusters=self.k, n_init=3, batch_size=4096,
                             random_state=self.random_state).fit(X[idx_may])
        tamanos = np.bincount(km.labels_, minlength=self.k)
        L = math.ceil(self.frac_keep * self.k)
        elegidos = np.argsort(tamanos)[::-1][:L]        # los L subgrupos mas grandes
        self.tamanos_, self.elegidos_ = tamanos, elegidos
        self.n_min_ = len(idx_min)

        n = len(y)
        tareas = []
        for g in elegidos:
            maj = idx_may[km.labels_ == g]
            mino = rng.choice(idx_min, size=len(idx_min), replace=True)   # bootstrap
            bs = np.concatenate([maj, mino])
            en_bs = np.zeros(n, dtype=bool)
            en_bs[bs] = True
            tareas.append((bs, np.where(~en_bs)[0]))                      # OBS = D - BS

        res = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_ajustar_base)(X, y, bs, obs, self.n_cs, self.razon, self.random_state)
            for bs, obs in tareas)

        self.coefs_ = [m.coef_.ravel() for _, m, _, _ in res]
        self.intercepts_ = [float(m.intercept_[0]) for _, m, _, _ in res]
        self.auc_obs_ = [auc for auc, _, _, _ in res]
        self.lambdas_ = [1.0 / C for _, _, C, _ in res]
        self.pos_rejilla_ = [j for _, _, _, j in res]      # 0 = mas penalizado
        self.coefs_ = np.vstack(self.coefs_)
        self.intercepts_ = np.array(self.intercepts_)
        self.pesos_ = 1.0 / (1.0 + np.exp(-np.array(self.auc_obs_)))        # Ec. (4)
        return self

    def predict_proba(self, X):
        """Promedio ponderado de las probabilidades de los L modelos base."""
        X = np.asarray(X, dtype=float)
        z = X @ self.coefs_.T + self.intercepts_            # (n, L)
        p = 1.0 / (1.0 + np.exp(-z))
        return (p * self.pesos_).sum(axis=1) / self.pesos_.sum()

    def prior_bs_(self):
        """Prevalencia media de la clase 1 en los conjuntos balanceados BS."""
        n_may = self.tamanos_[self.elegidos_]
        return float(np.mean(self.n_min_ / (self.n_min_ + n_may)))

    def predict_proba_corregido(self, X, prior):
        """
        Analisis complementario (no esta en el articulo): corrige el desplazamiento
        de prior causado por entrenar con conjuntos balanceados. Suma al logit del
        promedio ponderado log(prior/(1-prior)) - log(p_bs/(1-p_bs)), donde `prior`
        es la prevalencia de la clase 1 en el entrenamiento completo y p_bs la
        prevalencia media en los BS. Es una correccion aproximada porque el
        ensamble promedia probabilidades y no logits.
        """
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p))
        p_bs = self.prior_bs_()
        logit += np.log(prior / (1 - prior)) - np.log(p_bs / (1 - p_bs))
        return 1.0 / (1.0 + np.exp(-logit))

    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X) >= umbral).astype(int)

    def resumen_base(self):
        """Tabla de los L modelos base: subgrupo, tamano, lambda, AUC en OBS, peso."""
        import pandas as pd
        return pd.DataFrame({
            "subgrupo": self.elegidos_,
            "n_mayoritaria": self.tamanos_[self.elegidos_],
            "lambda": np.round(self.lambdas_, 4),
            "pos_rejilla": self.pos_rejilla_,
            "AUC_OBS": np.round(self.auc_obs_, 4),
            "peso": np.round(self.pesos_, 4),
            "vars_no_cero": (self.coefs_ != 0).sum(axis=1),
        })

    def importancia_ocurrencia(self, nombres):
        """VI_occu (Ec. 5): en cuantos modelos base cada variable es != 0."""
        occ = (self.coefs_ != 0).sum(axis=0)
        return dict(sorted(zip(nombres, occ), key=lambda t: -t[1]))
