"""
evaluate.py -- Paso 5: corridas de LLRE, LLR, RF y CART (AUC y F-measure)
==========================================================================
Protocolo del articulo: particion aleatoria 80 % / 20 %, 30 corridas, dos
conjuntos de variables (originales / derivadas), metricas AUC y F-measure.

Configuracion reportada de los comparadores (se usa tal cual):
  RF   : ntree = 500, mtry = 3
  CART : criterio Gini (rpart)
  LLR  : glmnet, family = binomial, type.measure = auc, alpha = 1
  LLRE : k = 14

SUPUESTOS de traduccion R -> sklearn (el articulo usa R):
  - CART (rpart) por defecto NO crece el arbol completo: cp = 0.01, minsplit = 20,
    maxdepth = 30. Se imita con min_impurity_decrease = 0.01 * Gini(raiz).
  - LLR: LogisticRegressionCV (L1, liblinear, scoring = roc_auc); glmnet usa
    10 folds y 100 valores de lambda; aqui 3 folds y 10 valores de C, desde
    l1_min_c hasta 1e4 veces ese valor, con tol = 1e-3 por costo de computo.
    `pos_rejilla` registra la posicion del lambda elegido (0 = mas penalizado);
    si coincide con el ultimo valor, la rejilla es demasiado corta.
  - F-measure de la clase positiva con umbral 0.5 para todos los modelos
    (etiqueta predicha por el propio clasificador).
Todo el codigo es independiente del origen de los datos: recibe un DataFrame.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c
from sklearn.tree import DecisionTreeClassifier

from features import FeatureBuilder
from llre import LLRE

warnings.filterwarnings("ignore", message=".*penalty.*")       # sklearn >= 1.8
warnings.filterwarnings("ignore", message=".*l1_ratio.*")

TARGET = "SeriousDlqin2yrs"


def _modelos(Xs_tr, y_tr, seed, k, cv_llr, n_jobs, n_cs):
    p = y_tr.mean()
    gini_raiz = 2 * p * (1 - p)
    c0 = l1_min_c(Xs_tr, y_tr, loss="log")   # equivalente al lambda_max de glmnet
    return {
        "LLRE": LLRE(k=k, n_cs=n_cs, n_jobs=n_jobs, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=500, max_features=3,
                                     n_jobs=n_jobs, random_state=seed),
        "LLR": LogisticRegressionCV(Cs=c0 * np.geomspace(1.0, 1e4, 10), cv=cv_llr,
                                    penalty="l1", solver="liblinear", tol=1e-3,
                                    scoring="roc_auc", max_iter=200, n_jobs=n_jobs,
                                    random_state=seed),
        "CART": DecisionTreeClassifier(criterion="gini", min_samples_split=20,
                                       max_depth=30,
                                       min_impurity_decrease=0.01 * gini_raiz,
                                       random_state=seed),
    }


def una_corrida(df, modo, run, k=14, cv_llr=3, n_jobs=2, n_cs=10,
                builder_kwargs=None, modelos=("LLRE", "RF", "LLR", "CART"),
                semilla_base=1000, etiqueta=None):
    """Una corrida: split aleatorio 80/20 -> features -> 4 modelos -> AUC y F1."""
    seed = semilla_base + run
    tr, te = train_test_split(df, test_size=0.2, random_state=seed)
    fb = FeatureBuilder(modo=modo, **(builder_kwargs or {}))
    X_tr, X_te = fb.fit_transform(tr), fb.transform(te)
    y_tr, y_te = tr[TARGET].values, te[TARGET].values
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)

    filas = []
    for nombre, mod in _modelos(Xs_tr, y_tr, seed, k, cv_llr, n_jobs, n_cs).items():
        if nombre not in modelos:
            continue
        lineal = nombre in ("LLRE", "LLR")
        A, B = (Xs_tr, Xs_te) if lineal else (X_tr.values, X_te.values)
        t0 = time.time()
        mod.fit(A, y_tr)
        proba = mod.predict_proba(B)
        proba = proba if proba.ndim == 1 else proba[:, 1]
        pred = (proba >= 0.5).astype(int)
        pos, n_rej = np.nan, np.nan          # posicion del lambda elegido en la rejilla
        if nombre == "LLRE":
            pos, n_rej = float(np.mean(mod.pos_rejilla_)), 3 * n_cs
        elif nombre == "LLR":
            pos = int(np.argmin(np.abs(np.log(mod.Cs_) - np.log(mod.C_[0]))))
            n_rej = len(mod.Cs_)
        filas.append(dict(run=run, variables=etiqueta or modo, clasificador=nombre,
                          auc=roc_auc_score(y_te, proba),
                          f1=f1_score(y_te, pred, zero_division=0),
                          n_pred_pos=int(pred.sum()),
                          pos_rejilla=pos, n_rejilla=n_rej,
                          segundos=round(time.time() - t0, 1)))
    return filas


def correr_experimento(df, n_runs, destino, modos=("original", "derived"), **kw):
    """Corre y guarda de forma incremental (reanudable) en `destino` (CSV)."""
    destino = Path(destino)
    hechos = set()
    if destino.exists():
        previo = pd.read_csv(destino)
        hechos = set(zip(previo["run"], previo["variables"]))
    for run in range(n_runs):
        for modo in modos:
            if (run, modo) in hechos:
                continue
            filas = una_corrida(df, modo, run, **kw)
            pd.DataFrame(filas).to_csv(destino, mode="a", index=False,
                                       header=not destino.exists())
            print(f"run {run} / {modo}: " +
                  ", ".join(f"{f['clasificador']} AUC={f['auc']:.3f} F1={f['f1']:.3f}"
                            for f in filas), flush=True)
    return pd.read_csv(destino)


def correr_extension(df, n_runs, ks, destino, **kw):
    """
    Extension (Paso 7): variables discretizadas en k rangos, k en `ks`.
    Usa las mismas particiones (semillas) que el Paso 5 para que la
    comparacion sea sobre los mismos datos. Guarda de forma incremental.
    """
    destino = Path(destino)
    hechos = set()
    if destino.exists():
        previo = pd.read_csv(destino)
        hechos = set(zip(previo["run"], previo["variables"]))
    for run in range(n_runs):
        for k in ks:
            etiqueta = f"discretized_k{k}"
            if (run, etiqueta) in hechos:
                continue
            filas = una_corrida(df, "discretized", run, builder_kwargs={"n_bins": k},
                                etiqueta=etiqueta, **kw)
            pd.DataFrame(filas).to_csv(destino, mode="a", index=False,
                                       header=not destino.exists())
            print(f"run {run} / {etiqueta}: " +
                  ", ".join(f"{f['clasificador']} AUC={f['auc']:.3f} F1={f['f1']:.3f}"
                            for f in filas), flush=True)
    return pd.read_csv(destino)
