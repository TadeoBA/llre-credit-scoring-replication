"""
simulate_data.py -- Paso 4: simulacion de datos
================================================
Simula un dataset equivalente al de Wang, Xu & Zhou (2015),
"Large Unbalanced Credit Scoring Using Lasso-Logistic Regression Ensemble",
PLoS ONE 10(2): e0117844 (dataset publico "Give Me Some Credit").

Principio: lo REPORTADO por el articulo se reproduce exactamente; todo lo
demas es un SUPUESTO, marcado como tal en el codigo y en el README (Paso 3).

REPORTADO (se reproduce exacto)
  - N = 150,000; clase 0 = 139,974; clase 1 = 10,026 (razon ~14:1)
  - 10 variables predictoras + objetivo SeriousDlqin2yrs
  - age: min = 0, max = 109, media = 52.3

SUPUESTO (no reportado; ver README seccion 3.3)
  - Forma de la distribucion de cada variable
  - Desviacion estandar de age
  - % de faltantes: MonthlyIncome ~20%, NumberOfDependents ~2.6%
  - Codigos especiales 96/98 en las variables de atraso (~0.18% de filas)
  - Magnitud de la relacion de cada variable con el impago (el articulo solo
    da la direccion cualitativa via importancia de variables)
  - Variables independientes entre si dado la clase

Toda la aleatoriedad sale de un unico generador con semilla fija.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- REPORTADO
N_NO_DEFAULT = 139_974
N_DEFAULT = 10_026
N_TOTAL = N_NO_DEFAULT + N_DEFAULT          # 150,000
AGE_MEAN, AGE_MIN, AGE_MAX = 52.3, 0, 109
TARGET = "SeriousDlqin2yrs"

# ----------------------------------------------------------------- SUPUESTOS
RANDOM_STATE = 42
AGE_SD = 14.8                 # supuesto
AGE_SHIFT_DEFAULT = -2.0      # supuesto: los morosos son ligeramente mas jovenes
P_MISSING_INCOME = 0.20       # supuesto (descripcion publica del dataset)
P_MISSING_DEPENDENTS = 0.026  # supuesto (descripcion publica del dataset)
P_SPECIAL_CODE = 0.0018       # supuesto: filas con codigo 96/98 en atrasos

COLUMNAS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def simular_objetivo(rng):
    """Vector y con los conteos EXACTOS reportados, en orden aleatorio."""
    y = np.array([0] * N_NO_DEFAULT + [1] * N_DEFAULT)
    rng.shuffle(y)
    return y


def simular_age(rng, y):
    """
    age: Normal(mu_clase, AGE_SD) truncada a [21, 109] (+ un registro en 0).
    - REPORTADO: media 52.3, min 0, max 109.
    - SUPUESTO: sd y el desplazamiento por clase.
    El 0 se fuerza en UN registro porque el articulo reporta min = 0 (un
    error de captura del dataset original); el 109 tambien se fuerza en uno.
    El centro de la normal se calibra para que la media final sea 52.3.
    """
    z = rng.standard_normal(len(y))          # mismo ruido en cada iteracion
    loc = AGE_MEAN
    for _ in range(30):                      # calibracion de la media
        age = np.clip(np.round(loc + AGE_SHIFT_DEFAULT * y + AGE_SD * z), 21, AGE_MAX)
        age[np.argmax(z)] = AGE_MAX          # un 109 exacto
        idx0 = np.where(y == 0)[0][0]
        age[idx0] = AGE_MIN                  # un 0 exacto
        loc += AGE_MEAN - age.mean()
    return age.astype(int)


def simular_predictoras(rng, y):
    """Las 9 variables restantes, condicionadas a la clase (todo SUPUESTO)."""
    n = len(y)
    r = y.astype(float)

    # Utilizacion de credito: Beta sesgada + cola extrema (>1)
    util = rng.beta(0.6, 2.2, n) + 0.18 * r * rng.random(n)
    cola = rng.random(n) < 0.02
    util[cola] *= rng.uniform(2, 5, cola.sum())
    util = np.clip(util, 0, 6)

    # Atrasos (conteos raros; mas altos en la clase 1)
    a3059 = rng.poisson(0.15 + 0.9 * r)
    a90 = rng.poisson(0.05 + 0.6 * r)
    a6089 = rng.poisson(0.04 + 0.4 * r)

    # Codigos especiales 96 / 98 en las tres variables de atraso (mismas filas)
    esp = rng.random(n) < P_SPECIAL_CODE
    cod = rng.choice([96, 98], size=esp.sum())
    a3059[esp], a90[esp], a6089[esp] = cod, cod, cod

    debt = np.clip(rng.lognormal(-0.7 + 0.35 * r, 1.1, n), 0, 15)

    income = rng.lognormal(8.3 - 0.25 * r, 0.75, n).astype(float)
    income[rng.random(n) < P_MISSING_INCOME] = np.nan

    lineas = rng.poisson(8.5 - 1.2 * r)
    inmob = rng.poisson(1.0 - 0.2 * r)

    dep = rng.poisson(0.76, n).astype(float)           # sin relacion con y
    dep[rng.random(n) < P_MISSING_DEPENDENTS] = np.nan

    return {
        "RevolvingUtilizationOfUnsecuredLines": util,
        "NumberOfTime30-59DaysPastDueNotWorse": a3059,
        "DebtRatio": debt,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": lineas,
        "NumberOfTimes90DaysLate": a90,
        "NumberRealEstateLoansOrLines": inmob,
        "NumberOfTime60-89DaysPastDueNotWorse": a6089,
        "NumberOfDependents": dep,
    }


def simular_give_me_some_credit(random_state=RANDOM_STATE):
    """DataFrame completo (150,000 x 11), reproducible con la semilla."""
    rng = np.random.default_rng(random_state)
    y = simular_objetivo(rng)
    cols = simular_predictoras(rng, y)
    cols["age"] = simular_age(rng, y)
    df = pd.DataFrame(cols)[COLUMNAS]
    df[TARGET] = y
    return df


def verificar(df):
    """Compara contra lo REPORTADO. Devuelve un DataFrame de chequeos."""
    filas = [
        ("N total", N_TOTAL, len(df)),
        ("Clase 0", N_NO_DEFAULT, int((df[TARGET] == 0).sum())),
        ("Clase 1", N_DEFAULT, int((df[TARGET] == 1).sum())),
        ("# predictoras", 10, df.shape[1] - 1),
        ("age min", AGE_MIN, int(df["age"].min())),
        ("age max", AGE_MAX, int(df["age"].max())),
        ("age media", AGE_MEAN, round(float(df["age"].mean()), 2)),
    ]
    out = pd.DataFrame(filas, columns=["chequeo", "reportado", "simulado"])
    out["ok"] = np.isclose(out["reportado"], out["simulado"], atol=0.05)
    return out


if __name__ == "__main__":
    from pathlib import Path
    d = simular_give_me_some_credit()
    destino = Path(__file__).resolve().parent.parent / "data" / "datos_simulados.csv"
    destino.parent.mkdir(exist_ok=True)
    d.to_csv(destino, index=False)
    print(f"Guardado: {destino}\n")
    print(verificar(d).to_string(index=False))
    print("\nFaltantes (%):\n", (d.isna().mean() * 100).round(2))
