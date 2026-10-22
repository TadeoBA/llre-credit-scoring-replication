"""
features.py -- Paso 5: variables originales vs. derivadas
==========================================================
Wang et al. (2015) corren todos los clasificadores con dos conjuntos de
variables: las 10 ORIGINALES y un conjunto de 80 DERIVADAS. Las tecnicas que
declaran (seccion "Variables generation") son tres:
  1. log(x + 1) para variables con sesgo alto
  2. banderas ("nuevas variables") para valores especiales y NA, en lugar de
     imputar a ciegas
  3. binning / discretizacion en pocos intervalos
Para `age` dan la formula exacta (Ec. 9): bandera Juvenile (age <= 17),
log(age - 17) para 18..59 y bandera Senior (age >= 60).

SUPUESTO: el articulo NO lista las 80 variables. Aqui se
reconstruye un conjunto equivalente con las tecnicas declaradas y los
nombres que si menciona (NeverPastDue, DisposableIncome,
LogRevolvingUtilizationTimesLines, DelinquenciesPerRevolvingLine,
ExcessUtilization, HasRevolvingLines, LogNumberOfTimesPastDue, LogAge, ...).
Las definiciones exactas de esas variables son supuestos de esta implementacion (no vienen en el
articulo). Cada grupo de variables es seleccionable por separado para poder
hacer ablaciones (Paso 7).

Todo se ajusta (fit) SOLO con entrenamiento y se aplica (transform) a prueba.
"""

import numpy as np
import pandas as pd

ATRASOS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
CONTINUAS_BIN = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome",
]
GRUPOS = ("log", "flags", "ingenieria", "age_eq9", "bins")


class FeatureBuilder:
    """
    modo = "original" : las 10 variables, imputando NA con la mediana de train
                        (glmnet/RF/CART no aceptan NA; es el supuesto para
                        el conjunto original).
    modo = "derived"  : originales + derivadas (grupos en `grupos`).
    modo = "discretized" : (extension, Paso 7) las 10 variables se reemplazan por
                        indicadores de rango: n_bins rangos por cuantiles de
                        train (o, para conteos con mucha masa en el minimo, los
                        primeros valores distintos: 0 | 1 | 2 | ...) y, si la
                        variable tiene NA en train, un rango adicional "NA".
                        No queda ninguna variable continua.

    n_bins controla el binning por cuantiles de train (grupo "bins" y modo
    "discretized").
    """

    def __init__(self, modo="derived", grupos=GRUPOS, n_bins=5,
                 incluir_continuas=True):
        assert modo in ("original", "derived", "discretized")
        self.modo, self.grupos = modo, tuple(grupos)
        self.n_bins, self.incluir_continuas = n_bins, incluir_continuas

    # ------------------------------------------------------------------ fit
    def fit(self, df):
        self.medianas_ = df.median(numeric_only=True)
        base = self._base(df)
        self.cortes_ = {}
        if self.modo == "discretized":
            raw = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
            self.tiene_na_ = {}
            for c in raw.columns:
                vals = raw[c].dropna()
                q = np.unique(vals.quantile(np.linspace(0, 1, self.n_bins + 1)[1:-1]).values)
                q = q[q > vals.min()]                    # sin rangos vacios
                if len(q) < self.n_bins - 1:
                    # variable de conteo con mucha masa en el minimo (p. ej. atrasos):
                    # los cuantiles colapsan; se usan los primeros valores distintos
                    # sobre el minimo (0 | 1 | 2 | ... | k+), que son rangos naturales
                    distintos = np.sort(vals.unique())
                    q = distintos[distintos > vals.min()][: self.n_bins - 1]
                self.cortes_[c] = q
                self.tiene_na_[c] = bool(raw[c].isna().any())
        if self.modo == "derived" and "bins" in self.grupos:
            for c in CONTINUAS_BIN:
                q = np.unique(base[c].quantile(
                    np.linspace(0, 1, self.n_bins + 1)[1:-1]).values)
                self.cortes_[c] = q
        self.columnas_ = list(self._construir(df).columns)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df):
        return self._construir(df)[self.columnas_]

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    # ------------------------------------------------------------- internos
    def _base(self, df):
        """10 originales con NA imputados por mediana de train."""
        x = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore").copy()
        return x.fillna(self.medianas_)

    def _construir(self, df):
        base = self._base(df)
        if self.modo == "original":
            return base
        raw = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
        if self.modo == "discretized":
            out = {}
            for c, cortes in self.cortes_.items():
                v = raw[c]
                idx = np.digitize(v.fillna(0).values, cortes)
                n_cat = len(cortes) + 1
                if self.tiene_na_[c]:
                    idx = np.where(v.isna().values, n_cat, idx)
                    n_cat += 1
                for b in range(n_cat):
                    out[f"{c}_r{b}"] = (idx == b).astype(int)
            return pd.DataFrame(out, index=df.index)
        out = {}
        if self.incluir_continuas:
            for c in base.columns:
                out[c] = base[c]

        # codigos especiales (96/98) -> se tratan como "atraso desconocido"
        especial = (raw[ATRASOS] >= 90).any(axis=1)
        atr = raw[ATRASOS].where(raw[ATRASOS] < 90, 0)
        n_atrasos = atr.sum(axis=1)

        if "flags" in self.grupos:
            out["MonthlyIncome_NA"] = raw["MonthlyIncome"].isna().astype(int)
            out["NumberOfDependents_NA"] = raw["NumberOfDependents"].isna().astype(int)
            out["SpecialCodePastDue"] = especial.astype(int)
            out["NeverPastDue30_59"] = (atr[ATRASOS[0]] == 0).astype(int)
            out["NeverLate90"] = (atr[ATRASOS[1]] == 0).astype(int)
            out["NeverPastDue60_89"] = (atr[ATRASOS[2]] == 0).astype(int)
            out["NeverPastDue"] = (n_atrasos == 0).astype(int)
            out["ExcessUtilization"] = (base["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)

        if "ingenieria" in self.grupos:
            lineas_rev = (base["NumberOfOpenCreditLinesAndLoans"]
                          - base["NumberRealEstateLoansOrLines"]).clip(lower=0)
            out["HasRevolvingLines"] = (lineas_rev > 0).astype(int)
            out["NumberOfTimesPastDue"] = n_atrasos
            out["DelinquenciesPerRevolvingLine"] = n_atrasos / (lineas_rev + 1)
            out["DisposableIncome"] = base["MonthlyIncome"] * (1 - base["DebtRatio"])
            out["RevolvingUtilizationTimesLines"] = (
                base["RevolvingUtilizationOfUnsecuredLines"]
                * base["NumberOfOpenCreditLinesAndLoans"])

        if "log" in self.grupos:
            for c in ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
                      "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
                      "NumberRealEstateLoansOrLines", "NumberOfDependents"):
                out["Log" + c] = np.log1p(base[c].clip(lower=0))
            out["LogNumberOfTimesPastDue"] = np.log1p(n_atrasos)
            out["LogRevolvingUtilizationTimesLines"] = np.log1p(
                base["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0)
                * base["NumberOfOpenCreditLinesAndLoans"])
            out["LogDisposableIncome"] = np.log1p(
                (base["MonthlyIncome"] * (1 - base["DebtRatio"])).clip(lower=0))

        if "age_eq9" in self.grupos:                        # Ec. (9) del articulo
            a = base["age"]
            out["JuvenileAge"] = (a <= 17).astype(int)
            out["WorkingAgeLog"] = np.where((a >= 18) & (a <= 59),
                                            np.log(np.maximum(a - 17, 1)), 0.0)
            out["SeniorAge"] = (a >= 60).astype(int)
            out["LogAge"] = np.log1p(a)

        if "bins" in self.grupos:
            for c, cortes in self.cortes_.items():
                idx = np.digitize(base[c].values, cortes)   # 0..len(cortes)
                for b in range(len(cortes) + 1):
                    out[f"{c}_bin{b}"] = (idx == b).astype(int)

        return pd.DataFrame(out, index=df.index)
