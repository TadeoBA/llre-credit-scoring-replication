# Actividad Final — Machine Learning II
## Recreación de un estudio científico con datos simulados

**Estudiante:** Omar Tadeo Bayona Araque · Maestría en Analítica Aplicada
**Artículo base:** Wang H, Xu Q, Zhou L (2015). *Large Unbalanced Credit Scoring Using Lasso-Logistic Regression Ensemble.* PLoS ONE 10(2): e0117844. DOI: [10.1371/journal.pone.0117844](https://doi.org/10.1371/journal.pone.0117844) (acceso abierto)

---

## Paso 1. Selección del artículo

### ¿Por qué este artículo?

1. **Dominio alineado con el trabajo de grado.** La tesis estudia *credit scoring* con pruebas de conocimiento cero (ZKP), comparando variables crudas contra variables discretizadas en rangos (los rangos son lo que un circuito ZKP puede atestar sin revelar el valor exacto). Este artículo trabaja el mismo dominio y, además, mide explícitamente cuánto cambia el desempeño al pasar de variables originales a variables derivadas (logaritmos, banderas, *binning*).
2. **Problema central de la clase: desbalanceo.** La razón es ~14:1 (93,3 % / 6,7 %), igual de relevante que lo visto en la competencia de Kaggle (88,3 % / 11,7 %).
3. **Es simulable con supuestos explícitos.** El artículo reporta con precisión el tamaño de la muestra, el desbalanceo exacto, las variables, los hiperparámetros y las métricas. Lo que no reporta (distribuciones por variable) se puede declarar como supuesto explícito.
4. **Métodos comparables y reproducibles:** regresión logística Lasso, CART, Random Forest y un ensamble (LLRE), con AUC y F-measure.
5. **Acceso abierto** y con tablas de resultados numéricas con las que comparar en el Paso 6.

---

## Paso 2. Comprensión del estudio

**Objetivo.** Evaluar si un ensamble de regresiones logísticas con penalización Lasso (**LLRE**), entrenado sobre subconjuntos balanceados, supera a modelos individuales en un problema de *scoring* crediticio grande y muy desbalanceado.

**Metodología del LLRE (4 pasos).**
1. **Clustering de la clase mayoritaria** en *k* = 14 subgrupos (k ≈ razón de desbalanceo).
2. **Balanceo por subgrupo:** cada subgrupo de la clase 0 se combina con una muestra *bootstrap* de la clase 1 → conjunto balanceado (BS).
3. **Un Lasso-logístico por BS.** El λ se elige con los datos fuera del BS (OBS) usando AUC.
4. **Agregación ponderada:** los 14 modelos se combinan con pesos tipo sigmoide según su desempeño.

**Datos.** *Give Me Some Credit* (Kaggle): 150.000 clientes, 10 variables predictoras y la variable objetivo `SeriousDlqin2yrs` (impago grave en 2 años). Partición aleatoria 80 % entrenamiento / 20 % prueba.

**Variables derivadas.** A partir de las 10 originales se generan 80 variables: transformación `log(x+1)`, banderas para valores especiales / faltantes y *binning* (discretización).

**Algoritmos comparados y configuración reportada.**

| Algoritmo | Configuración reportada |
|---|---|
| LLRE (propuesto) | k = 14 subgrupos, Lasso-logístico base |
| Random Forest | `ntree` = 500, `mtry` = 3 |
| CART | criterio Gini |
| Lasso-logístico (LLR) | `glmnet`, `family = binomial`, `type.measure = auc`, `alpha = 1` |

**Evaluación.** 30 corridas; métricas AUC y F-measure; prueba de Friedman + post-hoc de Nemenyi (valor crítico z = 2,619).

**Resultados principales del artículo (a reproducir/comparar en el Paso 6).**

| AUC (Tabla 3) | Variables originales | Con variables derivadas |
|---|---|---|
| LLRE | 0,6796 | 0,8597 |
| Random Forest | 0,8488 | 0,8587 |
| Lasso-logístico | 0,4898 | 0,8567 |
| CART | 0,7702 | 0,7632 |

- **F-measure (Tabla 5):** LLRE ≈ 0,37–0,41 · RF ≈ 0,26–0,30 · LLR ≈ 0,24–0,28 · **CART = 0 en las 30 corridas** (nunca predice la clase minoritaria).
- **Friedman:** AUC = 77,88 y F-measure = 87,76 (p < 2,2e-16); las diferencias de Nemenyi superan el valor crítico.
- **Variables más importantes:** *NeverPastDue*, *DisposableIncome*, *Log RevolvingUtilizationTimesLines*. La menos importante: número de dependientes.

**Conclusión de los autores.** Combinar *clustering* + *bagging* permite usar modelos lineales clásicos (interpretables) en problemas grandes y desbalanceados sin sacrificar desempeño. La ingeniería de variables es decisiva para los modelos lineales (AUC del Lasso: 0,49 → 0,86).

**Relación con el trabajo de grado.** El salto del Lasso de 0,4898 a 0,8567 con variables derivadas es un antecedente directo para la pregunta de la tesis: ¿cuánto se pierde o se gana cuando el modelo trabaja con rangos/derivadas en lugar de valores crudos?

---

## Paso 3. Caracterización de los datos

Este paso distingue la información que **reporta el artículo** de la que **se supone** para la simulación. Toda característica no reportada se declara como supuesto.

### 3.1 Estructura general

| Elemento | Valor | ¿Lo reporta el artículo? |
|---|---|---|
| Registros (N) | 150.000 | ✅ Reportado |
| Variable objetivo | `SeriousDlqin2yrs` (1 = impago grave en 2 años) | ✅ Reportado |
| Clase 0 (no impago) | 139.974 (93,3 %) | ✅ Reportado |
| Clase 1 (impago) | 10.026 (6,7 %) | ✅ Reportado |
| Razón de desbalanceo | ≈ 14 : 1 | ✅ Reportado |
| Variables predictoras | 10 | ✅ Reportado |
| Variables derivadas | 80 | ✅ Reportado |
| Partición | 80 % train / 20 % test, aleatoria | ✅ Reportado |
| Corridas | 30 | ✅ Reportado |

### 3.2 Variables predictoras

| # | Variable | Tipo | Qué reporta el artículo | Supuesto de simulación (a justificar en Paso 4) |
|---|---|---|---|---|
| 1 | `RevolvingUtilizationOfUnsecuredLines` | Continua (razón) | Nombre; es de las más importantes | Sesgada a la derecha con cola larga (valores > 1); mayor en clase 1 |
| 2 | `age` | Entera | **min 0, max 109, media 52,3** | Normal truncada con media 52,3; **desviación estándar supuesta** (no reportada) |
| 3 | `NumberOfTime30-59DaysPastDueNotWorse` | Conteo | Nombre | Poisson; media mayor en clase 1 |
| 4 | `DebtRatio` | Continua (razón) | Nombre | Lognormal, sesgada a la derecha; mayor en clase 1 |
| 5 | `MonthlyIncome` | Continua | Nombre; el ingreso disponible es importante | Lognormal; menor en clase 1; **≈ 20 % de faltantes (supuesto)** |
| 6 | `NumberOfOpenCreditLinesAndLoans` | Conteo | Nombre | Poisson |
| 7 | `NumberOfTimes90DaysLate` | Conteo raro | Nombre; historial de atrasos es importante | Poisson con media baja; mayor en clase 1 |
| 8 | `NumberRealEstateLoansOrLines` | Conteo | Nombre | Poisson |
| 9 | `NumberOfTime60-89DaysPastDueNotWorse` | Conteo raro | Nombre | Poisson con media baja; mayor en clase 1 |
| 10 | `NumberOfDependents` | Conteo | Nombre; es la menos importante | Poisson; **≈ 2,6 % de faltantes (supuesto)**; sin relación fuerte con la clase |

### 3.3 Reportado vs. Supuesto

| Aspecto | Estado | Fundamento / nota |
|---|---|---|
| N, balance exacto de clases, nº de variables, partición 80/20 | **Reportado** | Se reproduce exactamente |
| Media, mínimo y máximo de `age` | **Reportado** | Se reproduce la media; el rango se respeta con truncamiento |
| Desviación estándar de `age` | **Supuesto** | El artículo no la da |
| Forma de la distribución de cada variable | **Supuesto** | No hay estadística descriptiva por variable; se usa la forma típica conocida del dataset público *Give Me Some Credit* |
| % de faltantes en `MonthlyIncome` (≈ 20 %) y `NumberOfDependents` (≈ 2,6 %) | **Supuesto** | Valores aproximados de la descripción pública del dataset original. El artículo solo habla de valores especiales/faltantes de forma cualitativa |
| Relación de cada variable con el impago | **Supuesto parcial** | El artículo solo da la dirección cualitativa vía importancia de variables (Figs. 3–4): historial de atrasos, utilización de crédito e ingreso disponible pesan más; dependientes pesa menos. La **magnitud** del efecto es un supuesto de la simulación |
| Correlaciones entre predictoras | **Supuesto** | No reportadas; en la primera versión se simulan independientes (condicionadas a la clase) |
| Hiperparámetros de LLRE, RF, CART, LLR | **Reportado** | Se usan tal cual |
| Resultados (AUC, F-measure) | **Reportado** | Sirven como referencia en el Paso 6, **no** se usan para calibrar la simulación |

### 3.4 Implicación metodológica

Como las distribuciones y la intensidad de la señal son supuestos, **no se espera reproducir los valores absolutos** del artículo (p. ej., AUC = 0,8597). Lo que se evalúa es si se reproducen los **patrones cualitativos**:
1. El Lasso-logístico mejora mucho con variables derivadas frente a las originales.
2. CART colapsa en la clase minoritaria (F-measure ≈ 0).
3. El ensamble LLRE tiene mejor F-measure que los modelos individuales.

Si algún patrón no se reproduce, se discute en el Paso 6 en lugar de ajustar la simulación hasta obtenerlo, lo que equivaldría a sobreajustar los datos a los resultados del artículo.

---

## Resultados (Pasos 5 a 7)

Todas las cifras provienen de datos simulados y de **3 corridas** (el artículo usa 30). El análisis completo está en `notebooks/00_analisis_principal.ipynb`.

**Con variables derivadas** (artículo: media de 30 corridas, Tablas 4 y 5):

| Clasificador | AUC artículo | AUC simulado | F-measure artículo | F-measure simulado |
|---|---|---|---|---|
| LLRE | 0,8604 | 0,9183 | 0,384 | 0,333 |
| Random Forest | 0,8578 | 0,9259 | 0,275 | 0,644 |
| Lasso-logístico (LLR) | 0,8582 | 0,9349 | 0,263 | 0,646 |
| CART | 0,7735 | 0,8948 | 0 | 0,653 |

**Con variables originales** (AUC):

| Clasificador | Artículo (Tabla 3) | Simulado |
|---|---|---|
| LLRE | 0,6796 | 0,9179 |
| Random Forest | 0,8488 | 0,9251 |
| Lasso-logístico (LLR) | 0,4898 | 0,7339 |
| CART | 0,7702 | 0,8565 |

**Patrones cualitativos (Paso 6).** De los siete patrones contrastados, cuatro se reproducen (mejora del LLR con variables derivadas, estabilidad de RF, LLRE por encima de LLR con variables originales, CART con el menor AUC) y tres no (F-measure de CART igual a 0, mayor F-measure del LLRE y mayor AUC del LLRE).

**Extensión (Paso 7): variables discretizadas en *k* rangos.** AUC medio:

| Clasificador | Derivadas | *k* = 2 | *k* = 3 | *k* = 5 | *k* = 10 |
|---|---|---|---|---|---|
| LLRE | 0,9183 | 0,9114 | 0,9179 | 0,9140 | 0,9170 |
| Random Forest | 0,9259 | 0,8980 | 0,8853 | 0,9057 | 0,9176 |
| LLR | 0,9349 | 0,9144 | 0,9271 | 0,9318 | 0,9341 |
| CART | 0,8948 | 0,8866 | 0,8573 | 0,8535 | 0,8526 |

F-measure medio:

| Clasificador | Derivadas | *k* = 2 | *k* = 3 | *k* = 5 | *k* = 10 |
|---|---|---|---|---|---|
| LLRE | 0,3326 | 0,3288 | 0,2340 | 0,1400 | 0,1395 |
| Random Forest | 0,6437 | 0,5385 | 0,5892 | 0,5988 | 0,5773 |
| LLR | 0,6462 | 0,5523 | 0,5933 | 0,6422 | 0,6449 |
| CART | 0,6531 | 0,5762 | 0,5509 | 0,6163 | 0,6085 |

## Conclusiones (Paso 8)

1. **Recreación.** Se implementaron el LLRE (K-means con *k* = 14, 11 subgrupos conservados, un Lasso-logístico por conjunto balanceado y pesos según el AUC), RF, LLR y CART, con el protocolo del artículo (partición 80 % / 20 %, variables originales y derivadas, AUC y F-measure), sobre datos simulados que reproducen exactamente lo reportado (tamaño de la muestra, conteo de clases y estadísticos de `age`). Se usaron 3 corridas; el artículo usa 30.
2. **La superioridad del LLRE no se reproduce.** Con variables derivadas el AUC es 0,935 (LLR), 0,926 (RF), 0,918 (LLRE) y 0,895 (CART). El F-measure del LLRE con umbral 0,5 (0,333) es menos de la mitad del de los otros tres clasificadores (≈ 0,65). En el artículo el LLRE es el mejor en ambas métricas.
3. **El F-measure bajo del LLRE se debe al umbral de decisión.** Como se entrena con conjuntos balanceados, el LLRE marca como morosos 8.460 de 30.000 clientes con umbral 0,5 (precisión 0,205, recall 0,897). Con el mejor umbral posible sobre prueba (0,83; cota superior, no utilizable en la práctica) su F1 sube a 0,649, similar al de RF y LLR. Su capacidad de ordenar clientes es comparable; lo que no está calibrado para la prevalencia real es la regla de decisión de 0,5.
4. **La ingeniería de variables importa para el modelo lineal.** El AUC de LLR pasa de 0,734 con variables originales a 0,935 con derivadas, mientras que el de RF no cambia (0,925 y 0,926). Es el mismo patrón del artículo (LLR de 0,490 a 0,857) con menor magnitud.
5. **El nivel de señal condiciona las métricas que dependen del umbral.** El AUC simulado (0,89 a 0,93) es mayor que el del artículo (0,77 a 0,86), y con él cambia el F-measure: por ejemplo, CART obtiene 0,653 frente a 0 en el artículo. De los siete patrones cualitativos contrastados, se reproducen cuatro (todos de AUC) y no se reproducen tres (CART, F-measure del LLRE y AUC del LLRE).
6. **Extensión (variables discretizadas en rangos).** Con un modelo lineal, reducir las variables a *k* ≥ 5 rangos cuesta como máximo 0,003 de AUC y 0,004 de F-measure respecto de las variables derivadas; con *k* = 2 cuesta 0,02 y 0,09. RF y CART pierden más y sin un patrón regular. El LLRE conserva el AUC, pero su F-measure con umbral 0,5 depende de la calibración de sus probabilidades.
7. **Implicaciones para el trabajo de grado (ZKP).** (a) En esta simulación, un modelo de *scoring* lineal que recibe rangos con al menos 5 niveles por variable conserva casi todo su poder de ordenamiento. (b) Los valores extremos y los códigos especiales son un riesgo para el modelo lineal (hipótesis de la sección 4.5, no verificada); tanto las banderas como los rangos lo neutralizan en esta simulación. (c) El AUC y las métricas que dependen del umbral se deben reportar por separado, y el umbral se debe calibrar con la prevalencia. (d) Estos resultados provienen de datos simulados y deben confirmarse con datos reales.

### Limitaciones

- **Número de corridas.** Se usaron 3 corridas y el artículo usa 30; no se aplicó la prueba de Friedman ni el post-hoc de Nemenyi, y las diferencias de AUC menores a 0,01 no se pueden interpretar.
- **Datos simulados.** La intensidad de la señal, la forma de las distribuciones y la independencia entre variables son supuestos. Las conclusiones son sobre estos datos y no sobre el conjunto real.
- **Variables derivadas.** El artículo no lista sus 80 variables; el conjunto usado tiene 56 columnas y es una reconstrucción con las técnicas declaradas.
- **Implementación.** El artículo usa R (`glmnet`, `rpart`, `randomForest`); aquí se usó scikit-learn, con una rejilla de λ de 10 valores, validación cruzada de 3 pliegues en LLR y tolerancia de 1e-3.
- **Reproducibilidad del LLRE.** Dos ejecuciones del análisis de umbral (sección 5.7) con la misma semilla dieron resultados ligeramente distintos (por ejemplo, 8.137 y 8.460 clientes marcados con umbral 0,5; F1 de 0,343 y 0,334). Las cifras de este documento corresponden a la última ejecución.
- **Hipótesis sin verificar.** Las explicaciones marcadas como hipótesis en el notebook principal no se probaron con experimentos específicos.

### Trabajo futuro

- Aumentar a 30 corridas y aplicar la prueba de Friedman con el post-hoc de Nemenyi, como en el artículo.
- Verificar las hipótesis planteadas: ablación sin los códigos 96/98 (efecto sobre LLR), variación de la intensidad de la señal (comportamiento de CART y de RF en F-measure) y reajuste de `mtry` y `cp` para las variables discretizadas.
- Repetir la extensión con rangos fijados por diseño, como los que atestaría un circuito de conocimiento cero, en lugar de cuantiles.
- Ejecutar el mismo código sobre el conjunto de datos real (*Give Me Some Credit*).

---

## Diseño para reutilización (tesis y portafolio)

Este repositorio está pensado para servir más allá de la clase:

- **Pipeline independiente de la fuente de datos.** Los módulos de `src/` (Pasos 5–7) reciben un `DataFrame` con las 10 columnas y el objetivo; no saben si vienen de la simulación o de un CSV real. Así, el mismo código se puede correr después sobre datos reales para la tesis.
- **Extensión (Paso 7) con sentido para la tesis.** La condición que se cambia es la forma en que el modelo recibe las variables: se discretizan en *k* rangos (2, 3, 5 y 10), con cortes por cuantiles del conjunto de entrenamiento, que es una aproximación de lo que un circuito ZKP puede atestar (rangos en lugar de valores exactos). Se mide cuánto cuesta ese cambio en AUC y F-measure.
- **Reproducible desde cero.** `pip install -r requirements.txt` y luego `python src/simulate_data.py`. Semilla fija: el CSV incluido en `data/` es exactamente el que regenera `python src/simulate_data.py`.
- **Supuestos documentados.** La tabla "Reportado vs. Supuesto" (Paso 3) forma parte del repositorio y separa los hechos del artículo de las decisiones de simulación.

**Limitaciones.** Con datos simulados se valida el *método* y el *pipeline*, no se obtienen conclusiones sobre crédito real. Esas solo salen al correr el mismo código sobre datos reales.

## Estructura del repositorio

```
llre-credit-scoring-replication/
├── README.md                          # este archivo
├── requirements.txt
├── LICENSE                            # MIT
├── .gitignore
├── data/
│   └── datos_simulados.csv            # 150.000 filas simuladas (Paso 4)  [generado]
├── notebooks/
│   ├── 00_analisis_principal.ipynb    # notebook principal: análisis, resultados e interpretación
│   ├── 01_simulacion_datos.ipynb      # Paso 4: simulación y verificación
│   ├── 02_analisis_articulo.ipynb     # Paso 5: recreación del análisis
│   └── 03_extension.ipynb             # Paso 7: extensión experimental
├── src/
│   ├── simulate_data.py               # generación de datos (semilla fija)
│   ├── features.py                    # variables derivadas (log, banderas, binning)
│   ├── llre.py                        # ensamble Lasso-logístico
│   └── evaluate.py                    # AUC, F-measure, corridas
├── results/                           # tablas de métricas (articulo/: Tablas 3, 4 y 5 transcritas)
└── figures/                           # gráficos
```

Todo el código reutilizable vive en `src/`; los notebooks solo lo importan y muestran resultados. Semilla global: `RANDOM_STATE = 42`.

## Cómo reproducir

```bash
pip install -r requirements.txt
python src/simulate_data.py            # regenera data/datos_simulados.csv (semilla 42)
python src/run_experiment.py --runs 3  # Paso 5: results/paso5_resultados_por_corrida.csv
```

Los notebooks se ejecutan en este orden: `01_simulacion_datos`, `02_analisis_articulo`, `03_extension` y, al final, `00_analisis_principal`, que lee los resultados guardados en `results/` y no reentrena modelos. El notebook 02 define `N_RUNS` (número de corridas) y el 03 debe usar un valor menor o igual, porque reutiliza las mismas particiones (semillas `1000 + corrida`). Con los datos completos, el Paso 5 con 3 corridas toma decenas de minutos; los resultados se guardan de forma incremental y, si la ejecución se interrumpe, se retoma donde quedó.

## Estado del proyecto

- [x] Paso 1: selección del artículo
- [x] Paso 2: comprensión del estudio
- [x] Paso 3: caracterización de los datos
- [x] Paso 4: simulación de los datos
- [x] Paso 5: recreación del análisis (LLRE, Lasso-logístico, RF, CART)
- [x] Paso 6: comparación con las Tablas 3 y 5 del artículo
- [x] Paso 7: extensión experimental (cambiar **una** condición)
- [x] Paso 8: conclusiones

## Referencia

Wang H, Xu Q, Zhou L (2015) Large Unbalanced Credit Scoring Using Lasso-Logistic Regression Ensemble. PLoS ONE 10(2): e0117844. https://doi.org/10.1371/journal.pone.0117844
