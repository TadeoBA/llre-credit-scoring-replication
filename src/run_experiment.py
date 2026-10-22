"""
run_experiment.py -- Paso 5: corre el experimento completo desde la terminal.

    python src/run_experiment.py --runs 30          # protocolo del articulo
    python src/run_experiment.py --runs 10          # version rapida

Escribe results/paso5_resultados_por_corrida.csv de forma incremental:
si se interrumpe, al volver a ejecutarlo continua donde quedo.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from evaluate import correr_experimento  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--data", default=str(RAIZ / "data" / "datos_simulados.csv"),
                    help="CSV con las 10 columnas + SeriousDlqin2yrs (simulado o real)")
    ap.add_argument("--out", default=str(RAIZ / "results" / "paso5_resultados_por_corrida.csv"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(exist_ok=True)
    correr_experimento(pd.read_csv(a.data), n_runs=a.runs, destino=a.out)
