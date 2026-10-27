import os
import time
import warnings
import inspect
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    import GPUtil
except ImportError:
    GPUtil = None

from src.project_config import obtener_ruta_dataset, preparar_directorios, DATA_DIR, RESULTS_DIR
from src.dataset_analyzer import cargar_dataset
from src.preprocessing import limpiar_dataset, preparar_features_fold, codificar_target_seguro

SEMILLA = 42
N_FOLDS = 5
POLITICA_NULOS = 'eliminar_filas'  # cambiar a 'imputar' SOLO si imágenes usan las mismas filas


def menu_interactivo():
    archivos = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv'))
    if not archivos:
        raise FileNotFoundError(f'No hay CSV en {DATA_DIR}')
    print('\nDatasets disponibles:')
    for i, f in enumerate(archivos, 1):
        print(f'{i}. {f}')
    i = int(input('Número del dataset: ')) - 1
    if i not in range(len(archivos)):
        raise ValueError('Selección de dataset inválida')
    archivo = archivos[i]
    ruta = obtener_ruta_dataset(archivo)
    columnas = pd.read_csv(ruta, nrows=0).columns.tolist()
    for j, col in enumerate(columnas, 1):
        print(f'{j}. {col}')
    j = int(input('Número de columna target: ')) - 1
    if j not in range(len(columnas)):
        raise ValueError('Selección de target inválida')
    return archivo, ruta, columnas[j]


def obtener_metricas_gpu():
    if GPUtil is None:
        return np.nan, np.nan
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100, gpus[0].memoryUsed / 1024
    except Exception as exc:
        warnings.warn(f'No se pudieron leer métricas GPU: {exc}')
    return np.nan, np.nan


def modelos_base():
    return {
        'Random_Forest': RandomForestClassifier(random_state=SEMILLA, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=SEMILLA, class_weight='balanced'),
        'MLP_Clasico': MLPClassifier(random_state=SEMILLA, max_iter=1000),
    }


def calcular_roc_auc(y_true, proba, clases_modelo):
    """AUC binaria o multiclase; NaN si el test no permite estimarla."""
    y_true = np.asarray(y_true)
    clases_modelo = np.asarray(clases_modelo)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        if len(clases_modelo) == 2:
            return float(roc_auc_score(y_true, proba[:, 1], labels=clases_modelo))
        if set(np.unique(y_true)) != set(clases_modelo):
            return np.nan
        return float(roc_auc_score(y_true, proba, labels=clases_modelo,
                                   multi_class='ovr', average='macro'))
    except ValueError as exc:
        warnings.warn(f'ROC-AUC no calculable: {exc}')
        return np.nan


def resumen_estadistico(df, metricas):
    registros = []
    for (dataset, estrategia, modelo), g in df.groupby(['Dataset', 'Estrategia', 'Modelo'], sort=False):
        row = {'Dataset': dataset, 'Estrategia': estrategia, 'Modelo': modelo,
               'Total_Folds': len(g), 'Fold_Actual': 'PROMEDIO'}
        for metrica in metricas:
            vals = pd.to_numeric(g[metrica], errors='coerce').dropna()
            row[metrica] = vals.mean() if len(vals) else np.nan
            row[metrica + '_STD'] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[metrica + '_N_Validos'] = len(vals)
        registros.append(row)
    return pd.DataFrame(registros)


def evaluar_modelos_tabular(X, y, columnas_categoricas=None, columnas_numericas=None,
                             nombre_dataset='dataset.csv'):
    """Mantiene la firma antigua; los tipos se identifican por fold."""
    if len(X) != len(y):
        raise ValueError('X e y tienen tamaños distintos')
    if y.isna().any():
        raise ValueError('El target contiene nulos')
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < N_FOLDS:
        raise ValueError(f'StratifiedKFold({N_FOLDS}) requiere al menos {N_FOLDS} muestras por clase; conteos: {counts.to_dict()}')
    estrategia = 'Original_PesosClases'
    # Se generan una sola vez para garantizar igualdad entre estrategias.
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=SEMILLA).split(X, y))
    filas = []
    proceso = psutil.Process(os.getpid())
    for fold_n, (idx_train, idx_test) in enumerate(folds, 1):
        Xt, Xv = X.iloc[idx_train], X.iloc[idx_test]
        yt, yv = y.iloc[idx_train], y.iloc[idx_test]
        yte, yve, mapa = codificar_target_seguro(yt, yv)
        Xte, Xve, info = preparar_features_fold(Xt, Xv, eliminar_constantes=False)
        print(f'Fold {fold_n}/{N_FOLDS}: train={len(idx_train)}, test={len(idx_test)}, '
              f'columnas constantes conservadas={info["columnas_constantes"]}')
        pesos_muestras = compute_sample_weight(class_weight='balanced', y=yte)
        for nombre, modelo in modelos_base().items():
            proceso.cpu_percent(interval=None)
            inicio_entrenamiento = time.perf_counter()
            if nombre == 'MLP_Clasico':
                if 'sample_weight' not in inspect.signature(modelo.fit).parameters:
                    raise RuntimeError('MLPClassifier sin soporte para sample_weight. Actualiza scikit-learn a >=1.7 para mantener la ponderación en todos los baselines.')
                modelo.fit(Xte, yte, sample_weight=pesos_muestras)
            else:
                modelo.fit(Xte, yte)
            tiempo_entrenamiento = time.perf_counter() - inicio_entrenamiento
            inicio_prediccion = time.perf_counter()
            pred = modelo.predict(Xve)
            tiempo_prediccion = time.perf_counter() - inicio_prediccion
            inicio_probabilidades = time.perf_counter()
            proba = modelo.predict_proba(Xve)
            tiempo_probabilidades = time.perf_counter() - inicio_probabilidades
            segundos = tiempo_entrenamiento + tiempo_prediccion + tiempo_probabilidades
            cpu = proceso.cpu_percent(interval=None)
            gpu, gpu_ram = obtener_metricas_gpu()
            auc = calcular_roc_auc(yve, proba, modelo.classes_)
            filas.append({
                'Dataset': nombre_dataset, 'Estrategia': estrategia, 'Modelo': nombre,
                'Total_Folds': N_FOLDS, 'Fold_Actual': f'Fold_{fold_n}',
                'Accuracy': accuracy_score(yve, pred),
                'F1_Score': f1_score(yve, pred, average='macro',
                                     labels=np.arange(len(mapa)), zero_division=0),
                'ROC_AUC': auc, 'Tiempo_s': segundos,
                'Tiempo_Entrenamiento_s': tiempo_entrenamiento,
                'Tiempo_Clasificacion_s': tiempo_prediccion,
                'Tiempo_Probabilidades_s': tiempo_probabilidades,
                'RAM_GB': proceso.memory_info().rss / 1024**3,
                'CPU_Usage_%': cpu, 'GPU_RAM_GB': gpu_ram, 'GPU_Usage_%': gpu,
                'N_Train_Original': len(idx_train), 'N_Train_Usado': len(yte),
                'N_Test': len(idx_test), 'Clases_Train': str(yt.value_counts().to_dict()),
                'Clases_Test': str(yv.value_counts().to_dict()),
                'Columnas_Constantes_Train': str(info['columnas_constantes']),
                'Seed': SEMILLA, 'Fecha': datetime.now().isoformat(timespec='seconds')
            })
            print(f'  {nombre}: Accuracy={filas[-1]["Accuracy"]:.4f}, '
                  f'F1-macro={filas[-1]["F1_Score"]:.4f}, AUC={auc:.4f}')
    df_folds = pd.DataFrame(filas)
    metricas = ['Accuracy', 'F1_Score', 'ROC_AUC', 'Tiempo_s',
                'Tiempo_Entrenamiento_s', 'Tiempo_Clasificacion_s',
                'Tiempo_Probabilidades_s', 'RAM_GB',
                'CPU_Usage_%', 'GPU_RAM_GB', 'GPU_Usage_%']
    df_resumen = resumen_estadistico(df_folds, metricas)
    salida = Path(RESULTS_DIR)
    salida.mkdir(parents=True, exist_ok=True)
    base = Path(nombre_dataset).stem
    ruta_folds = salida / f'Baseline_{base}_FOLDS.csv'
    ruta_resumen = salida / f'Baseline_{base}_RESUMEN.csv'
    df_folds.to_csv(ruta_folds, index=False)
    df_resumen.to_csv(ruta_resumen, index=False)
    print('\nResumen:\n', df_resumen[['Modelo', 'Estrategia', 'Accuracy', 'Accuracy_STD',
                                      'F1_Score', 'F1_Score_STD', 'ROC_AUC', 'ROC_AUC_STD']].to_string(index=False))
    print(f'Archivos guardados: {ruta_folds}, {ruta_resumen}')
    return df_folds, df_resumen


if __name__ == '__main__':
    preparar_directorios()
    nombre, ruta, target = menu_interactivo()
    df = cargar_dataset(ruta)
    df_limpio, registro = limpiar_dataset(df, target, politica_nulos=POLITICA_NULOS)
    df_limpio = df_limpio.reset_index(drop=True)
    X = df_limpio.drop(columns=[target])
    y = df_limpio[target]
    evaluar_modelos_tabular(X, y, registro['columnas_categoricas'],
                             registro['columnas_numericas'], nombre)
