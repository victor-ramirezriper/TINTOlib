from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.project_config import (
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    ANALYSIS_DIR,
    preparar_directorios,
)

# ==========================================
# CONFIGURACIÓN Y DIRECTORIOS DINÁMICOS
# ==========================================

BASE_DIR = PROJECT_ROOT
BASE_ANALYSIS_DIR = ANALYSIS_DIR

REGISTRO_REGISTROS_PATH = (
    DATA_DIR / "Registro" / "registro_datasets.csv"
)

# Crear directorios de trabajo si no existen
preparar_directorios()

# ==========================================
# MODELOS Y MÉTODOS
# ==========================================

MODELOS_BASELINE = [
    "Random_Forest",
    "SVM",
    "MLP_Clasico",
]

# Los 11 métodos de TINTOlib
METODOS_TINTO = [
    "TINTO",
    "IGTD",
    "REFINED",
    "BARGRAPH",
    "DISTANCEMATRIX",
    "COMBINATION",
    "SUPERTML",
    "FEATUREWRAP",
    "BIE",
    "FOTOMICS",
    "DEEPINSIGHT",
]

ARQUITECTURAS_ESPERADAS = [
    "ResNet18",
    "EfficientNet-B0",
    "ViT",
]

COLUMNAS_METRICAS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
]

COLUMNAS_TIEMPO = [
    "Tiempo_Transformacion_s",
    "Tiempo_Train_s",
    "Tiempo_Eval_s",
    "Tiempo_s",
]

COLUMNAS_RECURSOS = [
    "RAM_GB",
    "CPU_Usage_%",
    "VRAM_GB",
    "GPU_Usage_%",
]

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ==========================================
# FUNCIONES GENERALES Y NORMALIZACION
# ==========================================

def asegurar_directorio(ruta):
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def limpiar_nombre_archivo(texto):
    texto = str(texto)
    texto = re.sub(r"[^\w\-\.]+", "_", texto)
    return texto.strip("_")


def normalizar_texto(texto):
    if texto is None:
        return ""

    texto = str(texto).strip().lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n"
    }

    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)

    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)

    return texto.strip("_")


def normalizar_dataset_clave(texto):
    texto = normalizar_texto(texto)
    texto = re.sub(r"_lote_?\d+", "", texto)
    texto = re.sub(r"_(completo|original|orig|processed|preprocessed|train|test)$", "", texto)
    texto = re.sub(r"_(completo|original|orig|processed|preprocessed|train|test)_", "_", texto)
    texto = re.sub(r"_+", "_", texto)
    return texto.strip("_")


def convertir_numerico(serie):
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def buscar_columna(df, candidatos):
    if df is None or df.empty:
        return None

    mapa = {}
    for columna in df.columns:
        clave = normalizar_texto(columna)
        mapa[clave] = columna

    for candidato in candidatos:
        clave = normalizar_texto(candidato)
        if clave in mapa:
            return mapa[clave]

    return None


def obtener_columna_numerica(df, candidatos):
    columna = buscar_columna(df, candidatos)
    if columna is None:
        return pd.Series(np.nan, index=df.index)
    return convertir_numerico(df[columna])


def leer_csv_seguro(ruta):
    try:
        return pd.read_csv(ruta)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(ruta, encoding="latin-1")
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def obtener_info_registro_dataset(nombre_dataset):
    if not REGISTRO_REGISTROS_PATH.exists():
        return None
    try:
        df_reg = pd.read_csv(REGISTRO_REGISTROS_PATH)
        norm_obj = normalizar_dataset_clave(nombre_dataset)
        for _, fila in df_reg.iterrows():
            ds_reg = str(fila.get("Dataset", ""))
            if normalizar_dataset_clave(ds_reg) == norm_obj or norm_obj in normalizar_dataset_clave(ds_reg):
                inst = fila.get("Instancias", np.nan)
                feat = fila.get("Features", np.nan)
                clases = fila.get("Num_Clases", np.nan)
                return {
                    "Dataset": fila.get("Dataset", nombre_dataset),
                    "Instancias": int(inst) if pd.notna(inst) and str(inst).replace('.','',1).isdigit() else "N/A",
                    "Features": int(feat) if pd.notna(feat) and str(feat).replace('.','',1).isdigit() else "N/A",
                    "Num_Clases": int(clases) if pd.notna(clases) and str(clases).replace('.','',1).isdigit() else "N/A"
                }
    except Exception:
        pass
    return None


# ==========================================
# IDENTIFICACION DE ARQUITECTURAS Y METODOS
# ==========================================

def normalizar_arquitectura(valor):
    if valor is None:
        return None

    texto = normalizar_texto(valor)

    if "resnet18" in texto:
        return "ResNet18"

    if "efficientnet" in texto or "efficient_net" in texto:
        return "EfficientNet-B0"

    if texto == "vit" or texto.startswith("vit_") or "vision_transformer" in texto or "vit_tiny" in texto:
        return "ViT"
        
    return None


def detectar_arquitectura_en_ruta(ruta):
    ruta = Path(ruta)
    for parte in reversed(list(ruta.parts)):
        arquitectura = normalizar_arquitectura(parte)
        if arquitectura is not None:
            return arquitectura
    return None


def normalizar_metodo(valor):
    if valor is None:
        return None

    texto = normalizar_texto(valor)

    equivalencias = {
        "tinto": "TINTO",
        "igtd": "IGTD",
        "refined": "REFINED",
        "bargraph": "BARGRAPH",
        "bar_graph": "BARGRAPH",
        "distancematrix": "DISTANCEMATRIX",
        "distance_matrix": "DISTANCEMATRIX",
        "combination": "COMBINATION",
        "supertml": "SUPERTML",
        "featurewrap": "FEATUREWRAP",
        "bie": "BIE",
        "fotomics": "FOTOMICS",
        "deepinsight": "DEEPINSIGHT"
    }

    if texto in equivalencias:
        return equivalencias[texto]

    for clave, resultado in equivalencias.items():
        if clave in texto:
            return resultado

    return None


# ==========================================
# CARGA DE DATOS (BASELINES Y DEEP LEARNING)
# ==========================================

def buscar_baselines():
    if not RESULTS_DIR.exists():
        return []
    return sorted([archivo for archivo in RESULTS_DIR.glob("Baseline_*.csv") if archivo.is_file()])


def extraer_dataset_desde_baseline(ruta):
    nombre = ruta.stem
    if nombre.lower().startswith("baseline_"):
        nombre = nombre[len("Baseline_"):]
    return nombre


def cargar_baseline(ruta):
    print(f"\nCargando baseline:\n  {ruta}")
    df = leer_csv_seguro(ruta)
    if df.empty:
        print("  Baseline vacio o no disponible.")
        return pd.DataFrame()

    col_estrategia = buscar_columna(df, ["Estrategia", "Strategy"])
    if col_estrategia is not None:
        estrategias_disponibles = df[col_estrategia].astype(str).str.strip().str.upper().unique()
        if "SMOTE" in estrategias_disponibles:
            print("  -> Estrategia detectada: Se aplica filtro prioritario a SMOTE.")
            df = df[df[col_estrategia].astype(str).str.strip().str.upper() == "SMOTE"].copy()
        elif "ORIGINAL" in estrategias_disponibles:
            print("  -> Estrategia detectada: Solo se encontro Original, cargando Original.")
            df = df[df[col_estrategia].astype(str).str.strip().str.upper() == "ORIGINAL"].copy()

    col_fold = buscar_columna(df, ["Fold_Actual", "Fold", "fold"])
    if col_fold is not None:
        df = df[df[col_fold].astype(str).str.contains("Fold", case=False, na=False)].copy()

    if df.empty:
        print("  No hay registros validos despues del filtrado de estrategia y folds.")
        return pd.DataFrame()

    resultado = pd.DataFrame(index=df.index)
    resultado["Tipo"] = "Baseline"

    columna_modelo = buscar_columna(df, ["Modelo", "model", "Algoritmo"])
    resultado["Modelo"] = df[columna_modelo].astype(str).str.strip() if columna_modelo else ""
    
    resultado["Arquitectura"] = "Tabular"
    resultado["Metodo_TINTO"] = "BASELINE"
    resultado["Fold"] = obtener_columna_numerica(df, ["Fold_Actual", "Fold", "fold"])
    
    for metrica in COLUMNAS_METRICAS:
        serie = obtener_columna_numerica(df, [metrica, f"Baseline_{metrica}", metrica.lower()])
        if serie.notna().any() and serie.max(skipna=True) <= 1.0:
            serie = serie * 100.0
        resultado[metrica] = serie

    resultado["Tiempo_Transformacion_s"] = obtener_columna_numerica(df, ["Tiempo_Transformacion_s", "Tiempo_Transformacion", "Transform_Time_s"])
    resultado["Tiempo_Train_s"] = obtener_columna_numerica(df, ["Tiempo_Entrenamiento_s", "Tiempo_Train_s", "Training_Time_s"])
    resultado["Tiempo_Eval_s"] = obtener_columna_numerica(df, ["Tiempo_Evaluacion_s", "Tiempo_Eval_s", "Evaluation_Time_s"])
    resultado["Tiempo_s"] = obtener_columna_numerica(df, ["Tiempo_s", "Tiempo_Prom_Fold_s", "Tiempo_Total_s"])

    resultado["RAM_GB"] = obtener_columna_numerica(df, ["RAM_GB", "Max_RAM_GB"])
    resultado["CPU_Usage_%"] = obtener_columna_numerica(df, ["CPU_Usage_%", "CPU_Usage", "Avg_CPU_Percent"])
    resultado["VRAM_GB"] = obtener_columna_numerica(df, ["Max_VRAM_GB", "VRAM_GB", "GPU_RAM_GB", "VRAM"])
    resultado["GPU_Usage_%"] = obtener_columna_numerica(df, ["GPU_Usage_%", "GPU_Usage", "Avg_GPU_Percent"])

    resultado["Fuente"] = str(ruta)
    resultado["Dataset"] = extraer_dataset_desde_baseline(ruta)

    mask_tiempo = (resultado["Tiempo_s"].isna() & resultado["Tiempo_Train_s"].notna() & resultado["Tiempo_Eval_s"].notna())
    resultado.loc[mask_tiempo, "Tiempo_s"] = resultado.loc[mask_tiempo, "Tiempo_Train_s"] + resultado.loc[mask_tiempo, "Tiempo_Eval_s"]

    return resultado


def cargar_experiment_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
        if isinstance(datos, dict):
            return datos
    except Exception:
        pass
    return {}


def extraer_metodos_experiment(experiment):
    metodos = experiment.get("metodos_evaluados", [])
    if isinstance(metodos, str):
        metodos = [metodos]
    if not isinstance(metodos, list):
        return []
    return list(dict.fromkeys([normalizar_metodo(m) for m in metodos if normalizar_metodo(m) is not None]))


def obtener_dataset_experiment(experiment):
    for clave in ["dataset_lote", "dataset", "Dataset", "Dataset_Lote"]:
        if clave in experiment and experiment[clave]:
            return str(experiment[clave])
    return ""


def obtener_red_experiment(experiment):
    for clave in ["red_neuronal", "modelo_red", "arquitectura", "modelo"]:
        if clave in experiment and experiment[clave]:
            return str(experiment[clave])
    return ""


def dataset_coincide(dataset_seleccionado, dataset_experiment):
    if not dataset_seleccionado or not dataset_experiment:
        return False
    sel = normalizar_dataset_clave(dataset_seleccionado)
    exp = normalizar_dataset_clave(dataset_experiment)
    return sel == exp or sel in exp or exp in sel


def buscar_experimentos_deep_learning(dataset_seleccionado):
    if not RESULTS_DIR.exists():
        return []

    experimentos = []
    encontrados_json = set()

    for experiment_json in RESULTS_DIR.rglob("experiment.json"):
        if not experiment_json.is_file():
            continue
        carpeta_experimento = experiment_json.parent
        if str(experiment_json) in encontrados_json:
            continue
        encontrados_json.add(str(experiment_json))

        datos = cargar_experiment_json(experiment_json)
        if not datos:
            continue

        if not dataset_coincide(dataset_seleccionado, obtener_dataset_experiment(datos)):
            continue

        arquitectura = normalizar_arquitectura(obtener_red_experiment(datos))
        if arquitectura is None:
            arquitectura = detectar_arquitectura_en_ruta(carpeta_experimento)

        if arquitectura is None:
            continue

        experimentos.append({
            "carpeta": carpeta_experimento,
            "arquitectura": arquitectura,
            "metodos": extraer_metodos_experiment(datos)
        })

    return experimentos


def cargar_resultados_folds(experimento):
    carpeta = experimento["carpeta"]
    arquitectura = experimento["arquitectura"]
    metodos_json = experimento["metodos"]

    ruta = carpeta / "resultados_folds.csv"
    if not ruta.exists():
        return pd.DataFrame()

    df = leer_csv_seguro(ruta)
    if df.empty:
        return pd.DataFrame()

    resultado = pd.DataFrame(index=df.index)
    resultado["Tipo"] = "Deep Learning"
    resultado["Arquitectura"] = arquitectura

    columna_metodo = buscar_columna(df, ["Metodo_TINTO", "Metodo", "Method"])
    if columna_metodo is not None:
        resultado["Metodo_TINTO"] = df[columna_metodo].astype(str).apply(normalizar_metodo)
    elif len(metodos_json) == 1:
        resultado["Metodo_TINTO"] = metodos_json[0]
    else:
        resultado["Metodo_TINTO"] = np.nan

    columna_fold = buscar_columna(df, ["Fold", "Fold_Actual", "fold"])
    resultado["Fold"] = convertir_numerico(df[columna_fold]) if columna_fold is not None else np.arange(1, len(df) + 1)
    resultado["Modelo"] = resultado["Metodo_TINTO"].fillna("").astype(str)

    resultado["Accuracy"] = obtener_columna_numerica(df, ["Accuracy", "Acc", "accuracy"])
    resultado["F1_Score"] = obtener_columna_numerica(df, ["F1_Score", "F1", "f1_score"])
    resultado["ROC_AUC"] = obtener_columna_numerica(df, ["ROC_AUC", "AUC", "roc_auc"])

    bal_acc_col = buscar_columna(df, ["Balanced_Accuracy", "BalancedAcc", "Bal_Acc"])
    if bal_acc_col is not None:
        resultado["Balanced_Accuracy"] = obtener_columna_numerica(df, [bal_acc_col])
    else:
        resultado["Balanced_Accuracy"] = resultado["Accuracy"]

    prec_col = buscar_columna(df, ["Precision", "Macro_Precision"])
    if prec_col is not None:
        resultado["Precision"] = obtener_columna_numerica(df, [prec_col])
    else:
        resultado["Precision"] = resultado["F1_Score"]

    rec_col = buscar_columna(df, ["Recall", "Macro_Recall"])
    if rec_col is not None:
        resultado["Recall"] = obtener_columna_numerica(df, [rec_col])
    else:
        resultado["Recall"] = resultado["Balanced_Accuracy"]

    resultado["Tiempo_Transformacion_s"] = obtener_columna_numerica(df, ["Tiempo_Transformacion_s", "Tiempo_Transformacion", "Transform_Time_s"])
    resultado["Tiempo_Train_s"] = obtener_columna_numerica(df, ["Tiempo_Entrenamiento_s", "Tiempo_Train_s", "Training_Time_s"])
    resultado["Tiempo_Eval_s"] = obtener_columna_numerica(df, ["Tiempo_Evaluacion_s", "Tiempo_Eval_s", "Evaluation_Time_s"])
    
    resultado["RAM_GB"] = obtener_columna_numerica(df, ["Max_RAM_GB", "RAM_GB"])
    resultado["CPU_Usage_%"] = obtener_columna_numerica(df, ["Avg_CPU_Percent", "CPU_Usage_%", "CPU_Usage"])
    resultado["VRAM_GB"] = obtener_columna_numerica(df, ["Max_VRAM_GB", "VRAM_GB", "GPU_RAM_GB", "VRAM"])
    resultado["GPU_Usage_%"] = obtener_columna_numerica(df, ["GPU_Usage_%", "GPU_Usage", "Avg_GPU_Percent"])

    resultado["Tiempo_s"] = resultado["Tiempo_Train_s"] + resultado["Tiempo_Eval_s"]
    resultado["Fuente"] = str(ruta)

    return resultado


def cargar_historial_entrenamiento(experimento):
    carpeta = experimento["carpeta"]
    arquitectura = experimento["arquitectura"]
    metodos_json = experimento["metodos"]

    ruta = carpeta / "historial_entrenamiento.csv"
    if not ruta.exists():
        return pd.DataFrame()

    df = leer_csv_seguro(ruta)
    if df.empty:
        return pd.DataFrame()

    resultado = pd.DataFrame(index=df.index)
    resultado["Arquitectura"] = arquitectura

    columna_metodo = buscar_columna(df, ["Metodo_TINTO", "Metodo", "Method"])
    if columna_metodo is not None:
        resultado["Metodo_TINTO"] = df[columna_metodo].astype(str).apply(normalizar_metodo)
    elif len(metodos_json) == 1:
        resultado["Metodo_TINTO"] = metodos_json[0]
    else:
        resultado["Metodo_TINTO"] = np.nan

    resultado["Fold"] = obtener_columna_numerica(df, ["Fold", "Fold_Actual"])
    resultado["Epoch"] = obtener_columna_numerica(df, ["Epoch", "Epoca"])
    resultado["Train_Loss"] = obtener_columna_numerica(df, ["Train_Loss", "Training_Loss"])
    resultado["Val_Loss"] = obtener_columna_numerica(df, ["Val_Loss", "Validation_Loss"])
    resultado["Fuente"] = str(ruta)

    return resultado


def cargar_predicciones(experimento):
    carpeta = experimento["carpeta"]
    arquitectura = experimento["arquitectura"]
    metodos_json = experimento["metodos"]

    ruta = carpeta / "predicciones_folds.csv"
    if not ruta.exists():
        return pd.DataFrame()

    df = leer_csv_seguro(ruta)
    if df.empty:
        return pd.DataFrame()

    df["Arquitectura"] = arquitectura
    columna_metodo = buscar_columna(df, ["Metodo", "Metodo_TINTO", "Method"])
    if columna_metodo is not None:
        df["Metodo_TINTO"] = df[columna_metodo].astype(str).apply(normalizar_metodo)
    elif len(metodos_json) == 1:
        df["Metodo_TINTO"] = metodos_json[0]
    else:
        df["Metodo_TINTO"] = np.nan

    return df


def cargar_deep_learning(dataset_seleccionado):
    experimentos = buscar_experimentos_deep_learning(dataset_seleccionado)
    if not experimentos:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    folds, historiales, predicciones = [], [], []

    for experimento in experimentos:
        df_folds = cargar_resultados_folds(experimento)
        if not df_folds.empty:
            folds.append(df_folds)

        historial = cargar_historial_entrenamiento(experimento)
        if not historial.empty:
            historiales.append(historial)

        preds = cargar_predicciones(experimento)
        if not preds.empty:
            predicciones.append(preds)

    columnas_base = [
        "Tipo", "Arquitectura", "Metodo_TINTO", "Modelo", "Fold",
        "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC",
        "Tiempo_Transformacion_s", "Tiempo_Train_s", "Tiempo_Eval_s", "Tiempo_s",
        "RAM_GB", "CPU_Usage_%", "VRAM_GB", "GPU_Usage_%", "Fuente"
    ]

    df_folds = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(columns=columnas_base)
    df_historial = pd.concat(historiales, ignore_index=True) if historiales else pd.DataFrame()
    df_predicciones = pd.concat(predicciones, ignore_index=True) if predicciones else pd.DataFrame()

    return df_folds, df_historial, df_predicciones


def generar_resumen_desde_folds(df_folds):
    if df_folds is None or df_folds.empty:
        return pd.DataFrame()

    columnas_a_promediar = [
        "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC",
        "Tiempo_Transformacion_s", "Tiempo_Train_s", "Tiempo_Eval_s", "Tiempo_s",
        "RAM_GB", "CPU_Usage_%", "VRAM_GB", "GPU_Usage_%"
    ]

    disponibles = [col for col in columnas_a_promediar if col in df_folds.columns]
    agrupacion = ["Arquitectura", "Metodo_TINTO"]

    resumen = (
        df_folds
        .groupby(agrupacion, dropna=False)[disponibles]
        .mean()
        .reset_index()
    )

    resumen["Tipo"] = "Deep Learning"
    resumen["Modelo"] = resumen["Metodo_TINTO"]
    resumen["Fold"] = np.nan
    resumen["Fuente"] = "Agregado desde resultados_folds.csv"

    return resumen


# ==========================================
# ESTRUCTURA DE 14 MODELOS
# ==========================================

def crear_estructura_modelos(df_base, df_dl, arquitectura):
    filas = []

    if df_base is not None and not df_base.empty:
        for modelo in MODELOS_BASELINE:
            modelo_norm = normalizar_texto(modelo)
            datos_modelo = df_base[df_base["Modelo"].apply(normalizar_texto) == modelo_norm].copy()

            fila = {
                "Tipo": "Baseline",
                "Arquitectura": arquitectura,
                "Modelo": modelo,
                "Metodo_TINTO": "BASELINE"
            }

            for columna in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                if columna in datos_modelo.columns and not datos_modelo.empty:
                    fila[columna] = datos_modelo[columna].mean()
                else:
                    fila[columna] = np.nan

            filas.append(fila)
    else:
        for modelo in MODELOS_BASELINE:
            fila = {"Tipo": "Baseline", "Arquitectura": arquitectura, "Modelo": modelo, "Metodo_TINTO": "BASELINE"}
            for columna in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                fila[columna] = np.nan
            filas.append(fila)

    for metodo in METODOS_TINTO:
        fila = {
            "Tipo": "Deep Learning",
            "Arquitectura": arquitectura,
            "Modelo": metodo,
            "Metodo_TINTO": metodo
        }

        if df_dl is not None and not df_dl.empty:
            datos = df_dl[(df_dl["Arquitectura"] == arquitectura) & (df_dl["Metodo_TINTO"] == metodo)]
            if not datos.empty:
                for columna in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                    fila[columna] = datos[columna].mean() if columna in datos.columns else np.nan
            else:
                for columna in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                    fila[columna] = np.nan
        else:
            for columna in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                fila[columna] = np.nan

        filas.append(fila)

    return pd.DataFrame(filas)


def normalizar_metricas_porcentaje(df):
    resultado = df.copy()
    for columna in COLUMNAS_METRICAS:
        if columna not in resultado.columns:
            continue
        valores = resultado[columna]
        if valores.notna().any() and valores.max(skipna=True) <= 1.0:
            resultado[f"{columna}_pct"] = valores * 100.0
        else:
            resultado[f"{columna}_pct"] = valores
    return resultado


# ==========================================
# GENERACION DE TABLAS CSV Y VISUALES (IMAGEN)
# ==========================================

def generar_tablas_comparativas(df_modelos, arquitectura, output_dir_csv, output_dir_visual, dataset_nombre=""):
    if df_modelos.empty:
        return

    columnas_tabla = {
        "Modelo": "Metodo",
        "F1_Score_pct": "F1-Score (%)",
        "Accuracy_pct": "Acc. (%)",
        "ROC_AUC_pct": "ROC-AUC (%)",
        "VRAM_GB": "VRAM (GB)",
        "Tiempo_s": "Tiempo Total (s)"
    }

    disponibles = [col for col in columnas_tabla.keys() if col in df_modelos.columns]
    df_tabla = df_modelos[disponibles].rename(columns=columnas_tabla).copy()

    for col in df_tabla.columns:
        if col != "Metodo":
            df_tabla[col] = pd.to_numeric(df_tabla[col], errors="coerce").round(3)

    if "Acc. (%)" in df_tabla.columns:
        df_tabla = df_tabla.sort_values(by="Acc. (%)", ascending=False).reset_index(drop=True)

    prefijo = limpiar_nombre_archivo(arquitectura)
    
    ruta_csv = output_dir_csv / f"Tabla_Resultados_{prefijo}.csv"
    df_tabla.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"  -> Tabla CSV generada: {ruta_csv}")

    col_accuracy = next(
        (c for c in df_tabla.columns if c.strip().lower() in ["acc. (%)", "accuracy (%)", "accuracy_pct"]),
        None
    )

    col_f1 = next(
        (c for c in df_tabla.columns if c.strip().lower() in ["f1-score (%)", "f1_score (%)", "f1_score_pct"]),
        None
    )

    max_accuracy = None
    max_f1 = None
    if col_accuracy is not None:
        df_tabla[col_accuracy] = pd.to_numeric(df_tabla[col_accuracy], errors="coerce")
        max_accuracy = df_tabla[col_accuracy].max()

    if col_f1 is not None:
        df_tabla[col_f1] = pd.to_numeric(df_tabla[col_f1], errors="coerce")
        max_f1 = df_tabla[col_f1].max()

    info_reg = obtener_info_registro_dataset(dataset_nombre)
    if info_reg:
        inst_val = int(info_reg['Instancias']) if str(info_reg['Instancias']).isdigit() else info_reg['Instancias']
        feat_val = int(info_reg['Features']) if str(info_reg['Features']).isdigit() else info_reg['Features']
        clas_val = int(info_reg['Num_Clases']) if str(info_reg['Num_Clases']).isdigit() else info_reg['Num_Clases']
        texto_info = (f"Dataset: {info_reg['Dataset']} | Instancias: {inst_val} | "
                      f"Features: {feat_val} | Clases: {clas_val}")
    else:
        texto_info = f"Dataset: {dataset_nombre}"

    fig, ax = plt.subplots(
        figsize=(
            max(10, len(df_tabla.columns) * 1.5),
            max(3, len(df_tabla) * 0.6 + 2.5)
        )
    )

    ax.axis("off")

    plt.title("Mejores Modelos", fontsize=14, fontweight="bold", pad=15)

    tabla = ax.table(
        cellText=df_tabla.values,
        colLabels=df_tabla.columns,
        loc="center",
        cellLoc="center"
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)

    for (row, col), cell in tabla.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)
        if row == 0:
            cell.set_text_props(weight="bold")

    if col_accuracy is not None and max_accuracy is not None:
        indice_accuracy = df_tabla.columns.get_loc(col_accuracy)
        for fila in range(len(df_tabla)):
            if df_tabla.iloc[fila][col_accuracy] == max_accuracy:
                tabla[(fila + 1, indice_accuracy)].set_text_props(weight="bold")

    if col_f1 is not None and max_f1 is not None:
        indice_f1 = df_tabla.columns.get_loc(col_f1)
        for fila in range(len(df_tabla)):
            if df_tabla.iloc[fila][col_f1] == max_f1:
                tabla[(fila + 1, indice_f1)].set_text_props(weight="bold")

    plt.figtext(
        0.5, 0.02, texto_info,
        wrap=True, horizontalalignment='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='black', linewidth=2)
    )

    ruta_imagen = output_dir_visual / f"Tabla_Visual_{prefijo}.png"

    plt.savefig(
        ruta_imagen,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3
    )
    plt.close()
    print(f"  -> Tabla visual generada: {ruta_imagen}")


# ==========================================
# GENERACION DE TABLA RESUMEN GENERAL (MEJORES MODELOS GLOBALES)
# ==========================================

def generar_tabla_resumen_general():
    print("\n" + "=" * 70)
    print("GENERANDO RESUMEN GENERAL DE MEJORES MODELOS (Graficas_Generales)")
    print("=" * 70)

    baselines = buscar_baselines()
    if not baselines:
        print("No se encontraron baselines para resumir.")
        return

    resultados_globales = []

    for baseline in baselines:
        dataset_nombre = extraer_dataset_desde_baseline(baseline)
        df_base = cargar_baseline(baseline)
        df_dl, _, _ = cargar_deep_learning(dataset_nombre)

        info_reg = obtener_info_registro_dataset(dataset_nombre)
        instancias = int(info_reg.get("Instancias", 0)) if info_reg and str(info_reg.get("Instancias", "")).isdigit() else "N/A"
        features = int(info_reg.get("Features", 0)) if info_reg and str(info_reg.get("Features", "")).isdigit() else "N/A"
        num_clases = int(info_reg.get("Num_Clases", 0)) if info_reg and str(info_reg.get("Num_Clases", "")).isdigit() else "N/A"

        arquitecturas_disponibles = df_dl["Arquitectura"].unique() if not df_dl.empty else ["Tabular"]
        
        for arch in arquitecturas_disponibles:
            df_modelos = crear_estructura_modelos(df_base, df_dl, arch)
            df_modelos = normalizar_metricas_porcentaje(df_modelos)

            if df_modelos.empty:
                continue

            if "Accuracy_pct" in df_modelos.columns and df_modelos["Accuracy_pct"].notna().any():
                mejor_idx = df_modelos["Accuracy_pct"].idxmax()
                mejor_fila = df_modelos.loc[mejor_idx]

                resultados_globales.append({
                    "Dataset": dataset_nombre,
                    "Arquitectura": arch,
                    "Mejor_Modelo": mejor_fila.get("Modelo", "N/A"),
                    "Accuracy_%": round(mejor_fila.get("Accuracy_pct", np.nan), 3),
                    "F1_Score_%": round(mejor_fila.get("F1_Score_pct", np.nan), 3),
                    "ROC_AUC_%": round(mejor_fila.get("ROC_AUC_pct", np.nan), 3),
                    "Tiempo_Total_s": round(mejor_fila.get("Tiempo_s", np.nan), 3),
                    "Instancias": instancias,
                    "Features": features,
                    "Num_Clases": num_clases
                })

    if not resultados_globales:
        print("No se pudieron recopilar resultados para la tabla global.")
        return

    df_global = pd.DataFrame(resultados_globales)

    df_global["Instancias_num"] = pd.to_numeric(df_global["Instancias"], errors="coerce")
    df_global = df_global.sort_values(by="Instancias_num", ascending=True).drop(columns=["Instancias_num"]).reset_index(drop=True)

    dir_generales = asegurar_directorio(RESULTS_DIR / "Graficas_Generales")
    ruta_csv = dir_generales / "Resumen_Mejores_Modelos_Global.csv"
    df_global.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"-> CSV Resumen Global guardado en: {ruta_csv}")

    fig, ax = plt.subplots(figsize=(14, max(4, len(df_global) * 0.5 + 2.0)))
    ax.axis("off")
    plt.title("Mejores Modelos - Resumen General por Dataset (Ordenado por Instancias)", fontsize=14, fontweight="bold", pad=15)

    tabla = ax.table(
        cellText=df_global.values,
        colLabels=df_global.columns,
        loc="center",
        cellLoc="center"
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.4)

    for (row, col), cell in tabla.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)
        if row == 0:
            cell.set_text_props(weight="bold")

    ruta_imagen = dir_generales / "Resumen_Mejores_Modelos_Global.png"
    plt.savefig(ruta_imagen, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"-> Imagen Resumen Global guardada en: {ruta_imagen}")


# ==========================================
# FUNCIONES DE GRAFICADO CON UNIDADES ADECUADAS
# ==========================================

def generar_grafica_barras(df, columna, titulo, nombre_archivo, output_dir, y_label, porcentaje=False):
    if df.empty or columna not in df.columns:
        return

    datos = df[df[columna].notna()].copy()
    if datos.empty:
        return

    es_tiempo_recurso = any(c in columna for c in COLUMNAS_TIEMPO + COLUMNAS_RECURSOS)
    
    if es_tiempo_recurso:
        datos = datos.sort_values(by=columna, ascending=True).reset_index(drop=True)
    else:
        datos = datos.sort_values(by=columna, ascending=False).reset_index(drop=True)

    x = np.arange(len(datos))
    valores = datos[columna].astype(float).values
    modelos = datos["Modelo"].values

    if porcentaje and np.nanmax(np.abs(valores)) <= 1.0:
        valores = valores * 100.0

    fig, ax = plt.subplots(figsize=(15, 8))
    
    colores = ["#4C72B0"] * len(valores)
    if len(valores) > 0:
        colores[0] = "#2ca02c"
        colores[-1] = "#d62728"

    barras = ax.bar(x, valores, color=colores, edgecolor='black', linewidth=1)

    ax.set_title(titulo, fontsize=15, fontweight="bold")
    ax.set_xlabel("Modelos y Métodos Evaluados", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha="right", fontsize=10)

    for i, (barra, valor) in enumerate(zip(barras, valores)):
        texto_extra = ""
        if i == 0:
            texto_extra = "\n(Mejor)"
        elif i == len(valores) - 1:
            texto_extra = "\n(Peor)"

        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            f"{valor:.2f}{texto_extra}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if (i == 0 or i == len(valores) - 1) else "normal"
        )

    plt.tight_layout()
    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Gráfica: {ruta}")


def generar_scatter(df, x_col, y_col, titulo, nombre_archivo, output_dir, x_label, y_label):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    datos = df[[x_col, y_col, "Modelo", "Tipo"]].dropna()
    if datos.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    for tipo in datos["Tipo"].unique():
        subset = datos[datos["Tipo"] == tipo]
        ax.scatter(subset[x_col], subset[y_col], s=100, alpha=0.8, label=tipo)
        for _, fila in subset.iterrows():
            ax.annotate(str(fila["Modelo"]), (fila[x_col], fila[y_col]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_title(titulo, fontsize=15, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica: {ruta}")


def generar_boxplot_folds(df, arquitectura, columna, titulo, nombre_archivo, output_dir, y_label):
    if df.empty or columna not in df.columns:
        return

    datos = df[(df["Arquitectura"] == arquitectura) & (df[columna].notna())].copy()
    if datos.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(data=datos, x="Metodo_TINTO", y=columna, ax=ax)
    sns.stripplot(data=datos, x="Metodo_TINTO", y=columna, ax=ax, color="black", alpha=0.45, size=4)

    ax.set_title(titulo, fontsize=15, fontweight="bold")
    ax.set_xlabel("Metodos TINTOlib Evaluados por Folds", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica: {ruta}")


def calcular_intervalo_confianza(serie, confianza=0.95):
    serie = pd.Series(serie).dropna()
    n = len(serie)
    if n == 0:
        return np.nan, np.nan, np.nan
    media = serie.mean()
    if n == 1:
        return media, np.nan, np.nan
    std = serie.std(ddof=1)
    error = std / np.sqrt(n)
    try:
        from scipy.stats import t
        critico = t.ppf(1 - (1 - confianza) / 2, df=n - 1)
    except Exception:
        critico = 1.96
    margen = critico * error
    return media, media - margen, media + margen


def generar_grafica_ic95(df, arquitectura, columna, titulo, nombre_archivo, output_dir, y_label):
    if df.empty or columna not in df.columns:
        return

    datos = df[(df["Arquitectura"] == arquitectura) & (df[columna].notna())].copy()
    if datos.empty:
        return

    filas = []
    for metodo, grupo in datos.groupby("Metodo_TINTO"):
        media, inferior, superior = calcular_intervalo_confianza(grupo[columna])
        filas.append({"Metodo": metodo, "Media": media, "Inferior": inferior, "Superior": superior})

    resumen = pd.DataFrame(filas)
    if resumen.empty:
        return

    x = np.arange(len(resumen))
    media = resumen["Media"].values
    inferiores = np.nan_to_num(media - resumen["Inferior"].values)
    superiores = np.nan_to_num(resumen["Superior"].values - media)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.errorbar(x, media, yerr=[inferiores, superiores], fmt="o", capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(resumen["Metodo"], rotation=45, ha="right")
    ax.set_title(titulo, fontsize=15, fontweight="bold")
    ax.set_xlabel("Metodos TINTOlib", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

    plt.tight_layout()
    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica: {ruta}")


def calcular_estadisticas_folds(df_folds):
    if df_folds.empty:
        return pd.DataFrame()

    columnas = COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS
    columnas = [c for c in columnas if c in df_folds.columns]
    filas = []

    for (arquitectura, metodo), grupo in df_folds.groupby(["Arquitectura", "Metodo_TINTO"], dropna=False):
        for columna in columnas:
            serie = grupo[columna].dropna()
            if serie.empty:
                continue
            n = len(serie)
            media = serie.mean()
            std = serie.std(ddof=1) if n > 1 else np.nan
            mediana = serie.median()
            minimo = serie.min()
            maximo = serie.max()
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            cv = (std / abs(media)) * 100 if pd.notna(std) and media != 0 else np.nan
            _, ci_inf, ci_sup = calcular_intervalo_confianza(serie)

            filas.append({
                "Arquitectura": arquitectura,
                "Metodo_TINTO": metodo,
                "Metrica": columna,
                "N": n,
                "Media": media,
                "Desviacion_Std": std,
                "Mediana": mediana,
                "Minimo": minimo,
                "Maximo": maximo,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "CV_%": cv,
                "IC95_Inferior": ci_inf,
                "IC95_Superior": ci_sup
            })

    return pd.DataFrame(filas)


def calcular_estadisticas_arquitecturas(df_resumen):
    if df_resumen.empty:
        return pd.DataFrame()

    columnas = COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS
    columnas = [c for c in columnas if c in df_resumen.columns]

    return (
        df_resumen
        .groupby("Arquitectura")[columnas]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )


def generar_heatmap(df, columnas, titulo, nombre_archivo, output_dir, estandarizar=False):
    if df.empty:
        return

    disponibles = [c for c in columnas if c in df.columns]
    if not disponibles:
        return

    datos = df[["Modelo"] + disponibles].copy()
    datos = datos.drop_duplicates(subset=["Modelo"]).set_index("Modelo")
    datos = datos.apply(pd.to_numeric, errors="coerce")

    if estandarizar:
        for columna in datos.columns:
            media = datos[columna].mean()
            std = datos[columna].std()
            if pd.notna(std) and std != 0:
                datos[columna] = (datos[columna] - media) / std

    datos = datos.dropna(how="all")
    if datos.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(7, len(datos) * 0.45)))
    sns.heatmap(
        datos,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0 if estandarizar else None,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(titulo, fontsize=15, fontweight="bold")
    ax.set_xlabel("Metricas y Recursos Evaluados", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modelos Evaluados", fontsize=12, fontweight="bold")
    plt.tight_layout()

    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica: {ruta}")


def generar_correlacion(df, columnas, titulo, nombre_archivo, output_dir):
    if df.empty:
        return None

    disponibles = [c for c in columnas if c in df.columns]
    if len(disponibles) < 2:
        return None

    datos = df[disponibles].apply(pd.to_numeric, errors="coerce")
    correlacion = datos.corr()
    if correlacion.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(titulo, fontsize=15, fontweight="bold")
    plt.tight_layout()

    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica: {ruta}")

    return correlacion


def generar_graficas_historial(df_historial, output_dir):
    if df_historial.empty:
        return

    carpeta_loss = asegurar_directorio(output_dir)

    for arquitectura in sorted(df_historial["Arquitectura"].dropna().unique()):
        datos_arquitectura = df_historial[df_historial["Arquitectura"] == arquitectura]
        for metodo in sorted(datos_arquitectura["Metodo_TINTO"].dropna().unique()):
            datos = datos_arquitectura[datos_arquitectura["Metodo_TINTO"] == metodo].copy()
            if datos.empty or datos["Epoch"].isna().all():
                continue

            fig, ax = plt.subplots(figsize=(13, 8))
            for _, grupo in datos.groupby("Fold"):
                grupo = grupo.sort_values("Epoch")
                if grupo["Train_Loss"].notna().any():
                    ax.plot(grupo["Epoch"], grupo["Train_Loss"], alpha=0.25, linewidth=1)
                if grupo["Val_Loss"].notna().any():
                    ax.plot(grupo["Epoch"], grupo["Val_Loss"], alpha=0.25, linewidth=1, linestyle="--")

            agrupado = datos.groupby("Epoch").agg(
                Train_Media=("Train_Loss", "mean"),
                Train_STD=("Train_Loss", "std"),
                Val_Media=("Val_Loss", "mean"),
                Val_STD=("Val_Loss", "std")
            ).reset_index()

            if agrupado["Train_Media"].notna().any():
                ax.plot(agrupado["Epoch"], agrupado["Train_Media"], linewidth=3, label="Train Loss promedio")
                ax.fill_between(agrupado["Epoch"], agrupado["Train_Media"] - agrupado["Train_STD"].fillna(0), agrupado["Train_Media"] + agrupado["Train_STD"].fillna(0), alpha=0.15)

            if agrupado["Val_Media"].notna().any():
                ax.plot(agrupado["Epoch"], agrupado["Val_Media"], linewidth=3, linestyle="--", label="Validation Loss promedio")
                ax.fill_between(agrupado["Epoch"], agrupado["Val_Media"] - agrupado["Val_STD"].fillna(0), agrupado["Val_Media"] + agrupado["Val_STD"].fillna(0), alpha=0.15)

            ax.set_title(f"Curvas de perdida de entrenamiento y validacion para {arquitectura} con {metodo}", fontsize=15, fontweight="bold")
            ax.set_xlabel("Epocas de entrenamiento", fontsize=12, fontweight="bold")
            ax.set_ylabel("Valor de perdida Loss", fontsize=12, fontweight="bold")
            ax.legend()
            plt.tight_layout()

            nombre = f"Loss_{limpiar_nombre_archivo(arquitectura)}_{limpiar_nombre_archivo(metodo)}.png"
            plt.savefig(carpeta_loss / nombre, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  -> Grafica Loss: {carpeta_loss / nombre}")


def generar_analisis_matrices_confusion(df_predicciones, output_dir):
    if df_predicciones.empty or "y_true" not in df_predicciones.columns or "y_pred" not in df_predicciones.columns:
        return

    for arquitectura in sorted(df_predicciones["Arquitectura"].dropna().unique()):
        arch_dir = asegurar_directorio(output_dir / limpiar_nombre_archivo(arquitectura))
        subset_arch = df_predicciones[df_predicciones["Arquitectura"] == arquitectura]

        for metodo in sorted(subset_arch["Metodo_TINTO"].dropna().unique()):
            subset_met = subset_arch[subset_arch["Metodo_TINTO"] == metodo]
            if subset_met.empty:
                continue

            y_true = subset_met["y_true"].values
            y_pred = subset_met["y_pred"].values

            if len(y_true) == 0:
                continue

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"Matriz de Confusion - {arquitectura} ({metodo})", fontsize=14, fontweight="bold")
            ax.set_xlabel("Etiqueta Predicha", fontsize=12, fontweight="bold")
            ax.set_ylabel("Etiqueta Real", fontsize=12, fontweight="bold")
            plt.tight_layout()

            nombre_archivo = f"Matriz_Confusion_{limpiar_nombre_archivo(arquitectura)}_{limpiar_nombre_archivo(metodo)}.png"
            plt.savefig(arch_dir / nombre_archivo, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  -> Matriz de confusion generada: {arch_dir / nombre_archivo}")


def calcular_pruebas_estadisticas(df_folds):
    if df_folds.empty:
        return pd.DataFrame()

    try:
        from scipy.stats import kruskal, f_oneway
    except Exception:
        return pd.DataFrame()

    metricas = COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS
    filas = []

    for arquitectura in sorted(df_folds["Arquitectura"].dropna().unique()):
        datos_arch = df_folds[df_folds["Arquitectura"] == arquitectura]
        for metrica in metricas:
            if metrica not in datos_arch.columns:
                continue

            grupos = [grupo[metrica].dropna().values for _, grupo in datos_arch.groupby("Metodo_TINTO") if len(grupo[metrica].dropna().values) >= 2]
            if len(grupos) < 2:
                continue

            try:
                estadistico_kw, p_kw = kruskal(*grupos)
            except Exception:
                estadistico_kw, p_kw = np.nan, np.nan

            try:
                estadistico_anova, p_anova = f_oneway(*grupos)
            except Exception:
                estadistico_anova, p_anova = np.nan, np.nan

            filas.append({
                "Arquitectura": arquitectura,
                "Metrica": metrica,
                "Numero_Grupos": len(grupos),
                "Kruskal_Wallis_H": estadistico_kw,
                "Kruskal_Wallis_p": p_kw,
                "ANOVA_F": estadistico_anova,
                "ANOVA_p": p_anova
            })

    return pd.DataFrame(filas)


def generar_grafica_pvalues(df_pruebas, arquitectura, output_dir):
    if df_pruebas.empty:
        return

    datos = df_pruebas[(df_pruebas["Arquitectura"] == arquitectura) & (df_pruebas["Kruskal_Wallis_p"].notna())].copy()
    if datos.empty:
        return

    p = datos["Kruskal_Wallis_p"].clip(lower=1e-16)
    valores = -np.log10(p)

    fig, ax = plt.subplots(figsize=(13, 7))
    barras = ax.bar(datos["Metrica"], valores)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=2, label="Umbral p = 0.05 significativo")

    ax.set_title(f"Prueba estadistica Kruskal-Wallis por metrica para {arquitectura}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Metrica Evaluada", fontsize=12, fontweight="bold")
    ax.set_ylabel("Valor -log10 de p-value", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(), f"{valor:.2f}", ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()
    ruta = output_dir / f"Kruskal_Wallis_{limpiar_nombre_archivo(arquitectura)}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica estadistica: {ruta}")


# ==========================================
# FLUJO DE GENERACION POR ARQUITECTURA
# ==========================================

def generar_graficas_comparativas(df_base, df_resumen, df_folds, arquitectura, output_dir_comp, output_dir_tablas_csv, output_dir_tablas_visual, dataset_nombre=""):
    print(f"\n" + "-" * 70)
    print(f"Arquitectura: {arquitectura}")

    df_modelos = crear_estructura_modelos(df_base, df_resumen, arquitectura)
    df_modelos = normalizar_metricas_porcentaje(df_modelos)
    print(f"Modelos incluidos: {len(df_modelos)}\n")

    prefijo = limpiar_nombre_archivo(arquitectura)

    generar_tablas_comparativas(df_modelos, arquitectura, output_dir_tablas_csv, output_dir_tablas_visual, dataset_nombre)

    for col, tit, nom in [
        ("Accuracy_pct", f"Comparativa de Accuracy porcentaje para {arquitectura}", f"Comparativa_{prefijo}_Accuracy.png"),
        ("Balanced_Accuracy_pct", f"Comparativa de Balanced Accuracy porcentaje para {arquitectura}", f"Comparativa_{prefijo}_Balanced_Accuracy.png"),
        ("Precision_pct", f"Comparativa de Precision porcentaje para {arquitectura}", f"Comparativa_{prefijo}_Precision.png"),
        ("Recall_pct", f"Comparativa de Recall porcentaje para {arquitectura}", f"Comparativa_{prefijo}_Recall.png"),
        ("F1_Score_pct", f"Comparativa de F1-Score porcentaje para {arquitectura}", f"Comparativa_{prefijo}_F1_Score.png"),
        ("ROC_AUC_pct", f"Comparativa de ROC-AUC porcentaje para {arquitectura}", f"Comparativa_{prefijo}_ROC_AUC.png")
    ]:
        generar_grafica_barras(df_modelos, col, tit, nom, output_dir_comp, y_label="Porcentaje de Rendimiento (%)", porcentaje=False)

    for col, tit, nom in [
        ("Tiempo_s", f"Tiempo total de procesamiento en segundos para {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Total.png"),
        ("Tiempo_Transformacion_s", f"Tiempo de transformacion TINTOlib en segundos para {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Transformacion.png"),
        ("Tiempo_Train_s", f"Tiempo de entrenamiento en segundos para {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Train.png"),
        ("Tiempo_Eval_s", f"Tiempo de evaluacion en segundos para {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Eval.png")
    ]:
        generar_grafica_barras(df_modelos, col, tit, nom, output_dir_comp, y_label="Tiempo Total (segundos)")

    for col, tit, nom in [
        ("RAM_GB", f"Uso de Memoria RAM en Gigabytes para {arquitectura}", f"Comparativa_{prefijo}_RAM.png"),
        ("CPU_Usage_%", f"Uso promedio de CPU en porcentaje para {arquitectura}", f"Comparativa_{prefijo}_CPU.png"),
        ("VRAM_GB", f"Uso de Memoria VRAM de la GPU en Gigabytes para {arquitectura}", f"Comparativa_{prefijo}_VRAM.png"),
        ("GPU_Usage_%", f"Uso promedio de la GPU en porcentaje para {arquitectura}", f"Comparativa_{prefijo}_GPU.png")
    ]:
        generar_grafica_barras(df_modelos, col, tit, nom, output_dir_comp, y_label="Consumo de Recursos")

    generar_scatter(df_modelos, "Tiempo_s", "Accuracy_pct", f"Accuracy porcentaje contra Tiempo Total de procesamiento para {arquitectura}", f"Scatter_{prefijo}_Accuracy_vs_Tiempo.png", output_dir_comp, "Tiempo total de procesamiento (segundos)", "Accuracy (%)")
    generar_scatter(df_modelos, "Tiempo_Transformacion_s", "Accuracy_pct", f"Accuracy porcentaje contra Tiempo de Transformacion para {arquitectura}", f"Scatter_{prefijo}_Accuracy_vs_Transformacion.png", output_dir_comp, "Tiempo de transformacion TINTOlib (segundos)", "Accuracy (%)")
    generar_scatter(df_modelos, "RAM_GB", "Accuracy_pct", f"Accuracy porcentaje contra Memoria RAM consumida para {arquitectura}", f"Scatter_{prefijo}_Accuracy_vs_RAM.png", output_dir_comp, "Memoria RAM (GB)", "Accuracy (%)")

    generar_heatmap(
        df_modelos,
        ["Accuracy_pct", "Balanced_Accuracy_pct", "Precision_pct", "Recall_pct", "F1_Score_pct", "ROC_AUC_pct", "Tiempo_Transformacion_s", "Tiempo_s", "RAM_GB", "CPU_Usage_%", "VRAM_GB"],
        f"Mapa comparativo estandarizado de metricas y recursos para {arquitectura}",
        f"Heatmap_{prefijo}.png",
        output_dir_comp,
        estandarizar=True
    )

    return df_modelos


def generar_analisis_folds(df_folds, output_dir):
    if df_folds.empty:
        return

    for arquitectura in sorted(df_folds["Arquitectura"].dropna().unique()):
        prefijo = limpiar_nombre_archivo(arquitectura)
        for col, tit, nom, ylbl in [
            ("Accuracy", f"Variabilidad de Accuracy por Folds para {arquitectura}", f"Boxplot_{prefijo}_Accuracy_Folds.png", "Accuracy (%)"),
            ("Balanced_Accuracy", f"Variabilidad de Balanced Accuracy por Folds para {arquitectura}", f"Boxplot_{prefijo}_Balanced_Accuracy_Folds.png", "Balanced Accuracy (%)"),
            ("Precision", f"Variabilidad de Precision por Folds para {arquitectura}", f"Boxplot_{prefijo}_Precision_Folds.png", "Precision (%)"),
            ("Recall", f"Variabilidad de Recall por Folds para {arquitectura}", f"Boxplot_{prefijo}_Recall_Folds.png", "Recall (%)"),
            ("F1_Score", f"Variabilidad de F1-Score por Folds para {arquitectura}", f"Boxplot_{prefijo}_F1_Folds.png", "F1-Score Macro (%)"),
            ("ROC_AUC", f"Variabilidad de ROC-AUC por Folds para {arquitectura}", f"Boxplot_{prefijo}_ROC_AUC_Folds.png", "ROC-AUC (%)"),
            ("Tiempo_Transformacion_s", f"Variabilidad de Tiempo de Transformacion por Folds para {arquitectura}", f"Boxplot_{prefijo}_Tiempo_Transformacion_Folds.png", "Tiempo de Transformacion (segundos)")
        ]:
            datos_temp = df_folds.copy()
            if col in COLUMNAS_METRICAS and datos_temp[col].max(skipna=True) <= 1.0:
                datos_temp[col] = datos_temp[col] * 100.0
                ylbl = f"{col} (%)"

            generar_boxplot_folds(datos_temp, arquitectura, col, tit, nom, output_dir, y_label=ylbl)
            generar_grafica_ic95(datos_temp, arquitectura, col, f"Media e intervalo de confianza del 95 por ciento para {col} en {arquitectura}", f"IC95_{prefijo}_{col}.png", output_dir, y_label=ylbl)


def generar_analisis_arquitecturas(df_resumen, output_dir):
    if df_resumen.empty:
        return

    columnas = COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS
    for columna in columnas:
        if columna not in df_resumen.columns:
            continue
        datos = df_resumen[df_resumen[columna].notna()].copy()
        if datos.empty:
            continue

        datos = datos.groupby(["Metodo_TINTO", "Arquitectura"], as_index=False)[columna].mean()
        if datos.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 9))
        sns.barplot(data=datos, x="Metodo_TINTO", y=columna, hue="Arquitectura", ax=ax)
        ax.set_title(f"Comparacion entre arquitecturas de Deep Learning para la metrica {columna}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Metodos TINTOlib", fontsize=12, fontweight="bold")
        ylbl = f"{columna} (segundos)" if "Tiempo" in columna else columna
        ax.set_ylabel(ylbl, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        ruta = output_dir / f"Comparacion_Arquitecturas_{limpiar_nombre_archivo(columna)}.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Grafica arquitectura: {ruta}")


# ==========================================
# MAIN PRINCIPAL CON MENÚ REPETIBLE
# ==========================================

def main():
    print("\n" + "=" * 70)
    print("        ANALISIS VISUAL Y ESTADISTICO TINTOlib")
    print("=" * 70)

    if not RESULTS_DIR.exists():
        print(f"\nNo existe la carpeta Results:\n  {RESULTS_DIR}")
        return

    # Generar el resumen general global una sola vez al inicio
    generar_tabla_resumen_general()

    baselines = buscar_baselines()
    if not baselines:
        print("\nNo se encontraron archivos Baseline_*.csv")
        return

    datasets = []
    for indice, baseline in enumerate(baselines, start=1):
        dataset = extraer_dataset_desde_baseline(baseline)
        datasets.append({"nombre": dataset, "baseline": baseline})

    # Bucle del menú interactivo repetible
    while True:
        print("\n" + "=" * 50)
        print("          MENÚ DE SELECCIÓN DE DATASETS")
        print("=" * 50)
        for indice, ds in enumerate(datasets, start=1):
            print(f"  {indice}. {ds['nombre']}")
        print("  q. Salir del programa")
        print("-" * 50)

        seleccion = input("\nSelecciona el número del dataset a graficar (o 'q' para salir): ").strip().lower()

        if seleccion == 'q':
            print("\nSaliendo del programa. ¡Hasta luego!")
            break

        try:
            numero = int(seleccion)
            if 1 <= numero <= len(datasets):
                dataset_seleccionado = datasets[numero - 1]["nombre"]
                baseline_path = datasets[numero - 1]["baseline"]

                print(f"\n[Procesando] Dataset seleccionado: {dataset_seleccionado}")

                nombre_muestra_limpio = limpiar_nombre_archivo(dataset_seleccionado)
                dir_dataset = BASE_ANALYSIS_DIR / nombre_muestra_limpio

                dir_comparativas = asegurar_directorio(dir_dataset / "Comparativas")
                dir_tablas = asegurar_directorio(dir_dataset / "Tablas")
                dir_tablas_visuales = asegurar_directorio(dir_dataset / "Tablas_Visuales")
                dir_folds = asegurar_directorio(dir_dataset / "Folds")
                dir_arquitecturas = asegurar_directorio(dir_dataset / "Arquitecturas")
                dir_loss = asegurar_directorio(dir_dataset / "Loss")
                dir_estadisticas = asegurar_directorio(dir_dataset / "Estadisticas")
                dir_matrices = asegurar_directorio(dir_dataset / "Matrices_Confusion")
                dir_csvs = asegurar_directorio(dir_dataset / "CSVs")

                df_base = cargar_baseline(baseline_path)
                df_folds, df_historial, df_predicciones = cargar_deep_learning(dataset_seleccionado)

                df_resumen = generar_resumen_desde_folds(df_folds)

                for df_target in [df_resumen, df_folds]:
                    if not df_target.empty:
                        for col in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                            if col in df_target.columns:
                                df_target[col] = convertir_numerico(df_target[col])

                arquitecturas = []
                if not df_resumen.empty:
                    arquitecturas.extend(df_resumen["Arquitectura"].dropna().unique().tolist())
                if not df_folds.empty:
                    arquitecturas.extend(df_folds["Arquitectura"].dropna().unique().tolist())

                arquitecturas = list(dict.fromkeys([a for a in arquitecturas if a in ARQUITECTURAS_ESPERADAS]))
                if not df_resumen.empty:
                    for a in df_resumen["Arquitectura"].dropna().unique():
                        if a not in arquitecturas:
                            arquitecturas.append(a)

                for arquitectura in arquitecturas:
                    generar_graficas_comparativas(df_base, df_resumen, df_folds, arquitectura, dir_comparativas, dir_tablas, dir_tablas_visuales, dataset_seleccionado)

                if not df_folds.empty:
                    generar_analisis_folds(df_folds, dir_folds)

                if not df_predicciones.empty:
                    generar_analisis_matrices_confusion(df_predicciones, dir_matrices)

                if not df_resumen.empty:
                    generar_analisis_arquitecturas(df_resumen, dir_arquitecturas)

                if not df_historial.empty:
                    df_historial.to_csv(dir_csvs / "Historial_Loss_Unificado.csv", index=False, encoding="utf-8-sig")
                    generar_graficas_historial(df_historial, dir_loss)

                stats_folds = calcular_estadisticas_folds(df_folds)
                if not stats_folds.empty:
                    stats_folds.to_csv(dir_csvs / "Estadisticas_Folds.csv", index=False, encoding="utf-8-sig")

                stats_arq = calcular_estadisticas_arquitecturas(df_resumen)
                if not stats_arq.empty:
                    stats_arq.to_csv(dir_csvs / "Estadisticas_Arquitecturas.csv", index=False, encoding="utf-8-sig")

                pruebas = calcular_pruebas_estadisticas(df_folds)
                if not pruebas.empty:
                    pruebas.to_csv(dir_csvs / "Pruebas_Estadisticas.csv", index=False, encoding="utf-8-sig")
                    for arquitectura in sorted(pruebas["Arquitectura"].dropna().unique()):
                        generar_grafica_pvalues(pruebas, arquitectura, dir_estadisticas)

                if not df_resumen.empty:
                    corr = generar_correlacion(
                        df_resumen,
                        COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS,
                        "Correlacion global entre metricas de rendimiento y recursos del sistema",
                        "Correlacion_Global.png",
                        dir_estadisticas
                    )
                    if corr is not None:
                        corr.to_csv(dir_csvs / "Matriz_Correlacion.csv", encoding="utf-8-sig")

                if not df_folds.empty:
                    df_folds.to_csv(dir_csvs / "Resultados_Folds_Unificados.csv", index=False, encoding="utf-8-sig")

                if not df_resumen.empty:
                    df_resumen.to_csv(dir_csvs / "Resultados_Resumen_Unificados.csv", index=False, encoding="utf-8-sig")

                print("\n" + "=" * 70)
                print(f"ANALISIS COMPLETADO PARA: {dataset_seleccionado}")
                print(f"Directorio guardado en: {dir_dataset}")
                print("=" * 70)
            else:
                print("[Error] Número de dataset inválido. Inténtalo de nuevo.")
        except ValueError:
            print("[Error] Entrada no reconocida. Introduce un número válido o 'q'.")


if __name__ == "__main__":
    main()