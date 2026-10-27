from pathlib import Path
import json
import re
import textwrap
import warnings
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
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
# CONFIGURACIÓN Y DIRECTORIOS DINÁMICOS EN C:\EXP_TINTO
# ==========================================

BASE_DIR = Path(r"C:\EXP_TINTO")
BASE_ANALYSIS_DIR = BASE_DIR / "Graficos_Analisis"

REGISTRO_REGISTROS_PATH = (
    DATA_DIR / "Registro" / "registro_datasets.csv"
)

COLAB_RESULTS_DIR = BASE_DIR / "Results_colab"
COLAB_ANALYSIS_DIR = BASE_DIR / "Graficos_colab"

ORIGENES = {
    "1": {
        "nombre": "Local (Results)",
        "results_dir": RESULTS_DIR,
        "analysis_dir": BASE_ANALYSIS_DIR,
        "generales_dir": BASE_DIR / "Graficas_Generales",
    },
    "2": {
        "nombre": "Google Colab (Results_colab)",
        "results_dir": COLAB_RESULTS_DIR,
        "analysis_dir": COLAB_ANALYSIS_DIR,
        "generales_dir": COLAB_ANALYSIS_DIR / "Graficas_Generales",
    },
}

preparar_directorios()

# ==========================================
# MODELOS Y MÉTODOS
# ==========================================

MODELOS_BASELINE = [
    "Random_Forest",
    "SVM",
    "MLP_Clasico",
]

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

ORDEN_MODELOS = MODELOS_BASELINE + METODOS_TINTO

ARQUITECTURAS_ESPERADAS = [
    "Tabular",
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

METRICAS_RESUMEN = [
    ("Accuracy_pct", "Accuracy (%)"),
    ("F1_Score_pct", "F1-Score (%)"),
    ("ROC_AUC_pct", "ROC-AUC (%)"),
]

_PALETA = sns.color_palette("tab20", len(ORDEN_MODELOS))
COLORES_MODELOS = {m: _PALETA[i] for i, m in enumerate(ORDEN_MODELOS)}

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ==========================================
# FUNCIONES GENERALES Y NORMALIZACION ROBUSTA
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
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)
    return texto.strip("_")


def normalizar_dataset_clave(texto):
    if texto is None:
        return ""
    texto = str(texto).strip().lower()
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)
    texto = re.sub(r"[^\w\s]", "_", texto)
    texto = re.sub(r"_lote_?\d+", "", texto)
    texto = re.sub(r"_(completo|original|orig|processed|preprocessed|train|test|ds)$", "", texto)
    texto = re.sub(r"_+", "_", texto)
    return texto.strip("_")


def convertir_numerico(serie):
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def escalar_a_porcentaje(serie):
    if serie.notna().any() and serie.max(skipna=True) <= 1.0:
        return serie * 100.0
    return serie


def buscar_columna(df, candidatos):
    if df is None or df.empty:
        return None
    mapa = {normalizar_texto(col): col for col in df.columns}
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
        
        mejor_fila = None
        max_score = 0.0

        for _, fila in df_reg.iterrows():
            ds_reg = str(fila.get("Dataset", ""))
            norm_reg = normalizar_dataset_clave(ds_reg)
            
            if norm_reg == norm_obj or norm_reg in norm_obj or norm_obj in norm_reg:
                mejor_fila = fila
                break
            
            score = SequenceMatcher(None, norm_obj, norm_reg).ratio()
            if score > max_score:
                max_score = score
                mejor_fila = fila

        if mejor_fila is not None and (max_score > 0.65 or normalizar_dataset_clave(str(mejor_fila.get("Dataset", ""))) in norm_obj or norm_obj in normalizar_dataset_clave(str(mejor_fila.get("Dataset", "")))):
            inst = mejor_fila.get("Instancias", np.nan)
            feat = mejor_fila.get("Features", np.nan)
            clases = mejor_fila.get("Num_Clases", np.nan)
            cat = mejor_fila.get("Categoria", mejor_fila.get("Category", "N/A"))
            desb = mejor_fila.get("Desbalance", mejor_fila.get("Imbalance", "N/A"))
            return {
                "Dataset": mejor_fila.get("Dataset", nombre_dataset),
                "Instancias": int(round(float(inst))) if pd.notna(inst) and str(inst).replace('.','',1).isdigit() else np.nan,
                "Features": int(round(float(feat))) if pd.notna(feat) and str(feat).replace('.','',1).isdigit() else np.nan,
                "Num_Clases": int(round(float(clases))) if pd.notna(clases) and str(clases).replace('.','',1).isdigit() else np.nan,
                "Categoria": str(cat) if pd.notna(cat) else "N/A",
                "Desbalance": str(desb) if pd.notna(desb) else "N/A"
            }
    except Exception:
        pass
    return None


def meta_numerica_dataset(nombre_dataset):
    info = obtener_info_registro_dataset(nombre_dataset) or {}
    return {
        "Instancias": info.get("Instancias", np.nan),
        "Features": info.get("Features", np.nan),
        "Num_Clases": info.get("Num_Clases", np.nan),
        "Categoria": info.get("Categoria", "N/A"),
        "Desbalance": info.get("Desbalance", "N/A")
    }


# ==========================================
# DETECCIÓN ESTRICTA DE DATASETS FINALIZADOS
# ==========================================

def obtener_datasets_finalizados(results_dir):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    carpetas = [d.name for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith("Graficas")]
    baselines = list(results_dir.glob("Baseline_*.csv"))
    nombres_baselines = {extraer_dataset_desde_baseline(b) for b in baselines}

    datasets_validos = []
    for carp in carpetas:
        norm_carp = normalizar_dataset_clave(carp)
        coincide = False
        for nb in nombres_baselines:
            norm_nb = normalizar_dataset_clave(nb)
            if norm_nb == norm_carp or norm_carp in norm_nb or norm_nb in norm_carp:
                coincide = True
                break
        if coincide:
            datasets_validos.append(carp)

    return sorted(list(set(datasets_validos)))


def construir_mapeo_bases(datasets_validos):
    mapeo = {}
    for i, nombre in enumerate(datasets_validos, start=1):
        alias = f"BASE_{i}"
        info = meta_numerica_dataset(nombre)
        mapeo[nombre] = {
            "Alias": alias,
            "Dataset_Real": nombre,
            "Instancias": info.get("Instancias", np.nan),
            "Features": info.get("Features", np.nan),
            "Num_Clases": info.get("Num_Clases", np.nan),
            "Categoria": info.get("Categoria", "N/A"),
            "Desbalance": info.get("Desbalance", "N/A")
        }
    return mapeo


def guardar_tabla_mapeo_bases(mapeo_bases, generales_dir):
    dir_gen = asegurar_directorio(generales_dir)
    filas = list(mapeo_bases.values())
    df_map = pd.DataFrame(filas)
    
    for col in ["Instancias", "Features", "Num_Clases"]:
        if col in df_map.columns:
            df_map[col] = pd.to_numeric(df_map[col], errors="coerce").fillna(0).astype(int)

    ruta_csv = dir_gen / "Tabla_Mapeo_Bases.csv"
    df_map.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, max(6, len(df_map) * 0.45 + 2.5)))
    ax.axis("off")
    plt.title("Tabla de Referencia de Datasets Finalizados (Alias vs Original)", fontsize=16, fontweight="bold", pad=20)
    
    tabla = ax.table(cellText=df_map.values, colLabels=df_map.columns, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.5)
    tabla.auto_set_column_width(col=list(range(len(df_map.columns))))
    tabla.scale(1, 1.5)

    ruta_png = dir_gen / "Tabla_Mapeo_Bases.png"
    plt.savefig(ruta_png, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    return df_map


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
        "tinto": "TINTO", "igtd": "IGTD", "refined": "REFINED", "bargraph": "BARGRAPH",
        "bar_graph": "BARGRAPH", "distancematrix": "DISTANCEMATRIX", "distance_matrix": "DISTANCEMATRIX",
        "combination": "COMBINATION", "supertml": "SUPERTML", "featurewrap": "FEATUREWRAP",
        "bie": "BIE", "fotomics": "FOTOMICS", "deepinsight": "DEEPINSIGHT"
    }
    if texto in equivalencias:
        return equivalencias[texto]
    for clave, resultado in equivalencias.items():
        if clave in texto:
            return resultado
    return None


# ==========================================
# CARGA DE DATOS
# ==========================================

def encontrar_archivo_baseline_para_dataset(dataset_nombre, results_dir):
    results_dir = Path(results_dir)
    norm_ds = normalizar_dataset_clave(dataset_nombre)
    for baseline_path in results_dir.glob("Baseline_*.csv"):
        nb = extraer_dataset_desde_baseline(baseline_path)
        norm_nb = normalizar_dataset_clave(nb)
        if norm_nb == norm_ds or norm_ds in norm_nb or norm_nb in norm_ds:
            return baseline_path
    return None


def extraer_dataset_desde_baseline(ruta):
    nombre = ruta.stem
    if nombre.lower().startswith("baseline_"):
        nombre = nombre[len("Baseline_"):]
    return nombre


def cargar_baseline(ruta):
    df = leer_csv_seguro(ruta)
    if df.empty:
        return pd.DataFrame()

    # Los experimentos actuales no utilizan SMOTE.
    col_estrategia = buscar_columna(df, ["Estrategia", "Strategy"])
    if col_estrategia is not None:
        df = df[df[col_estrategia].astype(str).str.strip().str.upper() != "SMOTE"].copy()
        if df.empty:
            return pd.DataFrame()

    resultado = pd.DataFrame(index=df.index)
    resultado["Tipo"] = "Baseline"
    columna_modelo = buscar_columna(df, ["Modelo", "model", "Algoritmo"])
    resultado["Modelo"] = df[columna_modelo].astype(str).str.strip() if columna_modelo else ""
    resultado["Arquitectura"] = "Tabular"
    resultado["Metodo_TINTO"] = "BASELINE"

    for metrica in COLUMNAS_METRICAS:
        serie = obtener_columna_numerica(df, [metrica, f"Baseline_{metrica}", metrica.lower()])
        resultado[metrica] = escalar_a_porcentaje(serie)

    t_train = obtener_columna_numerica(df, ["Tiempo_Train_s", "Tiempo_s"])
    t_eval = obtener_columna_numerica(df, ["Tiempo_Eval_s"])
    resultado["Tiempo_Train_s"] = t_train
    resultado["Tiempo_Eval_s"] = t_eval
    resultado["Tiempo_s"] = t_train + t_eval.fillna(0)
    resultado["RAM_GB"] = obtener_columna_numerica(df, ["RAM_GB", "Max_RAM_GB"])
    resultado["GPU_Usage_%"] = obtener_columna_numerica(df, ["GPU_Usage_%", "GPU_RAM_GB"])
    return resultado


def cargar_experiment_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception:
        return {}


def extraer_metodos_experiment(experiment):
    metodos = experiment.get("metodos_evaluados", [])
    if isinstance(metodos, str):
        metodos = [metodos]
    if not isinstance(metodos, list):
        return []
    return list(dict.fromkeys([normalizar_metodo(m) for m in metodos if normalizar_metodo(m) is not None]))


def cargar_resultados_folds_de_carpeta(carpeta_dataset):
    carpeta_dataset = Path(carpeta_dataset)
    if not carpeta_dataset.exists() or not carpeta_dataset.is_dir():
        return pd.DataFrame()

    folds = []
    for exp_json in carpeta_dataset.rglob("experiment.json"):
        carpeta_exp = exp_json.parent
        datos_json = cargar_experiment_json(exp_json)
        arq = normalizar_arquitectura(datos_json.get("red_neuronal") or datos_json.get("modelo_red") or datos_json.get("arquitectura"))
        if arq is None:
            arq = detectar_arquitectura_en_ruta(carpeta_exp)
        if arq is None:
            continue

        ruta_folds = carpeta_exp / "resultados_folds.csv"
        if not ruta_folds.exists():
            continue
        df_f = leer_csv_seguro(ruta_folds)
        if df_f.empty:
            continue

        res_f = pd.DataFrame(index=df_f.index)
        res_f["Tipo"] = "Deep Learning"
        res_f["Arquitectura"] = arq

        col_metodo = buscar_columna(df_f, ["Metodo_TINTO", "Metodo", "Method"])
        metodos_exp = extraer_metodos_experiment(datos_json)
        if col_metodo is not None:
            res_f["Metodo_TINTO"] = df_f[col_metodo].astype(str).apply(normalizar_metodo)
        elif len(metodos_exp) == 1:
            res_f["Metodo_TINTO"] = metodos_exp[0]
        else:
            res_f["Metodo_TINTO"] = np.nan

        res_f["Modelo"] = res_f["Metodo_TINTO"].fillna("").astype(str)

        for metrica in COLUMNAS_METRICAS:
            serie = obtener_columna_numerica(df_f, [metrica, metrica.lower()])
            res_f[metrica] = escalar_a_porcentaje(serie)

        t_train = obtener_columna_numerica(df_f, ["Tiempo_Entrenamiento_s", "Tiempo_Train_s", "Training_Time_s"])
        t_eval = obtener_columna_numerica(df_f, ["Tiempo_Evaluacion_s", "Tiempo_Eval_s", "Evaluation_Time_s"])
        res_f["Tiempo_s"] = t_train + t_eval.fillna(0)
        res_f["Tiempo_Train_s"] = t_train
        res_f["Tiempo_Eval_s"] = t_eval
        res_f["RAM_GB"] = obtener_columna_numerica(df_f, ["Max_RAM_GB", "RAM_GB"])
        res_f["GPU_Usage_%"] = obtener_columna_numerica(df_f, ["Max_VRAM_GB", "GPU_Usage_%", "Avg_CPU_Percent"])

        folds.append(res_f)

    if not folds:
        return pd.DataFrame()
    return pd.concat(folds, ignore_index=True)


def crear_estructura_modelos(df_base, df_dl, arquitectura):
    filas = []
    if arquitectura == "Tabular":
        for modelo in MODELOS_BASELINE:
            modelo_norm = normalizar_texto(modelo)
            datos_modelo = df_base[df_base["Modelo"].apply(normalizar_texto) == modelo_norm] if not df_base.empty else pd.DataFrame()
            fila = {"Tipo": "Baseline", "Arquitectura": "Tabular", "Modelo": modelo, "Metodo_TINTO": "BASELINE"}
            for col in COLUMNAS_METRICAS + ["Tiempo_s", "Tiempo_Train_s", "Tiempo_Eval_s", "RAM_GB", "GPU_Usage_%"]:
                if col in datos_modelo.columns:
                    fila[col] = datos_modelo[col].mean()
                else:
                    fila[col] = np.nan
            filas.append(fila)
    else:
        for metodo in METODOS_TINTO:
            fila = {"Tipo": "Deep Learning", "Arquitectura": arquitectura, "Modelo": metodo, "Metodo_TINTO": metodo}
            datos = df_dl[(df_dl["Arquitectura"] == arquitectura) & (df_dl["Metodo_TINTO"] == metodo)] if not df_dl.empty else pd.DataFrame()
            for col in COLUMNAS_METRICAS + ["Tiempo_s", "Tiempo_Train_s", "Tiempo_Eval_s", "RAM_GB", "GPU_Usage_%"]:
                if col in datos.columns:
                    fila[col] = datos[col].mean()
                else:
                    fila[col] = np.nan
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
# RECOPILACIÓN GLOBAL
# ==========================================

def recopilar_resultados_globales(results_dir, datasets_validos, mapeo_bases):
    print("\n" + "=" * 70)
    print("RECOPILANDO RESULTADOS GLOBALES DE DATASETS FINALIZADOS")
    print("=" * 70)

    results_dir = Path(results_dir)
    bloques = []

    for ds_nombre in datasets_validos:
        alias_info = mapeo_bases.get(ds_nombre, {})
        alias = alias_info.get("Alias", ds_nombre)

        baseline_path = encontrar_archivo_baseline_para_dataset(ds_nombre, results_dir)
        df_base = cargar_baseline(baseline_path) if baseline_path else pd.DataFrame()

        carpeta_ds = results_dir / ds_nombre
        df_dl = cargar_resultados_folds_de_carpeta(carpeta_ds)

        arquitecturas = ["Tabular"] + list(df_dl["Arquitectura"].dropna().unique())
        arquitecturas = list(dict.fromkeys(arquitecturas))

        for arch in arquitecturas:
            df_modelos = crear_estructura_modelos(df_base, df_dl, arch)
            df_modelos = normalizar_metricas_porcentaje(df_modelos)
            
            df_modelos["Dataset"] = alias
            df_modelos["Dataset_Real"] = ds_nombre
            df_modelos["Instancias"] = int(alias_info.get("Instancias", 0)) if pd.notna(alias_info.get("Instancias")) else np.nan
            df_modelos["Features"] = int(alias_info.get("Features", 0)) if pd.notna(alias_info.get("Features")) else np.nan
            df_modelos["Num_Clases"] = int(alias_info.get("Num_Clases", 0)) if pd.notna(alias_info.get("Num_Clases")) else np.nan
            df_modelos["Categoria"] = alias_info.get("Categoria", "N/A")
            df_modelos["Desbalance"] = alias_info.get("Desbalance", "N/A")
            bloques.append(df_modelos)

    if not bloques:
        return pd.DataFrame()
    return pd.concat(bloques, ignore_index=True)


def tabla_datasets(df_global):
    t = (
        df_global
        .groupby("Dataset", sort=False)[["Instancias", "Features", "Num_Clases", "Categoria", "Desbalance", "Dataset_Real"]]
        .first()
        .reset_index()
    )
    t["_orden"] = t["Instancias"].fillna(np.inf)
    return t.sort_values("_orden", kind="stable").drop(columns="_orden").reset_index(drop=True)


def pivot_modelos_datasets(df_arch, metrica, tabla):
    p = df_arch.pivot_table(index="Modelo", columns="Dataset", values=metrica, aggfunc="mean")
    filas = [m for m in ORDEN_MODELOS if m in p.index]
    cols = [d for d in tabla["Dataset"] if d in p.columns]
    p = p.reindex(index=filas, columns=cols)
    return p.dropna(axis=1, how="all").dropna(axis=0, how="all")


# ==========================================
# RESUMEN GENERAL Y ANÁLISIS DE VICTORIAS
# ==========================================

def generar_tabla_resumen_general(df_global, generales_dir):
    print("\n[Global] Generando Resumen General de Mejores Modelos y Análisis de Victorias...")
    if df_global is None or df_global.empty:
        return

    resultados = []
    for (ds, arch), g in df_global.groupby(["Dataset", "Arquitectura"], sort=False):
        if "Accuracy_pct" not in g.columns or not g["Accuracy_pct"].notna().any():
            continue
        mejor = g.loc[g["Accuracy_pct"].idxmax()]
        
        tipo_ganador = "Baseline" if mejor.get("Modelo") in MODELOS_BASELINE else "Deep Learning"
        
        inst = mejor.get("Instancias", np.nan)
        feat = mejor.get("Features", np.nan)
        clases = mejor.get("Num_Clases", np.nan)
        tiempo = mejor.get("Tiempo_s", np.nan)

        resultados.append({
            "Dataset": ds,
            "Dataset_Real": mejor.get("Dataset_Real", ds),
            "Arquitectura": arch,
            "Mejor_Modelo": mejor.get("Modelo", "N/A"),
            "Tipo_Ganador": tipo_ganador,
            "Accuracy_%": round(mejor.get("Accuracy_pct", np.nan), 3),
            "F1_Score_%": round(mejor.get("F1_Score_pct", np.nan), 3),
            "ROC_AUC_%": round(mejor.get("ROC_AUC_pct", np.nan), 3),
            "Tiempo_Total_s": round(tiempo, 3) if pd.notna(tiempo) else np.nan,
            "Instancias": int(round(float(inst))) if pd.notna(inst) else np.nan,
            "Features": int(round(float(feat))) if pd.notna(feat) else np.nan,
            "Num_Clases": int(round(float(clases))) if pd.notna(clases) else np.nan,
            "Categoria": mejor.get("Categoria", "N/A"),
            "Desbalance": mejor.get("Desbalance", "N/A"),
        })

    if not resultados:
        return

    df_res = pd.DataFrame(resultados)
    
    df_res["Instancias_num"] = pd.to_numeric(df_res["Instancias"], errors="coerce")
    df_res["Features_num"] = pd.to_numeric(df_res["Features"], errors="coerce")
    df_res["Num_Clases_num"] = pd.to_numeric(df_res["Num_Clases"], errors="coerce")
    df_res["Tiempo_Total_s"] = pd.to_numeric(df_res["Tiempo_Total_s"], errors="coerce")

    df_res_sorted = df_res.sort_values(by="Instancias_num", ascending=True).drop(columns=["Instancias_num", "Features_num", "Num_Clases_num"]).reset_index(drop=True)

    dir_gen = asegurar_directorio(generales_dir)
    df_res_sorted.to_csv(dir_gen / "Resumen_Mejores_Modelos_Global.csv", index=False, encoding="utf-8-sig")

    resumen_tipo = df_res.groupby(["Arquitectura", "Tipo_Ganador"]).agg(
        Total_Victorias=("Dataset", "count"),
        Promedio_Instancias=("Instancias_num", "mean"),
        Promedio_Features=("Features_num", "mean"),
        Promedio_Clases=("Num_Clases_num", "mean"),
        Promedio_Tiempo_s=("Tiempo_Total_s", "mean")
    ).reset_index()

    resumen_tipo["Promedio_Instancias"] = resumen_tipo["Promedio_Instancias"].round(0).astype(int)
    resumen_tipo["Promedio_Features"] = resumen_tipo["Promedio_Features"].round(0).astype(int)
    resumen_tipo["Promedio_Clases"] = resumen_tipo["Promedio_Clases"].round(0).astype(int)
    resumen_tipo["Promedio_Tiempo_s"] = resumen_tipo["Promedio_Tiempo_s"].round(2)

    resumen_tipo.to_csv(dir_gen / "Resumen_Victorias_Baseline_vs_DL.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, max(4, len(resumen_tipo) * 0.6 + 2)))
    ax.axis("off")
    plt.title("Resumen de Victorias: Baselines vs Deep Learning por Características", fontsize=14, fontweight="bold", pad=15)
    ax.table(cellText=resumen_tipo.values, colLabels=resumen_tipo.columns, loc="center", cellLoc="center").auto_set_font_size(False)
    plt.savefig(dir_gen / "Resumen_Victorias_Baseline_vs_DL.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()


# ==========================================
# TAMAÑO DE DATASETS (INCLUYE LOS 19 DATASETS)
# ==========================================

def generar_grafica_tamano_datasets_multicriterio(tabla, output_dir):
    criterios = [
        ("Instancias", "Número de Instancias"),
        ("Features", "Número de Features"),
        ("Num_Clases", "Número de Clases")
    ]
    
    for col, label_eje in criterios:
        t = tabla.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
        if t.empty:
            continue

        valores = t[col].astype(int).values
        fig, ax = plt.subplots(figsize=(13, max(4, len(t) * 0.55 + 2)))
        y = np.arange(len(t))
        
        clases = t["Num_Clases"] if "Num_Clases" in t.columns else pd.Series(1, index=t.index)
        norm = plt.Normalize(clases.min(), clases.max()) if clases.notna().any() else None
        cmap = mpl.colormaps['viridis']
        colores = [cmap(norm(c)) if (norm is not None and pd.notna(c)) else "skyblue" for c in clases]

        ax.barh(y, valores, color=colores, edgecolor="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(t["Dataset"], fontsize=10, fontweight="bold")
        ax.invert_yaxis()

        if valores.min() > 0 and valores.max() / (valores.min() + 1e-5) > 50:
            ax.set_xscale("log")
            ax.set_xlabel(f"{label_eje} (escala logarítmica)", fontsize=12, fontweight="bold")
        else:
            ax.set_xlabel(label_eje, fontsize=12, fontweight="bold")

        for i, fila in t.iterrows():
            ax.annotate(
                f"Inst={int(fila['Instancias']):,} | Feat={int(fila['Features'])} | Clases={int(fila['Num_Clases'])}",
                (fila[col], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9
            )
        ax.margins(x=0.45)

        if norm is not None and col != "Num_Clases":
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Número de Clases", pad=0.01)

        ax.set_title(f"Datasets ordenados por {label_eje} (Mayor a Menor)", fontsize=15, fontweight="bold")
        plt.tight_layout()
        ruta = output_dir / f"Tamano_Datasets_por_{col}.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close()


# ==========================================
# GRÁFICAS DE TENDENCIA (LÍNEAS) - MÁS ANCHAS
# ==========================================

def _dibujar_lineas_vertical(ax, pivot, x_labels, ylabel, titulo=None):
    x = np.arange(pivot.shape[1])
    for modelo in pivot.index:
        es_base = modelo in MODELOS_BASELINE
        ax.plot(
            x, pivot.loc[modelo].values,
            color=COLORES_MODELOS[modelo],
            linewidth=1.3,
            linestyle="--" if es_base else "-",
            marker="s" if es_base else "o",
            markersize=4,
            alpha=0.9,
            label=modelo,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    
    ymin, ymax = ax.get_ylim()
    ymin = max(0, (int(ymin // 10)) * 10)
    ymax = min(100, (int(ymax // 10) + 1) * 10)
    if ymax <= ymin:
        ymax = ymin + 10
    ax.set_yticks(np.arange(ymin, ymax + 1, 10))

    if titulo:
        ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.35)


def generar_lineas_datasets_ordenadas(df_arch, arquitectura, tabla, output_dir):
    criterios_orden = [
        ("Instancias", "Instancias"),
        ("Features", "Features"),
        ("Num_Clases", "Clases"),
        ("Desbalance", "Desbalance"),
        ("Categoria", "Categoria")
    ]

    for col_metrica, label_metrica in METRICAS_RESUMEN:
        if col_metrica not in df_arch.columns:
            continue

        for criterio_col, criterio_nombre in criterios_orden:
            if criterio_col == "Desbalance":
                # Orden numérico cuando el registro contiene ratios/porcentajes;
                # si contiene categorías, conservar el orden alfabético.
                valores = tabla[criterio_col].astype(str).str.strip()
                numeros = pd.to_numeric(valores.str.replace(",", ".", regex=False).str.rstrip("%"), errors="coerce")
                if numeros.notna().any():
                    t_ord = (tabla.assign(_orden_desbalance=numeros)
                             .sort_values(by="_orden_desbalance", ascending=True, na_position="last")
                             .drop(columns="_orden_desbalance").reset_index(drop=True))
                else:
                    t_ord = tabla.sort_values(by=criterio_col, ascending=True, na_position="last").reset_index(drop=True)
            else:
                t_ord = tabla.sort_values(by=criterio_col, ascending=True, na_position="last").reset_index(drop=True)
            p = pivot_modelos_datasets(df_arch, col_metrica, t_ord)
            if p.empty:
                continue

            x_labels = list(p.columns)

            fig, ax = plt.subplots(figsize=(18, 8))
            _dibujar_lineas_vertical(
                ax, p, x_labels, label_metrica,
                f"{label_metrica} - {arquitectura}\n(Ordenado por {criterio_nombre})"
            )
            ax.set_xlabel("Datasets (BASE_X)", fontsize=11, fontweight="bold")
            
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="Modelo")
            plt.subplots_adjust(right=0.85, bottom=0.2)

            pref = limpiar_nombre_archivo(arquitectura)
            ruta = output_dir / f"Lineas_{pref}_{limpiar_nombre_archivo(col_metrica)}_ordenado_por_{criterio_nombre}.png"
            plt.savefig(ruta, dpi=300, bbox_inches="tight")
            plt.close()


# ==========================================
# DISTRIBUCIÓN DE MODELOS
# ==========================================

def generar_distribucion_modelos(df_arch, arquitectura, columna, label, output_dir):
    datos = df_arch[["Modelo", "Dataset", columna]].dropna()
    if datos.empty:
        return

    medianas = datos.groupby("Modelo")[columna].median().sort_values(ascending=False)
    orden = list(medianas.index)
    listas = [datos.loc[datos["Modelo"] == m, columna].values for m in orden]

    fig, ax = plt.subplots(figsize=(15, 8))
    bp = ax.boxplot(listas, patch_artist=True, widths=0.6, showfliers=False)
    for parche, modelo in zip(bp["boxes"], orden):
        parche.set_facecolor(COLORES_MODELOS[modelo])
        parche.set_alpha(0.75)
        if modelo in MODELOS_BASELINE:
            parche.set_hatch("//")
    for mediana in bp["medians"]:
        mediana.set_color("black")
        mediana.set_linewidth(2)

    rng = np.random.default_rng(0)
    for i, valores in enumerate(listas, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(valores)), valores, color="black", alpha=0.5, s=14, zorder=3)

    ax.set_xticks(range(1, len(orden) + 1))
    ax.set_xticklabels(orden, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Modelos (Mejor a peor)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{label} (↑ mayor es mejor)", fontsize=12, fontweight="bold")
    
    ymin, ymax = ax.get_ylim()
    ymin = max(0, (int(ymin // 10)) * 10)
    ymax = min(100, (int(ymax // 10) + 1) * 10)
    if ymax <= ymin:
        ymax = ymin + 10
    ax.set_yticks(np.arange(ymin, ymax + 1, 10))

    titulo_grafica = f"DISTRIBUCION DE ACCURACY (%) {arquitectura}" if "Accuracy" in label else f"DISTRIBUCION DE {label.upper()} {arquitectura}"
    ax.set_title(titulo_grafica, fontsize=14, fontweight="bold")
    plt.tight_layout()

    ruta = output_dir / f"Distribucion_Modelos_{limpiar_nombre_archivo(arquitectura)}_{limpiar_nombre_archivo(columna)}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# RANKING GLOBAL
# ==========================================

def generar_ranking_global(df_arch, arquitectura, columna, label, tabla, output_dir):
    p = pivot_modelos_datasets(df_arch, columna, tabla)
    if p.shape[0] < 2 or p.shape[1] < 1:
        return None, None

    completos = p.dropna(axis=1)
    rangos = completos.rank(axis=0, ascending=False, method="average") if completos.shape[1] >= 2 else p.rank(axis=0, ascending=False, method="average")
    
    resumen = pd.DataFrame({
        "Modelo": rangos.index,
        "Rango_Medio": rangos.mean(axis=1).values,
        "Rango_Std": rangos.std(axis=1, ddof=0, skipna=True).values,
        "Victorias": (rangos <= 1.0).sum(axis=1).values,
        "Valor_Medio": p.mean(axis=1).values,
        "N_Datasets": p.notna().sum(axis=1).values,
    }).sort_values("Rango_Medio", ascending=True).reset_index(drop=True)
    
    resumen["Tipo"] = np.where(resumen["Modelo"].isin(MODELOS_BASELINE), "Baseline", "Deep Learning")

    prefijo = f"{limpiar_nombre_archivo(arquitectura)}_{limpiar_nombre_archivo(columna)}"
    resumen.to_csv(output_dir / f"Ranking_{prefijo}.csv", index=False, encoding="utf-8-sig")

    n = len(resumen)
    cmap_rdylgn = mpl.colormaps['RdYlGn']
    colores = [cmap_rdylgn(0.85 * (1 - (i / max(n - 1, 1))) + 0.05) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    ax = axes[0]
    barras = ax.barh(resumen["Modelo"], resumen["Rango_Medio"], xerr=resumen["Rango_Std"],
                     color=colores, edgecolor="black", capsize=3)
    ax.invert_yaxis()
    for i, v in enumerate(resumen["Rango_Medio"]):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Rango medio (1 = mejor posición promedio)", fontsize=11, fontweight="bold")
    ax.set_title("Rango Medio General (menor es mejor)", fontsize=13, fontweight="bold")

    ax = axes[1]
    resumen_vic = resumen.sort_values("Victorias", ascending=False).reset_index(drop=True)
    barras_v = ax.barh(resumen_vic["Modelo"], resumen_vic["Victorias"], color=colores, edgecolor="black")
    ax.invert_yaxis()
    for i, v in enumerate(resumen_vic["Victorias"]):
        ax.text(v + 0.05, i, f"{int(v)}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Cantidad de datasets donde quedó en 1er lugar", fontsize=11, fontweight="bold")
    ax.set_title("Número de Victorias (mayor es mejor)", fontsize=13, fontweight="bold")

    fig.suptitle(f"Ranking global por {label} - {arquitectura}\n*Nota: Se muestran los promedios generales de rangos y victorias entre datasets.", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta = output_dir / f"Ranking_{prefijo}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    return resumen, None


# ==========================================
# PARETO
# ==========================================

def calcular_frente_pareto(x, y):
    orden = np.argsort(x, kind="stable")
    mejor_y = -np.inf
    frente = []
    for i in orden:
        if y[i] > mejor_y:
            frente.append(i)
            mejor_y = y[i]
    return frente


def generar_scatter_pareto(df, x_col, y_col, titulo, nombre_archivo, output_dir, x_label, y_label):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    datos = df[[x_col, y_col, "Modelo", "Tipo"]].dropna().reset_index(drop=True)
    if len(datos) < 2:
        return

    x = datos[x_col].astype(float).values
    y = datos[y_col].astype(float).values
    frente = calcular_frente_pareto(x, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    for tipo, marcador in [("Baseline", "s"), ("Deep Learning", "o")]:
        sub = datos[datos["Tipo"] == tipo]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[y_col], s=90, marker=marcador, alpha=0.85,
                   c=[COLORES_MODELOS.get(m, "gray") for m in sub["Modelo"]],
                   edgecolor="black", linewidth=0.8, label=tipo)

    xf = x[frente]
    yf = y[frente]
    ax.step(xf, yf, where="post", color="black", linewidth=1.2, linestyle="--", label="Frente de Pareto")
    ax.scatter(xf, yf, s=200, facecolors="none", edgecolors="black", linewidths=1.5)

    for i, fila in datos.iterrows():
        negrita = i in frente
        ax.annotate(str(fila["Modelo"]), (fila[x_col], fila[y_col]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, fontweight="bold" if negrita else "normal")

    if x.max() / (x.min() + 1e-5) > 20:
        ax.set_xscale("log")
        x_label = x_label + " (Escala Logarítmica)"

    ax.set_title(f"{titulo}\n(Promedios generales de eficiencia)", fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label + " (↓ menor es mejor)", fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label + " (↑ mayor es mejor)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()


def generar_pareto_global(df_arch, arquitectura, output_dir):
    if "Tiempo_s" not in df_arch.columns or "Accuracy_pct" not in df_arch.columns:
        return

    agregado = df_arch.groupby("Modelo").agg(
        Accuracy_pct=("Accuracy_pct", "mean"),
        Tiempo_s=("Tiempo_s", "mean"),
        Tipo=("Tipo", "first"),
    ).reset_index()

    generar_scatter_pareto(
        agregado, "Tiempo_s", "Accuracy_pct",
        f"Eficiencia global (Promedios): Accuracy vs Tiempo - {arquitectura}",
        f"Pareto_Global_{limpiar_nombre_archivo(arquitectura)}.png",
        output_dir, "Tiempo promedio (s)", "Accuracy medio (%)"
    )


# ==========================================
# NUEVAS GRÁFICAS DE ARQUITECTURAS Y PANELES DE TIEMPO INDEPENDIENTES
# ==========================================

def generar_graficas_comparativas_arquitecturas(df_global, generales_dir):
    if df_global is None or df_global.empty:
        return

    dir_gen = asegurar_directorio(Path(generales_dir) / "Comparativa_Arquitecturas")

    # Mostrar cada baseline por separado en lugar de agruparlos como "Tabular".
    df_global = df_global.copy()
    df_global.loc[df_global["Arquitectura"] == "Tabular", "Arquitectura"] = df_global.loc[
        df_global["Arquitectura"] == "Tabular", "Modelo"
    ]
    
    # 1. Rendimiento promedio (Incluye Tabular / Baselines y Deep Learning)
    resumen_metricas = df_global.groupby("Arquitectura")[["Accuracy_pct", "F1_Score_pct", "ROC_AUC_pct"]].mean().reset_index()
    if not resumen_metricas.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(resumen_metricas))
        width = 0.25
        
        ax.bar(x - width, resumen_metricas["Accuracy_pct"], width, label="Accuracy (%)", color="#1E90FF", edgecolor="black")
        ax.bar(x, resumen_metricas["F1_Score_pct"], width, label="F1-Score (%)", color="#556B2F", edgecolor="black")
        ax.bar(x + width, resumen_metricas["ROC_AUC_pct"], width, label="ROC-AUC (%)", color="#4682B4", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(resumen_metricas["Arquitectura"], fontweight="bold")
        ax.set_ylabel("Porcentaje (%)", fontsize=11, fontweight="bold")
        ax.set_title("Comparativa de Rendimiento Promedio por Arquitectura y Baselines", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(dir_gen / "Comparativa_Rendimiento_Arquitecturas.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 2. Gráficos de Tiempo con Paneles Independientes (A. Entrenamiento y B. Clasificación) + Tabulares
    if "Tiempo_Train_s" in df_global.columns and "Tiempo_Eval_s" in df_global.columns:
        resumen_tiempos = df_global.groupby("Arquitectura")[["Tiempo_Train_s", "Tiempo_Eval_s", "Tiempo_s"]].mean().reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: Tiempo de entrenamiento (escala independiente)
        arq_list = resumen_tiempos["Arquitectura"].values
        train_vals = resumen_tiempos["Tiempo_Train_s"].fillna(0).values
        y_pos = np.arange(len(arq_list))

        ax1.barh(y_pos, train_vals, color="#1E90FF", edgecolor="black")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(arq_list, fontweight="bold")
        ax1.set_xlabel("Segundos (s)", fontsize=11, fontweight="bold")
        ax1.set_title("A. Tiempo de entrenamiento\nEscala independiente · segundos", fontsize=12, fontweight="bold")
        for i, val in enumerate(train_vals):
            ax1.text(val * 1.02, i, f"{val:.1f} s", va="center", fontsize=9, fontweight="bold")

        # Panel B: Tiempo de clasificación (escala independiente)
        eval_vals = resumen_tiempos["Tiempo_Eval_s"].fillna(0).values
        ax2.barh(y_pos, eval_vals, color="#556B2F", edgecolor="black")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(arq_list, fontweight="bold")
        ax2.set_xlabel("Segundos (s)", fontsize=11, fontweight="bold")
        ax2.set_title("B. Tiempo de clasificación\nEscala independiente · segundos", fontsize=12, fontweight="bold")
        for i, val in enumerate(eval_vals):
            ax2.text(val * 1.02, i, f"{val:.1f} s", va="center", fontsize=9, fontweight="bold")

        fig.suptitle("Comparativa de Tiempos por Arquitectura y Baselines\n(Nota: Las longitudes se calculan con escalas diferentes en cada panel)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(dir_gen / "Tiempos_Paneles_Independientes_Entrenamiento_Clasificacion.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Recursos (RAM y GPU)
    if "RAM_GB" in df_global.columns and "GPU_Usage_%" in df_global.columns:
        resumen_recursos = df_global.groupby("Arquitectura")[["RAM_GB", "GPU_Usage_%"]].mean().reset_index()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(resumen_recursos))
        width = 0.35

        ax1.bar(x - width/2, resumen_recursos["RAM_GB"], width, label="RAM (GB)", color="#4682B4", edgecolor="black")
        ax1.set_ylabel("RAM Promedio (GB)", fontsize=11, fontweight="bold", color="#4682B4")
        ax1.set_xticks(x)
        ax1.set_xticklabels(resumen_recursos["Arquitectura"], fontweight="bold")

        ax2 = ax1.twinx()
        ax2.bar(x + width/2, resumen_recursos["GPU_Usage_%"], width, label="GPU / VRAM Usage", color="#556B2F", edgecolor="black")
        ax2.set_ylabel("GPU / VRAM Promedio", fontsize=11, fontweight="bold", color="#556B2F")

        plt.title("Comparativa de Recursos (RAM y GPU) por Arquitectura", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(dir_gen / "Comparativa_Recursos_Arquitecturas.png", dpi=300, bbox_inches="tight")
        plt.close()


def generar_grafica_lineas_tiempo_gpu(df_global, generales_dir):
    if df_global is None or df_global.empty:
        return

    dir_gen = asegurar_directorio(Path(generales_dir) / "Comparativa_Arquitecturas")
    df_dl = df_global[df_global["Tipo"] == "Deep Learning"].dropna(subset=["Tiempo_s", "Accuracy_pct", "GPU_Usage_%"])
    if df_dl.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(
        df_dl["Tiempo_s"], df_dl["Accuracy_pct"],
        c=df_dl["GPU_Usage_%"], cmap="plasma", s=70, edgecolor="black", alpha=0.85
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Uso de GPU / VRAM", fontweight="bold")

    if df_dl["Tiempo_s"].max() / (df_dl["Tiempo_s"].min() + 1e-5) > 20:
        ax.set_xscale("log")
        ax.set_xlabel("Tiempo General (s) - Escala Log", fontsize=11, fontweight="bold")
    else:
        ax.set_xlabel("Tiempo General (s)", fontsize=11, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Relación Tiempo vs Accuracy coloreado por Uso de GPU/VRAM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(dir_gen / "Relacion_Tiempo_Accuracy_GPU.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# ORQUESTADOR GLOBAL
# ==========================================

def generar_graficas_globales(df_global, generales_dir, mapeo_bases):
    if df_global is None or df_global.empty:
        return

    print("\n" + "=" * 70)
    print("GENERANDO COMPARATIVAS ENTRE DATASETS FINALIZADOS")
    print("=" * 70)

    base = asegurar_directorio(Path(generales_dir) / "Comparativa_entre_Datasets")
    
    guardar_tabla_mapeo_bases(mapeo_bases, base)

    tabla = tabla_datasets(df_global)
    tabla.to_csv(base / "Tabla_Datasets.csv", index=False, encoding="utf-8-sig")

    generar_grafica_tamano_datasets_multicriterio(tabla, base)
    generar_graficas_comparativas_arquitecturas(df_global, generales_dir)
    generar_grafica_lineas_tiempo_gpu(df_global, generales_dir)

    arquitecturas = [a for a in sorted(df_global["Arquitectura"].dropna().unique()) if a != "Tabular"]
    if not arquitecturas:
        arquitecturas = ["Tabular"]

    for arquitectura in arquitecturas:
        df_arch = df_global[df_global["Arquitectura"] == arquitectura]
        # Incluir los tres baselines individualmente para los mismos datasets.
        if arquitectura != "Tabular":
            baselines = df_global[(df_global["Arquitectura"] == "Tabular") &
                                  (df_global["Dataset"].isin(df_arch["Dataset"]))]
            if not baselines.empty:
                df_arch = pd.concat([df_arch, baselines], ignore_index=True)

        carpeta = asegurar_directorio(base / limpiar_nombre_archivo(arquitectura))
        print(f"\n[Global] Arquitectura: {arquitectura}")

        generar_pareto_global(df_arch, arquitectura, carpeta)
        generar_lineas_datasets_ordenadas(df_arch, arquitectura, tabla, carpeta)

        for columna, label in METRICAS_RESUMEN:
            if columna not in df_arch.columns:
                continue
            generar_distribucion_modelos(df_arch, arquitectura, columna, label, carpeta)
            generar_ranking_global(df_arch, arquitectura, columna, label, tabla, carpeta)


# ==========================================
# FLUJO DE PROCESAMIENTO DE DATASET INDIVIDUAL
# ==========================================

def procesar_dataset(dataset_seleccionado, baseline_path, origen):
    pass


# ==========================================
# MAIN PRINCIPAL
# ==========================================

def seleccionar_origen():
    while True:
        print("\n" + "=" * 50)
        print("        ORIGEN DE LOS RESULTADOS")
        print("=" * 50)
        for clave, origen in ORIGENES.items():
            print(f"  {clave}. {origen['nombre']}  ->  {origen['results_dir']}")
        print("  q. Salir del programa")
        print("-" * 50)

        seleccion = input("\nSelecciona el origen de los resultados: ").strip().lower()
        if seleccion == "q":
            return None
        if seleccion in ORIGENES:
            origen = ORIGENES[seleccion]
            if not Path(origen["results_dir"]).exists():
                print(f"\n[Error] No existe la carpeta: {origen['results_dir']}")
                continue
            return origen
        print("[Error] Opción inválida.")


def main():
    print("\n" + "=" * 70)
    print("        ANALISIS VISUAL Y ESTADISTICO TINTOlib")
    print("=" * 70)

    salir = False
    while not salir:
        origen = seleccionar_origen()
        if origen is None:
            break

        results_dir = Path(origen["results_dir"])
        asegurar_directorio(origen["analysis_dir"])

        datasets_validos = obtener_datasets_finalizados(results_dir)
        if not datasets_validos:
            print("\nNo se encontraron datasets finalizados (carpetas en Results con su respectivo Baseline).")
            continue

        mapeo_bases = construir_mapeo_bases(datasets_validos)
        df_global = recopilar_resultados_globales(results_dir, datasets_validos, mapeo_bases)
        
        generar_tabla_resumen_general(df_global, origen["generales_dir"])
        generar_graficas_globales(df_global, origen["generales_dir"], mapeo_bases)

        datasets = [{"nombre": ds, "baseline": encontrar_archivo_baseline_para_dataset(ds, results_dir)} for ds in datasets_validos]

        while True:
            print("\n" + "=" * 50)
            print(f"   MENÚ DE DATASETS FINALIZADOS  [{origen['nombre']}]")
            print("=" * 50)
            for indice, ds in enumerate(datasets, start=1):
                alias = mapeo_bases.get(ds['nombre'], {}).get('Alias', f'BASE_{indice}')
                print(f"  {indice}. {alias} -> {ds['nombre']}")
            print("  o. Cambiar de origen")
            print("  q. Salir del programa")
            print("-" * 50)

            seleccion = input("\nSelecciona una opción: ").strip().lower()
            if seleccion == "q":
                salir = True
                break
            if seleccion == "o":
                break

            try:
                numero = int(seleccion)
                if 1 <= numero <= len(datasets):
                    procesar_dataset(datasets[numero - 1]["nombre"], datasets[numero - 1]["baseline"], origen)
            except ValueError:
                pass


if __name__ == "__main__":
    main()