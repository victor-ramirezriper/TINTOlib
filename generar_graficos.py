from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACION Y DIRECTORIOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "Results"
BASE_ANALYSIS_DIR = BASE_DIR / "Analysis"

MODELOS_BASELINE = [
    "Random_Forest",
    "SVM",
    "MLP_Clasico"
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
    "FOTOMICS"
]

ARQUITECTURAS_ESPERADAS = [
    "ResNet18",
    "EfficientNet-B0",
    "ViT"
]

COLUMNAS_METRICAS = [
    "Accuracy",
    "F1_Score",
    "ROC_AUC"
]

COLUMNAS_TIEMPO = [
    "Tiempo_Transformacion_s",
    "Tiempo_Train_s",
    "Tiempo_Eval_s",
    "Tiempo_s"
]

COLUMNAS_RECURSOS = [
    "RAM_GB",
    "CPU_Usage_%",
    "VRAM_GB",
    "GPU_Usage_%"
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

    resultado = pd.DataFrame(index=df.index)
    resultado["Tipo"] = "Baseline"

    columna_modelo = buscar_columna(df, ["Modelo", "model", "Algoritmo"])
    resultado["Modelo"] = df[columna_modelo].astype(str).str.strip() if columna_modelo else ""
    
    resultado["Arquitectura"] = "Tabular"
    resultado["Metodo_TINTO"] = "BASELINE"
    resultado["Fold"] = obtener_columna_numerica(df, ["Fold_Actual", "Fold", "fold"])
    
    # Normalización automática de escala (si está en 0-1, pasa a porcentaje 0-100)
    for metrica in COLUMNAS_METRICAS:
        serie = obtener_columna_numerica(df, [metrica])
        if serie.notna().any() and serie.max(skipna=True) <= 1.0:
            serie = serie * 100.0
        resultado[metrica] = serie

    resultado["Tiempo_Transformacion_s"] = obtener_columna_numerica(df, ["Tiempo_Transformacion_s", "Tiempo_Transformacion", "Transform_Time_s"])
    resultado["Tiempo_Train_s"] = obtener_columna_numerica(df, ["Tiempo_Entrenamiento_s", "Tiempo_Train_s", "Training_Time_s"])
    resultado["Tiempo_Eval_s"] = obtener_columna_numerica(df, ["Tiempo_Evaluacion_s", "Tiempo_Eval_s", "Evaluation_Time_s"])
    resultado["Tiempo_s"] = obtener_columna_numerica(df, ["Tiempo_s", "Tiempo_Prom_Fold_s", "Tiempo_Total_s"])

    resultado["RAM_GB"] = obtener_columna_numerica(df, ["RAM_GB", "Max_RAM_GB"])
    resultado["CPU_Usage_%"] = obtener_columna_numerica(df, ["CPU_Usage_%", "CPU_Usage", "Avg_CPU_Percent"])
    resultado["VRAM_GB"] = obtener_columna_numerica(df, ["GPU_RAM_GB", "VRAM_GB", "Max_VRAM_GB"])
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

    for metrica in COLUMNAS_METRICAS:
        resultado[metrica] = obtener_columna_numerica(df, [metrica])

    resultado["Tiempo_Transformacion_s"] = obtener_columna_numerica(df, ["Tiempo_Transformacion_s", "Tiempo_Transformacion", "Transform_Time_s"])
    resultado["Tiempo_Train_s"] = obtener_columna_numerica(df, ["Tiempo_Entrenamiento_s", "Tiempo_Train_s", "Training_Time_s"])
    resultado["Tiempo_Eval_s"] = obtener_columna_numerica(df, ["Tiempo_Evaluacion_s", "Tiempo_Eval_s", "Evaluation_Time_s"])
    resultado["RAM_GB"] = obtener_columna_numerica(df, ["Max_RAM_GB", "RAM_GB"])
    resultado["CPU_Usage_%"] = obtener_columna_numerica(df, ["Avg_CPU_Percent", "CPU_Usage_%", "CPU_Usage"])
    resultado["VRAM_GB"] = obtener_columna_numerica(df, ["Max_VRAM_GB", "VRAM_GB", "GPU_RAM_GB"])
    resultado["GPU_Usage_%"] = obtener_columna_numerica(df, ["GPU_Usage_%", "GPU_Usage"])

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


def cargar_deep_learning(dataset_seleccionado):
    experimentos = buscar_experimentos_deep_learning(dataset_seleccionado)
    if not experimentos:
        return pd.DataFrame(), pd.DataFrame()

    folds, historiales = [], []

    for experimento in experimentos:
        df_folds = cargar_resultados_folds(experimento)
        if not df_folds.empty:
            folds.append(df_folds)

        historial = cargar_historial_entrenamiento(experimento)
        if not historial.empty:
            historiales.append(historial)

    columnas_base = [
        "Tipo", "Arquitectura", "Metodo_TINTO", "Modelo", "Fold",
        "Accuracy", "F1_Score", "ROC_AUC", "Tiempo_Transformacion_s", "Tiempo_Train_s",
        "Tiempo_Eval_s", "Tiempo_s", "RAM_GB", "CPU_Usage_%",
        "VRAM_GB", "GPU_Usage_%", "Fuente"
    ]

    df_folds = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(columns=columnas_base)
    df_historial = pd.concat(historiales, ignore_index=True) if historiales else pd.DataFrame()

    return df_folds, df_historial


def generar_resumen_desde_folds(df_folds):
    if df_folds is None or df_folds.empty:
        return pd.DataFrame()

    columnas_a_promediar = [
        "Accuracy", "F1_Score", "ROC_AUC", "Tiempo_Transformacion_s", "Tiempo_Train_s",
        "Tiempo_Eval_s", "Tiempo_s", "RAM_GB", "CPU_Usage_%",
        "VRAM_GB", "GPU_Usage_%"
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
# ESTRUCTURA DE 13 MODELOS Y ESCALA UNIFICADA
# ==========================================

def crear_estructura_13_modelos(df_base, df_dl, arquitectura):
    filas = []

    # 1. BASELINE (3 modelos tradicionales)
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

    # 2. DEEP LEARNING (10 métodos TINTOlib)
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
# FUNCIONES DE GRAFICADO CON ETIQUETAS EXPLÍCITAS
# ==========================================

def generar_grafica_barras(df, columna, titulo, nombre_archivo, output_dir, y_label, porcentaje=False):
    if df.empty or columna not in df.columns:
        return

    datos = df[df[columna].notna()].copy()
    if datos.empty:
        return

    x = np.arange(len(datos))
    valores = datos[columna].astype(float).values

    if porcentaje and np.nanmax(np.abs(valores)) <= 1.0:
        valores = valores * 100.0

    fig, ax = plt.subplots(figsize=(15, 8))
    barras = ax.bar(x, valores)

    ax.set_title(titulo, fontsize=15, fontweight="bold")
    ax.set_xlabel("Modelos / Métodos Evaluados (Baseline y TINTOlib)", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datos["Modelo"], rotation=45, ha="right", fontsize=10)

    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            f"{valor:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    ruta = output_dir / nombre_archivo
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafica: {ruta}")


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
    ax.set_xlabel("Métodos TINTOlib (Variabilidad entre Folds)", fontsize=12, fontweight="bold")
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
    ax.set_xlabel("Métodos TINTOlib", fontsize=12, fontweight="bold")
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
    ax.set_xlabel("Métricas y Recursos Evaluados", fontsize=12, fontweight="bold")
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

            ax.set_title(f"Curvas de pérdida - {arquitectura} - {metodo}", fontsize=15, fontweight="bold")
            ax.set_xlabel("Épocas (Epoch)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Pérdida (Loss)", fontsize=12, fontweight="bold")
            ax.legend()
            plt.tight_layout()

            nombre = f"Loss_{limpiar_nombre_archivo(arquitectura)}_{limpiar_nombre_archivo(metodo)}.png"
            plt.savefig(carpeta_loss / nombre, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  -> Grafica Loss: {carpeta_loss / nombre}")


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
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=2, label="p = 0.05 (Significativo)")

    ax.set_title(f"Prueba Kruskal-Wallis por métrica - {arquitectura}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Métrica Evaluada", fontsize=12, fontweight="bold")
    ax.set_ylabel("-log10(p-value)", fontsize=12, fontweight="bold")
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

def generar_graficas_comparativas(df_base, df_resumen, df_folds, arquitectura, output_dir):
    print(f"\n" + "-" * 70)
    print(f"Arquitectura: {arquitectura}")

    df_13 = crear_estructura_13_modelos(df_base, df_resumen, arquitectura)
    df_13 = normalizar_metricas_porcentaje(df_13)
    print(f"Modelos incluidos: {len(df_13)}\n")

    prefijo = limpiar_nombre_archivo(arquitectura)

    # 1. Métricas con escala unificada (%)
    for col, tit, nom in [
        ("Accuracy_pct", f"Comparativa de Accuracy (%) - {arquitectura}", f"Comparativa_{prefijo}_Accuracy.png"),
        ("F1_Score_pct", f"Comparativa de F1-Score Macro (%) - {arquitectura}", f"Comparativa_{prefijo}_F1_Score.png"),
        ("ROC_AUC_pct", f"Comparativa de ROC-AUC (%) - {arquitectura}", f"Comparativa_{prefijo}_ROC_AUC.png")
    ]:
        generar_grafica_barras(df_13, col, tit, nom, output_dir, y_label="Porcentaje de Rendimiento (%)", porcentaje=False)

    # 2. Tiempos (incluyendo Transformación)
    for col, tit, nom in [
        ("Tiempo_s", f"Tiempo total de procesamiento (s) - {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Total.png"),
        ("Tiempo_Transformacion_s", f"Tiempo de transformación TINTOlib (s) - {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Transformacion.png"),
        ("Tiempo_Train_s", f"Tiempo de entrenamiento (s) - {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Train.png"),
        ("Tiempo_Eval_s", f"Tiempo de evaluación (s) - {arquitectura}", f"Comparativa_{prefijo}_Tiempo_Eval.png")
    ]:
        generar_grafica_barras(df_13, col, tit, nom, output_dir, y_label="Tiempo (segundos)")

    # 3. Recursos
    for col, tit, nom in [
        ("RAM_GB", f"Uso de Memoria RAM (GB) - {arquitectura}", f"Comparativa_{prefijo}_RAM.png"),
        ("CPU_Usage_%", f"Uso promedio de CPU (%) - {arquitectura}", f"Comparativa_{prefijo}_CPU.png"),
        ("VRAM_GB", f"Uso de Memoria VRAM GPU (GB) - {arquitectura}", f"Comparativa_{prefijo}_VRAM.png"),
        ("GPU_Usage_%", f"Uso promedio de GPU (%) - {arquitectura}", f"Comparativa_{prefijo}_GPU.png")
    ]:
        generar_grafica_barras(df_13, col, tit, nom, output_dir, y_label="Consumo de Recursos")

    # 4. Scatters
    generar_scatter(df_13, "Tiempo_s", "Accuracy_pct", f"Accuracy (%) vs Tiempo Total (s) - {arquitectura}", f"Scatter_{prefijo}_Accuracy_vs_Tiempo.png", output_dir, "Tiempo total de procesamiento (s)", "Accuracy (%)")
    generar_scatter(df_13, "Tiempo_Transformacion_s", "Accuracy_pct", f"Accuracy (%) vs Tiempo de Transformación (s) - {arquitectura}", f"Scatter_{prefijo}_Accuracy_vs_Transformacion.png", output_dir, "Tiempo de transformación TINTOlib (s)", "Accuracy (%)")
    generar_scatter(df_13, "RAM_GB", "Accuracy_pct", f"Accuracy (%) vs RAM (GB) - {arquitectura}", f"Scatter_{prefijo}_Accuracy_vs_RAM.png", output_dir, "Memoria RAM (GB)", "Accuracy (%)")

    # 5. Heatmap
    generar_heatmap(
        df_13,
        ["Accuracy_pct", "F1_Score_pct", "ROC_AUC_pct", "Tiempo_Transformacion_s", "Tiempo_s", "RAM_GB", "CPU_Usage_%", "VRAM_GB"],
        f"Mapa comparativo estandarizado de métricas y recursos - {arquitectura}",
        f"Heatmap_{prefijo}.png",
        output_dir,
        estandarizar=True
    )

    return df_13


def generar_analisis_folds(df_folds, output_dir):
    if df_folds.empty:
        return

    for arquitectura in sorted(df_folds["Arquitectura"].dropna().unique()):
        prefijo = limpiar_nombre_archivo(arquitectura)
        for col, tit, nom, ylbl in [
            ("Accuracy", f"Variabilidad de Accuracy por Fold - {arquitectura}", f"Boxplot_{prefijo}_Accuracy_Folds.png", "Accuracy (%)"),
            ("F1_Score", f"Variabilidad de F1-Score por Fold - {arquitectura}", f"Boxplot_{prefijo}_F1_Folds.png", "F1-Score Macro (%)"),
            ("ROC_AUC", f"Variabilidad de ROC-AUC por Fold - {arquitectura}", f"Boxplot_{prefijo}_ROC_AUC_Folds.png", "ROC-AUC (%)"),
            ("Tiempo_Transformacion_s", f"Variabilidad de Tiempo de Transformación por Fold - {arquitectura}", f"Boxplot_{prefijo}_Tiempo_Transformacion_Folds.png", "Tiempo de Transformación (s)")
        ]:
            # Normalizar folds de accuracy a porcentaje si están en 0-1
            datos_temp = df_folds.copy()
            if col in COLUMNAS_METRICAS and datos_temp[col].max(skipna=True) <= 1.0:
                datos_temp[col] = datos_temp[col] * 100.0
                ylbl = f"{col} (%)"

            generar_boxplot_folds(datos_temp, arquitectura, col, tit, nom, output_dir, y_label=ylbl)
            generar_grafica_ic95(datos_temp, arquitectura, col, f"Media e IC95 - {col} - {arquitectura}", f"IC95_{prefijo}_{col}.png", output_dir, y_label=ylbl)


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
        ax.set_title(f"Comparación entre arquitecturas de Deep Learning - {columna}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Métodos TINTOlib", fontsize=12, fontweight="bold")
        ax.set_ylabel(columna, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        ruta = output_dir / f"Comparacion_Arquitecturas_{limpiar_nombre_archivo(columna)}.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Grafica arquitectura: {ruta}")


# ==========================================
# MAIN PRINCIPAL
# ==========================================

def main():
    print("\n" + "=" * 70)
    print("        ANALISIS VISUAL Y ESTADISTICO TINTOlib")
    print("=" * 70)

    if not RESULTS_DIR.exists():
        print(f"\nNo existe la carpeta Results:\n  {RESULTS_DIR}")
        return

    baselines = buscar_baselines()
    if not baselines:
        print("\nNo se encontraron archivos Baseline_*.csv")
        return

    print("\nDatasets disponibles:")
    datasets = []
    for indice, baseline in enumerate(baselines, start=1):
        dataset = extraer_dataset_desde_baseline(baseline)
        datasets.append({"nombre": dataset, "baseline": baseline})
        print(f"  {indice}. {dataset}")

    while True:
        try:
            seleccion = input("\nSelecciona el numero del dataset a graficar: ").strip()
            numero = int(seleccion)
            if 1 <= numero <= len(datasets):
                break
            print("Seleccion invalida.")
        except ValueError:
            print("Introduce un numero valido.")

    dataset_seleccionado = datasets[numero - 1]["nombre"]
    baseline_path = datasets[numero - 1]["baseline"]

    print(f"\nDataset seleccionado: {dataset_seleccionado}")

    # ==========================================
    # CREAR ESTRUCTURA DE SUBCARPETAS ORGANIZADAS
    # ==========================================
    nombre_muestra_limpio = limpiar_nombre_archivo(dataset_seleccionado)
    dir_dataset = BASE_ANALYSIS_DIR / nombre_muestra_limpio

    dir_comparativas = asegurar_directorio(dir_dataset / "Comparativas")
    dir_folds = asegurar_directorio(dir_dataset / "Folds")
    dir_arquitecturas = asegurar_directorio(dir_dataset / "Arquitecturas")
    dir_loss = asegurar_directorio(dir_dataset / "Loss")
    dir_estadisticas = asegurar_directorio(dir_dataset / "Estadisticas")
    dir_csvs = asegurar_directorio(dir_dataset / "CSVs")

    # CARGAR DATOS
    df_base = cargar_baseline(baseline_path)
    df_folds, df_historial = cargar_deep_learning(dataset_seleccionado)

    # RECONSTRUIR EL RESUMEN EXCLUSIVAMENTE DESDE LOS FOLDS
    df_resumen = generar_resumen_desde_folds(df_folds)

    # NORMALIZAR COLUMNAS NUMÉRICAS
    for df_target in [df_resumen, df_folds]:
        if not df_target.empty:
            for col in (COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS):
                if col in df_target.columns:
                    df_target[col] = convertir_numerico(df_target[col])

    # DETECTAR ARQUITECTURAS
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

    # GENERAR COMPARATIVAS Y GUARDAR EN SUBCARPETAS
    for arquitectura in arquitecturas:
        generar_graficas_comparativas(df_base, df_resumen, df_folds, arquitectura, dir_comparativas)

    if not df_folds.empty:
        generar_analisis_folds(df_folds, dir_folds)

    if not df_resumen.empty:
        generar_analisis_arquitecturas(df_resumen, dir_arquitecturas)

    if not df_historial.empty:
        df_historial.to_csv(dir_csvs / "Historial_Loss_Unificado.csv", index=False, encoding="utf-8-sig")
        generar_graficas_historial(df_historial, dir_loss)

    # ESTADISTICAS Y PRUEBAS (CON ANÁLISIS DE VARIABILIDAD ENTRE FOLDS)
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

    # CORRELACION GLOBAL
    if not df_resumen.empty:
        corr = generar_correlacion(
            df_resumen,
            COLUMNAS_METRICAS + COLUMNAS_TIEMPO + COLUMNAS_RECURSOS,
            "Correlación global entre métricas, tiempos de transformación/entrenamiento y recursos",
            "Correlacion_Global.png",
            dir_estadisticas
        )
        if corr is not None:
            corr.to_csv(dir_csvs / "Matriz_Correlacion.csv", encoding="utf-8-sig")

    # CSVs UNIFICADOS POR MUESTRA
    if not df_folds.empty:
        df_folds.to_csv(dir_csvs / "Resultados_Folds_Unificados.csv", index=False, encoding="utf-8-sig")

    if not df_resumen.empty:
        df_resumen.to_csv(dir_csvs / "Resultados_Resumen_Unificados.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("ANALISIS VISUAL COMPLETADO")
    print("=" * 70)
    print(f"Dataset: {dataset_seleccionado}")
    print(f"Directorio estructurado: {dir_dataset}")
    print("=" * 70)


if __name__ == "__main__":
    main()