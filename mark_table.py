
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.project_config import (
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    ANALYSIS_DIR,
    preparar_directorios,
)

BASE_DIR = PROJECT_ROOT
ANALYSIS_DIR = ANALYSIS_DIR

REGISTRO_PATH = (
    DATA_DIR / "Registro" / "registro_datasets.csv"
)

SALIDA_DIR = RESULTS_DIR / "Graficas_Generales"
      

# LOS 14 MÉTODOS VÁLIDOS

METODOS_VALIDOS = [

    "Random_Forest",
    "SVM",
    "MLP_Clasico",

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
    "DEEPINSIGHT"
]


# ============================================================
# NORMALIZAR TEXTO
# ============================================================

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

        texto = texto.replace(
            origen,
            destino
        )

    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto
    )

    texto = re.sub(
        r"_+",
        "_",
        texto
    )

    return texto.strip("_")


def normalizar_dataset(nombre):

    nombre = normalizar_texto(
        nombre
    )

    sufijos = [
        "_completo",
        "_original",
        "_orig",
        "_processed",
        "_preprocessed",
        "_train",
        "_test"
    ]

    for sufijo in sufijos:

        if nombre.endswith(sufijo):

            nombre = nombre[
                :-len(sufijo)
            ]

    return nombre.strip("_")


def normalizar_metodo(nombre):

    texto = normalizar_texto(
        nombre
    )

    equivalencias = {

        "random_forest":
            "Random_Forest",

        "randomforest":
            "Random_Forest",

        "svm":
            "SVM",

        "mlp":
            "MLP_Clasico",

        "mlp_clasico":
            "MLP_Clasico",

        "tinto":
            "TINTO",

        "igtd":
            "IGTD",

        "refined":
            "REFINED",

        "bargraph":
            "BARGRAPH",

        "bar_graph":
            "BARGRAPH",

        "distancematrix":
            "DISTANCEMATRIX",

        "distance_matrix":
            "DISTANCEMATRIX",

        "combination":
            "COMBINATION",

        "supertml":
            "SUPERTML",

        "featurewrap":
            "FEATUREWRAP",

        "bie":
            "BIE",

        "fotomics":
            "FOTOMICS",

        "deepinsight":
            "DEEPINSIGHT"
    }

    return equivalencias.get(
        texto,
        None
    )


# ============================================================
# BUSCAR COLUMNA
# ============================================================

def buscar_columna(
    df,
    posibles
):

    columnas = {}

    for columna in df.columns:

        clave = normalizar_texto(
            columna
        )

        columnas[clave] = columna

    for posible in posibles:

        clave = normalizar_texto(
            posible
        )

        if clave in columnas:

            return columnas[clave]

    return None


# ============================================================
# LEER CSV
# ============================================================

def leer_csv(ruta):

    try:

        return pd.read_csv(
            ruta,
            encoding="utf-8-sig",
            dtype=str
        )

    except UnicodeDecodeError:

        try:

            return pd.read_csv(
                ruta,
                encoding="latin-1",
                dtype=str
            )

        except Exception:

            return pd.DataFrame()

    except Exception:

        return pd.DataFrame()


# ============================================================
# CARGAR REGISTRO
# ============================================================

def cargar_registro():

    if not REGISTRO_PATH.exists():

        print(
            "\nERROR: No existe:"
        )

        print(
            REGISTRO_PATH
        )

        return pd.DataFrame()

    try:

        return pd.read_csv(
            REGISTRO_PATH,
            encoding="utf-8-sig",
            dtype=str
        )

    except UnicodeDecodeError:

        return pd.read_csv(
            REGISTRO_PATH,
            encoding="latin-1",
            dtype=str
        )


# ============================================================
# INFORMACIÓN DEL DATASET
# ============================================================

def obtener_info_dataset(
    df_registro,
    nombre_dataset
):

    if df_registro.empty:

        return None

    col_dataset = buscar_columna(
        df_registro,
        ["Dataset"]
    )

    if col_dataset is None:

        return None

    dataset_buscado = normalizar_dataset(
        nombre_dataset
    )

    for _, fila in df_registro.iterrows():

        dataset_registro = (
            normalizar_dataset(
                fila[col_dataset]
            )
        )

        if dataset_registro == dataset_buscado:

            return {

                "Instancias":
                    fila.get(
                        "Instancias",
                        "N/A"
                    ),

                "Features":
                    fila.get(
                        "Features",
                        "N/A"
                    ),

                "Clases":
                    fila.get(
                        "Num_Clases",
                        "N/A"
                    )
            }

    return None


# ============================================================
# BUSCAR LOS 3 CSV DE TABLAS
# ============================================================

def buscar_tablas_dataset(
    nombre_dataset
):

    # IMPORTANTE:
    #
    # El nombre cambia dinámicamente.
    #
    # Cryotherapy ->
    # Analysis/Cryotherapy_COMPLETO/Tablas
    #
    # AIDS_Clinical... ->
    # Analysis/AIDS_Clinical..._COMPLETO/Tablas
    #

    carpeta = (
        ANALYSIS_DIR
        / f"{nombre_dataset}_COMPLETO"
        / "Tablas"
    )

    if not carpeta.exists():

        print(
            f"  ERROR: No existe:"
        )

        print(
            f"  {carpeta}"
        )

        return []

    archivos = sorted(
        carpeta.glob("*.csv")
    )

    return archivos


# ============================================================
# PROCESAR TABLA DE COMPARATIVAS
# ============================================================

def procesar_tabla(
    ruta
):

    df = leer_csv(
        ruta
    )

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Buscar columnas
    # --------------------------------------------------------

    col_metodo = buscar_columna(
        df,
        [
            "Metodo",
            "Modelo",
            "Method",
            "Model"
        ]
    )

    col_f1 = buscar_columna(
        df,
        [
            "F1-Score (%)",
            "F1_Score",
            "F1"
        ]
    )

    col_accuracy = buscar_columna(
        df,
        [
            "Acc. (%)",
            "Accuracy (%)",
            "Accuracy",
            "Acc"
        ]
    )

    col_auc = buscar_columna(
        df,
        [
            "ROC-AUC (%)",
            "ROC_AUC",
            "ROC-AUC",
            "AUC"
        ]
    )

    if col_metodo is None:
        return pd.DataFrame()

    if col_accuracy is None:
        return pd.DataFrame()

    # --------------------------------------------------------
    # NORMALIZAR MÉTODO
    # --------------------------------------------------------

    df["Metodo_Normalizado"] = (
        df[col_metodo]
        .apply(normalizar_metodo)
    )

    # --------------------------------------------------------
    # SOLO LOS 14 MÉTODOS
    # --------------------------------------------------------

    df = df[
        df["Metodo_Normalizado"].isin(
            METODOS_VALIDOS
        )
    ].copy()

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # GUARDAMOS LOS VALORES ORIGINALES
    #
    # NO SE REDONDEAN
    # NO SE CONVIERTEN PARA LA SALIDA
    # --------------------------------------------------------

    df["Accuracy_Texto"] = (
        df[col_accuracy]
        .astype(str)
        .str.strip()
    )

    if col_f1 is not None:

        df["F1_Texto"] = (
            df[col_f1]
            .astype(str)
            .str.strip()
        )

    else:

        df["F1_Texto"] = "N/A"

    if col_auc is not None:

        df["AUC_Texto"] = (
            df[col_auc]
            .astype(str)
            .str.strip()
        )

    else:

        df["AUC_Texto"] = "N/A"

    # --------------------------------------------------------
    # SOLO PARA COMPARAR EL MÁXIMO
    #
    # Esto NO modifica el valor original.
    # --------------------------------------------------------

    df["Accuracy_Numerico"] = pd.to_numeric(
        df["Accuracy_Texto"],
        errors="coerce"
    )

    return df


# ============================================================
# OBTENER MEJOR RESULTADO ENTRE LOS 3 CSV
# ============================================================

def obtener_mejor_resultado(
    nombre_dataset
):

    archivos = buscar_tablas_dataset(
        nombre_dataset
    )

    if not archivos:

        return None

    resultados = []

    print(
        f"\n  CSV de Tablas encontrados:"
    )

    for archivo in archivos:

        print(
            f"    - {archivo.name}"
        )

        df = procesar_tabla(
            archivo
        )

        if not df.empty:

            resultados.append(
                df
            )

    if not resultados:

        return None

    # --------------------------------------------------------
    # UNIR LOS RESULTADOS DE LAS 3 REDES
    # --------------------------------------------------------

    df_total = pd.concat(
        resultados,
        ignore_index=True
    )

    # --------------------------------------------------------
    # ELIMINAR SOLO FILAS SIN ACCURACY
    # --------------------------------------------------------

    df_total = df_total[
        df_total["Accuracy_Numerico"]
        .notna()
    ].copy()

    if df_total.empty:

        return None

    # --------------------------------------------------------
    # TOMAR EL ACCURACY MÁS ALTO
    #
    # NO PROMEDIO
    # NO MEDIA
    # NO ROUND
    # --------------------------------------------------------

    indice_mejor = (
        df_total[
            "Accuracy_Numerico"
        ].idxmax()
    )

    mejor = (
        df_total
        .loc[indice_mejor]
        .copy()
    )

    return mejor


# ============================================================
# GENERAR RESUMEN GENERAL
# ============================================================

def generar_resumen():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESUMEN GENERAL DE MEJORES RESULTADOS"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # REGISTRO
    # --------------------------------------------------------

    df_registro = (
        cargar_registro()
    )

    if df_registro.empty:

        return

    # --------------------------------------------------------
    # Buscar carpetas *_COMPLETO
    # --------------------------------------------------------

    carpetas = sorted(

        [
            carpeta

            for carpeta
            in ANALYSIS_DIR.iterdir()

            if carpeta.is_dir()

            and carpeta.name.endswith(
                "_COMPLETO"
            )

            and (
                carpeta / "Tablas"
            ).exists()
        ],

        key=lambda x:
        x.name.lower()
    )

    print(
        f"\nDatasets encontrados: "
        f"{len(carpetas)}"
    )

    resultados_finales = []

    # ========================================================
    # PROCESAR DATASETS
    # ========================================================

    for carpeta in carpetas:

        nombre_dataset = (
            carpeta.name[
                :-len("_COMPLETO")
            ]
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Dataset: "
            f"{nombre_dataset}"
        )

        # ----------------------------------------------------
        # OBTENER MEJOR RESULTADO
        # ----------------------------------------------------

        mejor = (
            obtener_mejor_resultado(
                nombre_dataset
            )
        )

        if mejor is None:

            print(
                "  No se encontró "
                "resultado válido."
            )

            continue

        # ----------------------------------------------------
        # DATOS DEL REGISTRO
        # ----------------------------------------------------

        info = (
            obtener_info_dataset(
                df_registro,
                nombre_dataset
            )
        )

        if info is None:

            info = {

                "Instancias": "N/A",
                "Features": "N/A",
                "Clases": "N/A"
            }

            print(
                "  ADVERTENCIA: "
                "No encontrado en registro."
            )

        # ----------------------------------------------------
        # UNA SOLA FILA POR DATASET
        # ----------------------------------------------------

        fila = {

            "Dataset":
                nombre_dataset,

            "Mejor Modelo":
                mejor[
                    "Metodo_Normalizado"
                ],

            # EXACTAMENTE COMO VIENE DEL CSV
            "F1-Score (%)":
                mejor[
                    "F1_Texto"
                ],

            # EXACTAMENTE COMO VIENE DEL CSV
            "Accuracy (%)":
                mejor[
                    "Accuracy_Texto"
                ],

            # EXACTAMENTE COMO VIENE DEL CSV
            "ROC-AUC (%)":
                mejor[
                    "AUC_Texto"
                ],

            "Instancias":
                info[
                    "Instancias"
                ],

            "Features":
                info[
                    "Features"
                ],

            "Clases":
                info[
                    "Clases"
                ]
        }

        resultados_finales.append(
            fila
        )

        print(
            f"  Mejor método: "
            f"{fila['Mejor Modelo']}"
        )

        print(
            f"  Accuracy: "
            f"{fila['Accuracy (%)']}"
        )

        print(
            f"  F1: "
            f"{fila['F1-Score (%)']}"
        )

        print(
            f"  ROC-AUC: "
            f"{fila['ROC-AUC (%)']}"
        )

        print(
            f"  Instancias: "
            f"{fila['Instancias']}"
        )

        print(
            f"  Features: "
            f"{fila['Features']}"
        )

        print(
            f"  Clases: "
            f"{fila['Clases']}"
        )

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    if not resultados_finales:

        print(
            "\nNo se encontraron resultados."
        )

        return

    df_final = pd.DataFrame(
        resultados_finales
    )

    # ========================================================
    # SALIDA
    # ========================================================

    SALIDA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_csv = (
        SALIDA_DIR
        / "Resumen_Mejores_Modelos_Global.csv"
    )

    df_final.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # TABLA VISUAL
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(
            17,
            max(
                4,
                len(df_final) * 0.55 + 2
            )
        )
    )

    ax.axis(
        "off"
    )

    plt.title(
        "Resumen General de Mejores Resultados",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    tabla = ax.table(

        cellText=df_final.values,

        colLabels=df_final.columns,

        loc="center",

        cellLoc="center"
    )

    tabla.auto_set_font_size(
        False
    )

    tabla.set_fontsize(
        9
    )

    tabla.scale(
        1,
        1.5
    )

    # --------------------------------------------------------
    # ENCABEZADOS
    # --------------------------------------------------------

    for columna in range(
        len(df_final.columns)
    ):

        tabla[
            (0, columna)
        ].set_text_props(
            weight="bold"
        )

    # ========================================================
    # MEJOR ACCURACY GLOBAL
    # ========================================================

    accuracy_numerico = pd.to_numeric(
        df_final["Accuracy (%)"],
        errors="coerce"
    )

    if accuracy_numerico.notna().any():

        max_accuracy = (
            accuracy_numerico.max()
        )

        col = (
            df_final.columns
            .get_loc(
                "Accuracy (%)"
            )
        )

        for fila in range(
            len(df_final)
        ):

            if (
                accuracy_numerico.iloc[fila]
                == max_accuracy
            ):

                tabla[
                    (
                        fila + 1,
                        col
                    )
                ].set_text_props(
                    weight="bold"
                )

    # ========================================================
    # MEJOR F1 GLOBAL
    # ========================================================

    f1_numerico = pd.to_numeric(
        df_final["F1-Score (%)"],
        errors="coerce"
    )

    if f1_numerico.notna().any():

        max_f1 = (
            f1_numerico.max()
        )

        col = (
            df_final.columns
            .get_loc(
                "F1-Score (%)"
            )
        )

        for fila in range(
            len(df_final)
        ):

            if (
                f1_numerico.iloc[fila]
                == max_f1
            ):

                tabla[
                    (
                        fila + 1,
                        col
                    )
                ].set_text_props(
                    weight="bold"
                )

    # ========================================================
    # GUARDAR PNG
    # ========================================================

    ruta_png = (
        SALIDA_DIR
        / "Resumen_Mejores_Modelos_Global.png"
    )

    plt.savefig(

        ruta_png,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.3
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PROCESO TERMINADO"
    )

    print(
        "=" * 80
    )

    print(
        f"\nDatasets procesados: "
        f"{len(df_final)}"
    )

    print(
        f"\nCSV:"
        f"\n{ruta_csv}"
    )

    print(
        f"\nImagen:"
        f"\n{ruta_png}"
    )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    generar_resumen()