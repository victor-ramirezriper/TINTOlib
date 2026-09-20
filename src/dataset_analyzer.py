import os
import pandas as pd
import warnings

def cargar_dataset(ruta):
    """
    Carga y realiza validaciones iniciales de estructura.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ruta}"
        )
    
    extension = os.path.splitext(ruta)[1].lower()

    if extension != ".csv":
        raise ValueError(
            "El archivo seleccionado no es válido. "
            "Solo se permiten archivos .csv"
        )

    # LEER EL DOCUMENTO
    try:
        df = pd.read_csv(ruta)
    except Exception as e:
        raise ValueError(
            f"No fue posible leer el archivo CSV.\n"
            f"Detalle: {e}"
        ) from e

    if df.empty:
        raise ValueError(
            "El archivo CSV está vacío."
        )

    # NOMBRE DE COLUMNAS
    columnas_vacias = [
        columna
        for columna in df.columns
        if str(columna).strip() == ""
    ]

    if columnas_vacias:
        raise ValueError(
            "El dataset contiene columnas sin nombre."
        )

    columnas_duplicadas = df.columns[
        df.columns.duplicated()
    ].tolist()

    if columnas_duplicadas:
        raise ValueError(
            f"El dataset contiene columnas duplicadas: "
            f"{columnas_duplicadas}"
        )

    # REVISIÓN DE NULOS
    nulos = df.isnull().sum()
    columnas_con_nulos = nulos[nulos > 0]

    columnas_texto = df.select_dtypes(
        include=["object", "string"]
    ).columns

    vacios = {}
    for columna in columnas_texto:
        cantidad_vacios = (
            df[columna]
            .astype("string")
            .str.strip()
            .eq("")
            .sum()
        )
        if cantidad_vacios > 0:
            vacios[columna] = cantidad_vacios

    if not columnas_con_nulos.empty or vacios:
        mensaje = "El dataset contiene campos nulos o vacíos.\n"
        if not columnas_con_nulos.empty:
            mensaje += "Valores nulos detectados que serán limpiados en el preprocesamiento.\n"
        warnings.warn(mensaje)

    return df


def analizar_dataset(df, nombre_dataset="Dataset_Desconocido", columna_target=None):
    """
    Analiza las características principales de un dataset previamente validado,
    incluyendo clases, características, tipo de problema y nivel de desbalanceo.
    """
    informacion = {}
    
    informacion["nombre_dataset"] = nombre_dataset

    # Dimensiones Base
    informacion["muestras"] = len(df)
    informacion["columnas_totales"] = len(df.columns)

    # Tipos de datos
    informacion["columnas_numericas"] = (
        df.select_dtypes(include=["number"]).columns.tolist()
    )
    informacion["columnas_categoricas"] = (
        df.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()
    )

    # Columnas constantes
    informacion["columnas_constantes"] = [
        columna
        for columna in df.columns
        if df[columna].nunique(dropna=False) <= 1
    ]

    informacion["nombres_columnas"] = df.columns.tolist()
    
    # ---------------------------------------------------------
    # ANÁLISIS ESPECÍFICO PARA TINTOLIB (Clases y Desbalanceo)
    # ---------------------------------------------------------
    
    # Si no se especifica la columna target, asumimos que es la última
    target_col = columna_target if columna_target else df.columns[-1]
    
    if target_col in df.columns:
        informacion["caracteristicas"] = len(df.columns) - 1
        informacion["nombre_target"] = target_col
        
        # Calcular Clases
        conteo_clases = df[target_col].value_counts()
        informacion["numero_clases"] = len(conteo_clases)
        informacion["distribucion_clases"] = conteo_clases.to_dict()
        
        # Identificar el tipo de problema
        if informacion["numero_clases"] == 2:
            informacion["tipo_problema"] = "Clasificación Binaria"
        elif 2 < informacion["numero_clases"] <= 20: 
            # Límite de 20 sugerido para datasets típicos de UCI
            informacion["tipo_problema"] = "Clasificación Multiclase"
        else:
            informacion["tipo_problema"] = "Multiclase Extrema o Regresión"
            warnings.warn("Se detectó un número inusualmente alto de clases. Verifica si el problema es de regresión.")

        # Análisis de Desbalanceo adaptado a Multiclase
        clase_mayoritaria = conteo_clases.max()
        clase_minoritaria = conteo_clases.min()
        
        if clase_minoritaria > 0:
            # En multiclase, comparamos la más grande con la más pequeña como métrica de riesgo máximo
            ratio_desbalanceo = round(clase_mayoritaria / clase_minoritaria, 2)
            informacion["ratio_riesgo_desbalanceo"] = ratio_desbalanceo
            
            # Criterio: Si la clase mayoritaria es más del doble de la minoritaria
            informacion["requiere_balanceo"] = bool(ratio_desbalanceo > 2.0)
        else:
            informacion["ratio_riesgo_desbalanceo"] = float('inf')
            informacion["requiere_balanceo"] = True
            
    else:
         informacion["caracteristicas"] = len(df.columns)
         informacion["tipo_problema"] = "Desconocido"
         informacion["numero_clases"] = 0
         informacion["requiere_balanceo"] = "N/A"
         warnings.warn(f"La columna target '{target_col}' no se encontró en el dataset.")

    return informacion