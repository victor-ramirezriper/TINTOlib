import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

def limpiar_dataset(df, columna_objetivo):
    """
    Paso 1: Limpieza dinámica del dataset antes del split.
    Soporta cualquier cantidad de características (features).
    Elimina infinitos, nulos, convierte booleanos a 0/1 y elimina columnas constantes.
    """
    df = df.copy()

    registro = {
        "filas_con_nulos_eliminadas": 0,
        "columnas_eliminadas": [],
        "columnas_categoricas": [],
        "columnas_numericas": []
    }

    if columna_objetivo not in df.columns:
        raise ValueError(
            f"La columna objetivo '{columna_objetivo}' no existe en el dataset."
        )

    # REEMPLAZAR INFINITOS POR NULOS PARA SU ELIMINACIÓN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ELIMINAR NULOS (Obligatorio para algoritmos matemáticos posteriores)
    nulos_iniciales = df.isnull().any(axis=1).sum()
    if nulos_iniciales > 0:
        df = df.dropna()
        registro["filas_con_nulos_eliminadas"] = nulos_iniciales

    # OPTIMIZACIÓN DE ESPACIO: Convertir booleanos directamente a 0 y 1
    columnas_bool = df.select_dtypes(include=['bool']).columns
    for col in columnas_bool:
        df[col] = df[col].astype(int)

    y = df[columna_objetivo]
    x = df.drop(columns=[columna_objetivo])

    # ELIMINAR COLUMNAS CONSTANTES (Previene errores matemáticos y reduce dimensionalidad muerta)
    columnas_constantes = [
        columna for columna in x.columns
        if x[columna].nunique(dropna=False) <= 1
    ]

    if columnas_constantes:
        x = x.drop(columns=columnas_constantes)
        registro["columnas_eliminadas"] = columnas_constantes

    registro["columnas_numericas"] = x.select_dtypes(include=["number"]).columns.tolist()
    registro["columnas_categoricas"] = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Reconstruimos temporalmente para devolver el DF limpio
    df_limpio = x.copy()
    df_limpio[columna_objetivo] = y.values

    return df_limpio, registro


def aplicar_encoding_seguro(X_train, X_test, columnas_categoricas):
    """
    Paso 2: Aplica One-Hot Encoding ajustando SOLO con X_train para evitar data leakage.
    Se expande dinámicamente según la cantidad de categorías encontradas.
    """
    if not columnas_categoricas:
        return X_train, X_test, False, []

    encoder = OneHotEncoder(
        handle_unknown="ignore", 
        sparse_output=False
    )

    # Fit y transform solo en entrenamiento
    train_cat_encoded = encoder.fit_transform(X_train[columnas_categoricas])
    # Solo transform en prueba
    test_cat_encoded = encoder.transform(X_test[columnas_categoricas])

    nombres_nuevos = encoder.get_feature_names_out(columnas_categoricas)

    # Reconstruir X_train
    x_train_cat = pd.DataFrame(train_cat_encoded, columns=nombres_nuevos, index=X_train.index)
    X_train = X_train.drop(columns=columnas_categoricas)
    X_train = pd.concat([X_train, x_train_cat], axis=1)

    # Reconstruir X_test
    x_test_cat = pd.DataFrame(test_cat_encoded, columns=nombres_nuevos, index=X_test.index)
    X_test = X_test.drop(columns=columnas_categoricas)
    X_test = pd.concat([X_test, x_test_cat], axis=1)

    return X_train, X_test, True, nombres_nuevos


def aplicar_escalado_seguro(X_train, X_test, columnas_numericas):
    """
    Paso 3: Aplica MinMaxScaler ajustando SOLO con X_train.
    Normaliza dimensiones N a un rango [0,1] óptimo para métodos espaciales.
    """
    if not columnas_numericas:
        return X_train, X_test, False

    scaler = MinMaxScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
    X_test_scaled[columnas_numericas] = scaler.transform(X_test[columnas_numericas])
    
    return X_train_scaled, X_test_scaled, True


def codificar_target_seguro(y_train, y_test):
    """
    Paso 4: Codifica la variable objetivo a enteros continuos (0, 1, 2...).
    Requisito estricto para las funciones de pérdida en PyTorch/TensorFlow.
    """
    encoder = LabelEncoder()
    
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    y_train_series = pd.Series(y_train_encoded, index=y_train.index, name=y_train.name)
    y_test_series = pd.Series(y_test_encoded, index=y_test.index, name=y_test.name)
    
    # Mapa para decodificar predicciones al final del proyecto
    mapa_clases = dict(zip(range(len(encoder.classes_)), encoder.classes_))
    
    return y_train_series, y_test_series, mapa_clases


def asegurar_target_al_final(X, y):
    """
    Paso 5: Ensamblaje final de características dinámicas + target.
    """
    df_procesado = X.copy()
    df_procesado[y.name] = y.values
    
    if df_procesado.columns[-1] != y.name:
        raise ValueError("Error en la estructura: La variable objetivo no quedó como última columna.")
        
    return df_procesado