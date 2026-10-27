import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


def limpiar_dataset(df, columna_objetivo, politica_nulos='eliminar_filas'):
    """Limpieza global sin selección de características basada en el test.

    politica_nulos='eliminar_filas' reproduce la política original (para
    conservar los mismos índices si la rama de imágenes la utilizó).
    politica_nulos='imputar' conserva filas con features ausentes; la imputación
    se ajusta después y exclusivamente con el train de cada fold.
    """
    if columna_objetivo not in df.columns:
        raise ValueError(f"Target inexistente: {columna_objetivo}")
    if politica_nulos not in ('eliminar_filas', 'imputar'):
        raise ValueError('politica_nulos debe ser eliminar_filas o imputar')
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    inicial = len(df)
    if politica_nulos == 'eliminar_filas':
        df = df.dropna(axis=0)
    else:
        df = df.dropna(subset=[columna_objetivo])
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    if df.empty:
        raise ValueError('No quedan filas después de la limpieza')
    X = df.drop(columns=[columna_objetivo])
    registro = {
        'filas_con_nulos_eliminadas': inicial - len(df),
        'columnas_eliminadas': [],  # se determinan dentro de cada fold
        'columnas_categoricas': X.select_dtypes(include=['object', 'category', 'string']).columns.tolist(),
        'columnas_numericas': X.select_dtypes(include='number').columns.tolist(),
    }
    return df, registro


def preparar_features_fold(X_train, X_test, eliminar_constantes=True):
    """Selección opcional de constantes, imputación, one-hot y escalado; fits solo en train."""
    X_train, X_test = X_train.copy(), X_test.copy()
    constantes = X_train.columns[X_train.nunique(dropna=False) <= 1].tolist()
    if eliminar_constantes:
        X_train = X_train.drop(columns=constantes)
        X_test = X_test.drop(columns=constantes)
    if X_train.shape[1] == 0:
        raise ValueError('No quedan características después de quitar constantes')
    cats = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    nums = X_train.select_dtypes(include='number').columns.tolist()
    otros = set(X_train.columns) - set(cats) - set(nums)
    if otros:
        raise TypeError(f'Tipos de columnas no soportados: {sorted(otros)}')
    # Los valores ausentes se imputan sin consultar el fold de prueba.
    if nums:
        imp_num = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train[nums] = imp_num.fit_transform(X_train[nums])
        X_test[nums] = imp_num.transform(X_test[nums])
    if cats:
        # Valores faltantes en categorías se codifican como una categoría explícita.
        X_train[cats] = X_train[cats].astype('object').fillna('__MISSING__').astype(str)
        X_test[cats] = X_test[cats].astype('object').fillna('__MISSING__').astype(str)
    X_train, X_test, _, _ = aplicar_encoding_seguro(X_train, X_test, cats)
    X_train, X_test, _ = aplicar_escalado_seguro(X_train, X_test, nums)
    if not np.isfinite(X_train.to_numpy(dtype=float)).all() or not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError('Quedaron valores no finitos tras el preprocesamiento')
    return X_train, X_test, {'columnas_constantes': constantes, 'categoricas': cats, 'numericas': nums}


def aplicar_encoding_seguro(X_train, X_test, columnas_categoricas):
    if not columnas_categoricas:
        return X_train.copy(), X_test.copy(), False, []
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    a = encoder.fit_transform(X_train[columnas_categoricas])
    b = encoder.transform(X_test[columnas_categoricas])
    names = encoder.get_feature_names_out(columnas_categoricas).tolist()
    def unir(original, arr):
        return pd.concat([original.drop(columns=columnas_categoricas),
                          pd.DataFrame(arr, index=original.index, columns=names)], axis=1)
    return unir(X_train, a), unir(X_test, b), True, names


def aplicar_escalado_seguro(X_train, X_test, columnas_numericas):
    if not columnas_numericas:
        return X_train.copy(), X_test.copy(), False
    scaler = MinMaxScaler()
    a, b = X_train.copy(), X_test.copy()
    a[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
    b[columnas_numericas] = scaler.transform(X_test[columnas_numericas])
    return a, b, True


def codificar_target_seguro(y_train, y_test):
    encoder = LabelEncoder()
    a = encoder.fit_transform(y_train)
    b = encoder.transform(y_test)
    return (pd.Series(a, index=y_train.index, name=y_train.name),
            pd.Series(b, index=y_test.index, name=y_test.name),
            dict(enumerate(encoder.classes_)))


def asegurar_target_al_final(X, y):
    df = X.copy()
    df[y.name] = y.to_numpy()
    return df
