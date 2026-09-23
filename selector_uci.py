import os
import re
from datetime import datetime

import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo

except ImportError:
    print("ERROR: no está instalado ucimlrepo.")
    print("\nInstálalo con:")
    print("pip install ucimlrepo")
    raise SystemExit


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_DIR = r"D:\EXP_TINTO\Data"

REGISTRO_DIR = os.path.join(
    DATA_DIR,
    "Registro"
)

REGISTRO_FILE = os.path.join(
    REGISTRO_DIR,
    "registro_datasets.csv"
)


# ============================================================
# CRITERIOS EXPERIMENTALES
# ============================================================

# ------------------------------------------------------------
# PEQUEÑO
# ------------------------------------------------------------

SMALL_MAX_INSTANCES = 1000
SMALL_MIN_FEATURES = 10
SMALL_MAX_FEATURES = 20


# ------------------------------------------------------------
# MEDIANO
# ------------------------------------------------------------

MEDIUM_MIN_INSTANCES = 1001
MEDIUM_MIN_FEATURES = 30
MEDIUM_MAX_FEATURES = 50


# ------------------------------------------------------------
# CLASIFICACIÓN
# ------------------------------------------------------------

MIN_CLASSES = 2
MIN_SAMPLES_PER_CLASS = 10


# ============================================================
# UTILIDADES
# ============================================================

def limpiar_nombre_archivo(nombre):
    """
    Convierte un nombre de dataset en un nombre válido
    para Windows.
    """

    nombre = str(nombre).strip()

    # Caracteres no permitidos en nombres de Windows
    nombre = re.sub(
        r'[<>:"/\\|?*]',
        '_',
        nombre
    )

    # Reemplazar espacios múltiples por _
    nombre = re.sub(
        r'\s+',
        '_',
        nombre
    )

    return nombre[:150]


def obtener_texto_metadata(valor):
    """
    Convierte la metadata a texto de forma segura.
    """

    if valor is None:
        return ""

    if isinstance(valor, (list, tuple, set)):
        return " ".join(
            map(str, valor)
        )

    return str(valor)


def detectar_marcadores_faltantes(df):
    """
    Busca marcadores comunes utilizados para representar
    valores faltantes dentro de columnas de texto.
    """

    marcadores = [
        "?",
        "NA",
        "N/A",
        "NULL",
        "null",
        "",
        "unknown",
        "Unknown"
    ]

    encontrados = {}

    for marcador in marcadores:

        cantidad_total = 0

        for columna in df.columns:

            if df[columna].dtype == "object":

                cantidad = (
                    df[columna]
                    .astype(str)
                    .str.strip()
                    .eq(marcador)
                    .sum()
                )

                cantidad_total += int(
                    cantidad
                )

        if cantidad_total > 0:

            encontrados[marcador] = (
                cantidad_total
            )

    return encontrados


def determinar_categoria(
    n_instancias,
    n_features
):
    """
    Determina si el dataset pertenece a PEQUEÑO,
    MEDIANO o queda fuera de los criterios.
    """

    # --------------------------------------------------------
    # PEQUEÑO
    # --------------------------------------------------------

    if (
        n_instancias <= SMALL_MAX_INSTANCES
        and
        SMALL_MIN_FEATURES
        <= n_features
        <= SMALL_MAX_FEATURES
    ):

        return "PEQUEÑO"


    # --------------------------------------------------------
    # MEDIANO
    # --------------------------------------------------------

    if (
        n_instancias >= MEDIUM_MIN_INSTANCES
        and
        MEDIUM_MIN_FEATURES
        <= n_features
        <= MEDIUM_MAX_FEATURES
    ):

        return "MEDIANO"


    # --------------------------------------------------------
    # FUERA DE CRITERIO
    # --------------------------------------------------------

    return None


def obtener_target_desde_uci(
    dataset,
    y
):
    """
    Identifica el target utilizando la información de UCI.

    Se requiere exactamente una variable objetivo.

    Retorna:
        target_name
        target
        target_unico
    """

    # --------------------------------------------------------
    # Intentar identificar el target mediante variables
    # --------------------------------------------------------

    try:

        variables = dataset.variables

        if (
            variables is not None
            and not variables.empty
            and "role" in variables.columns
            and "name" in variables.columns
        ):

            roles = (
                variables["role"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            targets_uci = variables[
                roles == "target"
            ]

            # Un único target
            if len(targets_uci) == 1:

                target_name = str(
                    targets_uci.iloc[0]["name"]
                )

                # y puede ser DataFrame
                if isinstance(
                    y,
                    pd.DataFrame
                ):

                    if target_name in y.columns:

                        target = (
                            y[target_name]
                            .copy()
                        )

                        return (
                            target_name,
                            target,
                            True
                        )

                # y puede ser Series
                if isinstance(
                    y,
                    pd.Series
                ):

                    target = y.copy()

                    if target.name is None:
                        target.name = target_name

                    return (
                        target_name,
                        target,
                        True
                    )

            # Múltiples targets
            elif len(targets_uci) > 1:

                nombres = ", ".join(
                    map(
                        str,
                        targets_uci["name"].tolist()
                    )
                )

                return (
                    nombres,
                    None,
                    False
                )

    except Exception:
        pass


    # --------------------------------------------------------
    # RESPALDO: utilizar y directamente
    # --------------------------------------------------------

    if isinstance(
        y,
        pd.Series
    ):

        target_name = (
            str(y.name)
            if y.name is not None
            else "target"
        )

        return (
            target_name,
            y.copy(),
            True
        )


    if isinstance(
        y,
        pd.DataFrame
    ):

        # Un único target
        if y.shape[1] == 1:

            target_name = str(
                y.columns[0]
            )

            return (
                target_name,
                y.iloc[:, 0].copy(),
                True
            )

        # Múltiples targets
        nombres = ", ".join(
            map(
                str,
                y.columns
            )
        )

        return (
            nombres,
            None,
            False
        )


    return (
        "target",
        None,
        False
    )


def detectar_tarea_clasificacion(dataset, target=None):
    """
    Determina si UCI identifica el dataset como
    problema de clasificación, revisando 'associated_tasks',
    'task' en la metadata y utilizando el target como respaldo.
    """
    try:
        metadata = dataset.metadata
    except Exception:
        metadata = None

    task_textos = []

    if metadata is not None:
        # Revisar campos comunes en ucimlrepo donde se especifica la tarea
        candidatos_campos = ["associated_tasks", "task", "purpose"]
        for campo in candidatos_campos:
            valor = None
            try:
                if hasattr(metadata, campo):
                    valor = getattr(metadata, campo)
                elif isinstance(metadata, dict):
                    valor = metadata.get(campo, None)
                elif hasattr(metadata, "get"):
                    valor = metadata.get(campo, None)
            except Exception:
                pass
            
            if valor is not None:
                task_textos.append(obtener_texto_metadata(valor).lower())

    texto_completo = " ".join(task_textos)

    # Validar en inglés o español
    if "classification" in texto_completo or "clasificación" in texto_completo:
        return True

    # Respaldo inteligente basado en el target si la metadata no es explícita
    if target is not None:
        try:
            if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target) or pd.api.types.is_bool_dtype(target):
                return True
            if pd.api.types.is_integer_dtype(target) and target.nunique() <= 30:
                return True
        except Exception:
            pass

    return False


# ============================================================
# REGISTRO
# ============================================================

def cargar_registro():

    os.makedirs(
        REGISTRO_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Si ya existe
    # --------------------------------------------------------

    if os.path.exists(
        REGISTRO_FILE
    ):

        try:

            return pd.read_csv(
                REGISTRO_FILE,
                encoding="utf-8-sig"
            )

        except Exception:

            print(
                "\nADVERTENCIA: no se pudo leer "
                "el registro existente."
            )


    # --------------------------------------------------------
    # Registro nuevo
    # --------------------------------------------------------

    columnas = [

        "Fecha_Evaluacion",
        "UCI_ID",
        "Dataset",
        "Categoria",
        "Instancias",
        "Features",
        "Target",
        "Num_Clases",
        "Min_Muestras_Clase",
        "Max_Muestras_Clase",
        "Nulos_NaN",
        "Marcadores_Faltantes",
        "Columnas_Duplicadas",
        "Columnas_Constantes",
        "Clasificacion",
        "Resultado",
        "Motivo",
        "Nombre_CSV"

    ]


    return pd.DataFrame(
        columns=columnas
    )


def guardar_registro(datos):

    os.makedirs(
        REGISTRO_DIR,
        exist_ok=True
    )

    registro = cargar_registro()

    nueva_fila = pd.DataFrame(
        [datos]
    )

    registro = pd.concat(
        [
            registro,
            nueva_fila
        ],
        ignore_index=True
    )

    registro.to_csv(
        REGISTRO_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluar_dataset(
    uci_id
):

    # --------------------------------------------------------
    # Fecha
    # --------------------------------------------------------

    fecha = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # --------------------------------------------------------
    # Registro inicial
    # --------------------------------------------------------

    registro_base = {

        "Fecha_Evaluacion": fecha,
        "UCI_ID": uci_id,
        "Dataset": "",
        "Categoria": "",
        "Instancias": "",
        "Features": "",
        "Target": "",
        "Num_Clases": "",
        "Min_Muestras_Clase": "",
        "Max_Muestras_Clase": "",
        "Nulos_NaN": "",
        "Marcadores_Faltantes": "",
        "Columnas_Duplicadas": "",
        "Columnas_Constantes": "",
        "Clasificacion": "",
        "Resultado": "",
        "Motivo": "",
        "Nombre_CSV": ""

    }


    # ========================================================
    # ENCABEZADO
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "                    EVALUACIÓN UCI"
    )

    print(
        "=" * 75
    )


    # ========================================================
    # DESCARGA
    # ========================================================

    print(
        f"\nDescargando dataset UCI ID: {uci_id} ..."
    )


    try:

        dataset = fetch_ucirepo(
            id=uci_id
        )

    except Exception as e:

        print(
            "\nERROR AL DESCARGAR EL DATASET"
        )

        print(e)

        registro_base["Resultado"] = "ERROR"

        registro_base["Motivo"] = (
            f"No fue posible descargar el dataset: {e}"
        )

        guardar_registro(
            registro_base
        )

        return


    # ========================================================
    # NOMBRE
    # ========================================================

    try:

        nombre_dataset = dataset.metadata.name

    except Exception:

        try:

            nombre_dataset = dataset.metadata.get(
                "name",
                f"Dataset_UCI_{uci_id}"
            )

        except Exception:

            nombre_dataset = (
                f"Dataset_UCI_{uci_id}"
            )


    registro_base["Dataset"] = (
        nombre_dataset
    )


    # ========================================================
    # FEATURES Y TARGET
    # ========================================================

    X = dataset.data.features
    y = dataset.data.targets


    if X is None or y is None:

        motivo = (
            "UCI no devolvió correctamente "
            "features y/o target."
        )

        print(
            f"\nNO APTO: {motivo}"
        )

        registro_base["Resultado"] = (
            "NO_APTO"
        )

        registro_base["Motivo"] = (
            motivo
        )

        guardar_registro(
            registro_base
        )

        return


    X = X.copy()


    # ========================================================
    # TARGET
    # ========================================================

    (
        target_name,
        target,
        target_unico
    ) = obtener_target_desde_uci(
        dataset,
        y
    )


    registro_base["Target"] = (
        target_name
    )


    # ========================================================
    # VALIDAR TARGET ÚNICO
    # ========================================================

    if not target_unico:

        motivo = (
            "El dataset tiene múltiples variables objetivo "
            "o no fue posible identificar un único target. "
            "Este experimento requiere un único target."
        )

        print(
            f"\nNO APTO: {motivo}"
        )

        registro_base["Resultado"] = (
            "NO_APTO"
        )

        registro_base["Motivo"] = (
            motivo
        )

        guardar_registro(
            registro_base
        )

        return


    # ========================================================
    # TAMAÑO
    # ========================================================

    n_instancias = len(X)
    n_features = X.shape[1]


    registro_base["Instancias"] = (
        n_instancias
    )

    registro_base["Features"] = (
        n_features
    )


    # ========================================================
    # TAREA (CORREGIDO CON SOPORTE ASSOCIATED_TASKS Y TARGET)
    # ========================================================

    es_clasificacion = (
        detectar_tarea_clasificacion(
            dataset,
            target=target
        )
    )


    # ========================================================
    # CLASES
    # ========================================================

    clases = target.value_counts(
        dropna=False
    )

    n_clases = len(clases)


    if len(clases) > 0:

        min_muestras_clase = int(
            clases.min()
        )

        max_muestras_clase = int(
            clases.max()
        )

    else:

        min_muestras_clase = 0
        max_muestras_clase = 0


    registro_base["Num_Clases"] = (
        n_clases
    )

    registro_base["Min_Muestras_Clase"] = (
        min_muestras_clase
    )

    registro_base["Max_Muestras_Clase"] = (
        max_muestras_clase
    )


    # ========================================================
    # NULOS
    # ========================================================

    nulos_X = int(
        X.isna().sum().sum()
    )

    nulos_y = int(
        target.isna().sum()
    )

    total_nulos = (
        nulos_X + nulos_y
    )


    registro_base["Nulos_NaN"] = (
        total_nulos
    )


    # ========================================================
    # MARCADORES DE FALTANTES
    # ========================================================

    marcadores_X = (
        detectar_marcadores_faltantes(
            X
        )
    )

    marcadores_y = (
        detectar_marcadores_faltantes(
            pd.DataFrame(
                {
                    target_name: target
                }
            )
        )
    )


    marcadores = {}

    for clave, valor in marcadores_X.items():

        marcadores[clave] = valor


    for clave, valor in marcadores_y.items():

        marcadores[
            f"target_{clave}"
        ] = valor


    registro_base[
        "Marcadores_Faltantes"
    ] = (
        str(marcadores)
        if marcadores
        else "Ninguno"
    )


    # ========================================================
    # NOMBRES DE COLUMNAS
    # ========================================================

    columnas_sin_nombre = [

        col

        for col in X.columns

        if (
            col is None
            or str(col).strip() == ""
            or str(col)
            .lower()
            .startswith("unnamed:")
        )

    ]


    columnas_duplicadas = list(
        X.columns[
            X.columns.duplicated()
        ]
    )


    # ========================================================
    # COLUMNAS CONSTANTES
    # ========================================================

    columnas_constantes = [

        col

        for col in X.columns

        if X[col].nunique(
            dropna=False
        ) <= 1

    ]


    registro_base[
        "Columnas_Duplicadas"
    ] = (

        str(columnas_duplicadas)

        if columnas_duplicadas

        else "Ninguna"

    )


    registro_base[
        "Columnas_Constantes"
    ] = (

        str(columnas_constantes)

        if columnas_constantes

        else "Ninguna"

    )


    # ========================================================
    # CATEGORÍA
    # ========================================================

    categoria = determinar_categoria(
        n_instancias,
        n_features
    )


    registro_base["Categoria"] = (

        categoria

        if categoria

        else "FUERA_DE_CRITERIO"

    )


    # ========================================================
    # INFORMACIÓN GENERAL
    # ========================================================

    print(
        f"\nDataset UCI : {nombre_dataset}"
    )

    print(
        f"UCI ID      : {uci_id}"
    )

    print(
        f"Instancias  : {n_instancias}"
    )

    print(
        f"Features    : {n_features}"
    )

    print(
        f"Target      : {target_name}"
    )

    print(
        f"Clases      : {n_clases}"
    )


    # ========================================================
    # DISTRIBUCIÓN
    # ========================================================

    print(
        "\nDistribución de clases:"
    )

    print(
        clases.to_string()
    )


    # ========================================================
    # VALIDACIONES
    # ========================================================

    print(
        "\n" + "-" * 75
    )

    print(
        "VALIDACIONES"
    )

    print(
        "-" * 75
    )


    # --------------------------------------------------------
    # CLASIFICACIÓN
    # --------------------------------------------------------

    print(
        f"Clasificación        : "
        f"{'OK' if es_clasificacion else 'NO'}"
    )


    registro_base["Clasificacion"] = (

        "SI"

        if es_clasificacion

        else "NO"

    )


    # --------------------------------------------------------
    # CATEGORÍA
    # --------------------------------------------------------

    if categoria == "PEQUEÑO":

        print(
            "Categoría            : PEQUEÑO"
        )

        print(
            f"Instancias           : OK "
            f"(≤ {SMALL_MAX_INSTANCES})"
        )

        print(
            f"Features             : OK "
            f"({SMALL_MIN_FEATURES}-"
            f"{SMALL_MAX_FEATURES})"
        )


    elif categoria == "MEDIANO":

        print(
            "Categoría            : MEDIANO"
        )

        print(
            f"Instancias           : OK "
            f"(> {SMALL_MAX_INSTANCES})"
        )

        print(
            f"Features             : OK "
            f"({MEDIUM_MIN_FEATURES}-"
            f"{MEDIUM_MAX_FEATURES})"
        )


    else:

        print(
            "Categoría            : NO CUMPLE"
        )

        print(
            "Instancias/Features  : NO"
        )


    # --------------------------------------------------------
    # CANTIDAD DE CLASES
    # --------------------------------------------------------

    clases_ok = (
        n_clases >= MIN_CLASSES
    )


    print(
        f"Cantidad de clases   : "
        f"{'OK' if clases_ok else 'NO'} "
        f"(mínimo {MIN_CLASSES})"
    )


    # --------------------------------------------------------
    # MUESTRAS POR CLASE
    # --------------------------------------------------------

    clases_muestras_ok = (
        min_muestras_clase
        >= MIN_SAMPLES_PER_CLASS
    )


    print(
        f"Muestras por clase   : "
        f"{'OK' if clases_muestras_ok else 'NO'} "
        f"(mínimo {MIN_SAMPLES_PER_CLASS})"
    )


    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    nulos_ok = (
        total_nulos == 0
    )


    print(
        f"Valores NaN          : "
        f"{'OK' if nulos_ok else 'NO'} "
        f"({total_nulos})"
    )


    # --------------------------------------------------------
    # MARCADORES
    # --------------------------------------------------------

    if marcadores:

        print(
            f"Marcadores faltantes : "
            f"REVISAR {marcadores}"
        )

    else:

        print(
            "Marcadores faltantes : OK"
        )


    # --------------------------------------------------------
    # NOMBRES
    # --------------------------------------------------------

    nombres_ok = (

        len(columnas_sin_nombre) == 0

        and

        len(columnas_duplicadas) == 0

    )


    print(
        f"Nombres de columnas  : "
        f"{'OK' if nombres_ok else 'REVISAR'}"
    )


    # ========================================================
    # COLUMNAS CONSTANTES
    # ========================================================

    if columnas_constantes:

        print(
            "\nColumnas constantes detectadas:"
        )

        for columna in columnas_constantes:

            print(
                f"  - {columna}"
            )

        print(
            "\nNOTA: las columnas constantes "
            "NO provocan rechazo automático."
        )


    # ========================================================
    # COLUMNAS ORIGINALES
    # ========================================================

    print(
        "\nColumnas originales:"
    )


    for i, columna in enumerate(
        X.columns,
        start=1
    ):

        print(
            f"  {i:2d}. {columna}"
        )


    # ========================================================
    # DECISIÓN FINAL
    # ========================================================

    motivos = []


    # --------------------------------------------------------
    # Clasificación
    # --------------------------------------------------------

    if not es_clasificacion:

        motivos.append(
            "UCI no identifica el dataset como clasificación."
        )


    # --------------------------------------------------------
    # Categoría
    # --------------------------------------------------------

    if categoria is None:

        motivos.append(
            "No cumple el rango de instancias/features."
        )


    # --------------------------------------------------------
    # Clases
    # --------------------------------------------------------

    if not clases_ok:

        motivos.append(
            f"Tiene menos de "
            f"{MIN_CLASSES} clases."
        )


    # --------------------------------------------------------
    # Muestras por clase
    # --------------------------------------------------------

    if not clases_muestras_ok:

        motivos.append(
            f"Una o más clases tienen menos de "
            f"{MIN_SAMPLES_PER_CLASS} muestras."
        )


    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    if not nulos_ok:

        motivos.append(
            f"Contiene {total_nulos} valores NaN."
        )


    # --------------------------------------------------------
    # Marcadores
    # --------------------------------------------------------

    if marcadores:

        motivos.append(
            "Contiene posibles marcadores "
            f"de valores faltantes: {marcadores}."
        )


    # --------------------------------------------------------
    # Nombres
    # --------------------------------------------------------

    if not nombres_ok:

        motivos.append(
            "Existen nombres de columnas "
            "faltantes o duplicados."
        )


    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    apto = (
        len(motivos) == 0
    )


    # ========================================================
    # GUARDAR DATASET
    # ========================================================

    print(
        "\n" + "=" * 75
    )


    if apto:

        print(
            "RESULTADO: DATASET APTO"
        )

        print(
            f"CATEGORÍA: {categoria}"
        )


        # ----------------------------------------------------
        # COPIA FEATURES
        # ----------------------------------------------------

        df_final = X.copy()


        # ----------------------------------------------------
        # Evitar conflicto con el nombre del target
        # ----------------------------------------------------

        target_name_final = (
            target_name
        )


        if target_name_final in df_final.columns:

            contador = 1

            nuevo_target = "target"


            while nuevo_target in df_final.columns:

                nuevo_target = (
                    f"target_{contador}"
                )

                contador += 1


            target_name_final = (
                nuevo_target
            )


        # ----------------------------------------------------
        # Agregar target
        # ----------------------------------------------------

        df_final[
            target_name_final
        ] = target.values


        # ----------------------------------------------------
        # Crear carpeta
        # ----------------------------------------------------

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Nombre archivo
        # ----------------------------------------------------

        nombre_archivo = (
            limpiar_nombre_archivo(
                nombre_dataset
            )
        )


        ruta_csv = os.path.join(
            DATA_DIR,
            f"{nombre_archivo}.csv"
        )


        # ----------------------------------------------------
        # Guardar
        # ----------------------------------------------------

        df_final.to_csv(
            ruta_csv,
            index=False,
            encoding="utf-8-sig"
        )


        registro_base[
            "Resultado"
        ] = "APTO"


        registro_base[
            "Motivo"
        ] = (
            "Cumple todos los criterios establecidos."
        )


        registro_base[
            "Nombre_CSV"
        ] = (
            f"{nombre_archivo}.csv"
        )


        print(
            "\nCSV guardado correctamente:"
        )

        print(
            ruta_csv
        )


        print(
            f"\nTarget guardado: "
            f"{target_name_final}"
        )


    # ========================================================
    # NO APTO
    # ========================================================

    else:

        print(
            "RESULTADO: DATASET NO APTO"
        )


        print(
            "\nMotivos:"
        )


        for motivo in motivos:

            print(
                f"  - {motivo}"
            )


        print(
            "\nEl dataset NO será guardado "
            "como candidato."
        )


        registro_base[
            "Resultado"
        ] = "NO_APTO"


        registro_base[
            "Motivo"
        ] = " | ".join(
            motivos
        )


    # ========================================================
    # GUARDAR REGISTRO SIEMPRE
    # ========================================================

    guardar_registro(
        registro_base
    )


    print(
        "\nRegistro actualizado:"
    )

    print(
        REGISTRO_FILE
    )


    print(
        "=" * 75
    )


# ============================================================
# MENÚ
# ============================================================

def menu():

    while True:

        print("\n")

        print(
            "=" * 75
        )

        print(
            "                SELECTOR DE DATASETS UCI"
        )

        print(
            "=" * 75
        )


        print(
            f"Pequeño : ≤ {SMALL_MAX_INSTANCES} instancias, "
            f"{SMALL_MIN_FEATURES}-{SMALL_MAX_FEATURES} features"
        )


        print(
            f"Mediano : > {SMALL_MAX_INSTANCES} instancias, "
            f"{MEDIUM_MIN_FEATURES}-{MEDIUM_MAX_FEATURES} features"
        )


        print(
            f"Mínimo por clase: "
            f"{MIN_SAMPLES_PER_CLASS}"
        )


        print(
            "-" * 75
        )


        print(
            "1. Evaluar dataset UCI"
        )

        print(
            "2. Ver ubicación del registro"
        )

        print(
            "3. Salir"
        )


        print(
            "=" * 75
        )


        opcion = input(
            "\nSelecciona una opción: "
        ).strip()


        # ====================================================
        # OPCIÓN 1
        # ====================================================

        if opcion == "1":

            print(
                "\nEjemplo:"
            )

            print(
                "Statlog German Credit Data → ID 144"
            )


            entrada = input(
                "\nIngresa el ID del dataset de UCI: "
            ).strip()


            try:

                uci_id = int(
                    entrada
                )

            except ValueError:

                print(
                    "\nERROR: el ID debe ser un número."
                )

                continue


            evaluar_dataset(
                uci_id
            )


            input(
                "\nPresiona ENTER para regresar al menú..."
            )


        # ====================================================
        # OPCIÓN 2
        # ====================================================

        elif opcion == "2":

            print(
                "\nEl registro se encuentra en:"
            )

            print(
                REGISTRO_FILE
            )


            input(
                "\nPresiona ENTER para regresar al menú..."
            )


        # ====================================================
        # OPCIÓN 3
        # ====================================================

        elif opcion == "3":

            print(
                "\nSaliendo..."
            )

            break


        # ====================================================
        # OPCIÓN INVÁLIDA
        # ====================================================

        else:

            print(
                "\nOpción no válida."
            )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    menu()