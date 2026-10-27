import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    print("ERROR: no está instalado ucimlrepo.")
    print("\nInstálalo con:")
    print("pip install ucimlrepo")
    raise SystemExit

from src.project_config import DATA_DIR, preparar_directorios

REGISTRO_DIR = DATA_DIR / "Registro"
REGISTRO_FILE = REGISTRO_DIR / "registro_datasets.csv"

MIN_CLASSES = 2
MIN_SAMPLES_PER_CLASS = 10

# Columnas oficiales y orden exacto del registro
COLUMNAS_REGISTRO = [
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


# ============================================================
# UTILIDADES
# ============================================================

def limpiar_nombre_archivo(nombre):
    nombre = str(nombre).strip()
    # Caracteres no permitidos en nombres de Windows
    nombre = re.sub(r'[<>:"/\\|?*]', '_', nombre)
    # Reemplazar espacios múltiples por _
    nombre = re.sub(r'\s+', '_', nombre)
    return nombre[:150]


def obtener_texto_metadata(valor):
    if valor is None:
        return ""
    if isinstance(valor, (list, tuple, set)):
        return ", ".join(map(str, valor))
    return str(valor)


def obtener_categoria_oficial_uci(dataset):
    """
    Extrae la categoría o área temática oficial directamente
    de la metadata de UCI (Subject Area / Category).
    """
    try:
        metadata = dataset.metadata
        if metadata is not None:
            # Buscar campos oficiales de UCI
            for campo in ["subject_area", "category", "area", "purpose"]:
                val = None
                if hasattr(metadata, campo):
                    val = getattr(metadata, campo)
                elif isinstance(metadata, dict):
                    val = metadata.get(campo, None)
                
                if val is not None:
                    texto = obtener_texto_metadata(val).strip()
                    if texto:
                        return texto
    except Exception:
        pass
    
    return "General"


def detectar_marcadores_faltantes(df):
    """
    Busca marcadores comunes utilizados para representar
    valores faltantes dentro de columnas de texto.
    """
    marcadores = ["?", "NA", "N/A", "NULL", "null", "", "unknown", "Unknown"]
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
                cantidad_total += int(cantidad)

        if cantidad_total > 0:
            encontrados[marcador] = cantidad_total

    return encontrados


def obtener_target_desde_uci(dataset, y):
    """
    Identifica el target utilizando la información de UCI.
    """
    try:
        variables = dataset.variables
        if (
            variables is not None
            and not variables.empty
            and "role" in variables.columns
            and "name" in variables.columns
        ):
            roles = variables["role"].astype(str).str.strip().str.lower()
            targets_uci = variables[roles == "target"]

            if len(targets_uci) == 1:
                target_name = str(targets_uci.iloc[0]["name"])

                if isinstance(y, pd.DataFrame):
                    if target_name in y.columns:
                        target = y[target_name].copy()
                        return target_name, target, True

                if isinstance(y, pd.Series):
                    target = y.copy()
                    if target.name is None:
                        target.name = target_name
                    return target_name, target, True

            elif len(targets_uci) > 1:
                nombres = ", ".join(map(str, targets_uci["name"].tolist()))
                return nombres, None, False
    except Exception:
        pass

    if isinstance(y, pd.Series):
        target_name = str(y.name) if y.name is not None else "target"
        return target_name, y.copy(), True

    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            target_name = str(y.columns[0])
            return target_name, y.iloc[:, 0].copy(), True
        nombres = ", ".join(map(str, y.columns))
        return nombres, None, False

    return "target", None, False


def detectar_tarea_clasificacion(dataset, target=None):
    try:
        metadata = dataset.metadata
    except Exception:
        metadata = None

    task_textos = []

    if metadata is not None:
        candidatos_campos = ["associated_tasks", "task", "purpose"]
        for campo in candidatos_campos:
            valor = None
            try:
                if hasattr(metadata, campo):
                    valor = getattr(metadata, campo)
                elif isinstance(metadata, dict):
                    valor = metadata.get(campo, None)
            except Exception:
                pass
            
            if valor is not None:
                task_textos.append(obtener_texto_metadata(valor).lower())

    texto_completo = " ".join(task_textos)

    if "classification" in texto_completo or "clasificación" in texto_completo:
        return True

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
    os.makedirs(REGISTRO_DIR, exist_ok=True)

    if os.path.exists(REGISTRO_FILE):
        try:
            df = pd.read_csv(REGISTRO_FILE, encoding="utf-8-sig")
            # Asegurar que todas las columnas existan y estén en orden
            for col in COLUMNAS_REGISTRO:
                if col not in df.columns:
                    df[col] = ""
            return df[COLUMNAS_REGISTRO]
        except Exception:
            print("\nADVERTENCIA: no se pudo leer el registro existente. Se creará uno nuevo.")

    return pd.DataFrame(columns=COLUMNAS_REGISTRO)


def guardar_registro(datos):
    os.makedirs(REGISTRO_DIR, exist_ok=True)
    registro = cargar_registro()
    
    # Crear DataFrame con la nueva fila asegurando las columnas correctas
    nueva_fila = pd.DataFrame([datos])
    for col in COLUMNAS_REGISTRO:
        if col not in nueva_fila.columns:
            nueva_fila[col] = ""
    nueva_fila = nueva_fila[COLUMNAS_REGISTRO]

    registro = pd.concat([registro, nueva_fila], ignore_index=True)
    registro.to_csv(REGISTRO_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluar_dataset(uci_id):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Asegurar que se asignan correctamente Fecha y UCI_ID puro
    registro_base = {
        "Fecha_Evaluacion": fecha,
        "UCI_ID": int(uci_id),
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

    print("\n" + "=" * 75)
    print("                    EVALUACIÓN UCI")
    print("=" * 75)

    print(f"\nDescargando dataset UCI ID: {uci_id} ...")

    try:
        dataset = fetch_ucirepo(id=uci_id)
    except Exception as e:
        print("\nERROR AL DESCARGAR EL DATASET")
        print(e)
        return

    # Obtener categoría oficial de UCI
    categoria_uci = obtener_categoria_oficial_uci(dataset)
    registro_base["Categoria"] = categoria_uci

    try:
        nombre_dataset = dataset.metadata.name
    except Exception:
        try:
            nombre_dataset = dataset.metadata.get("name", f"Dataset_UCI_{uci_id}")
        except Exception:
            nombre_dataset = f"Dataset_UCI_{uci_id}"

    registro_base["Dataset"] = nombre_dataset

    X = dataset.data.features
    y = dataset.data.targets

    if X is None or y is None:
        motivo = "UCI no devolvió correctamente features y/o target."
        print(f"\nNO APTO: {motivo}")
        print("\nEl dataset NO será guardado ni registrado.")
        return

    X = X.copy()

    target_name, target, target_unico = obtener_target_desde_uci(dataset, y)
    registro_base["Target"] = target_name

    if not target_unico:
        motivo = "El dataset tiene múltiples variables objetivo o no fue posible identificar un único target."
        print(f"\nNO APTO: {motivo}")
        print("\nEl dataset NO será guardado ni registrado.")
        return

    n_instancias = len(X)
    n_features = X.shape[1]

    registro_base["Instancias"] = n_instancias
    registro_base["Features"] = n_features

    es_clasificacion = detectar_tarea_clasificacion(dataset, target=target)

    clases = target.value_counts(dropna=False)
    n_clases = len(clases)

    if len(clases) > 0:
        min_muestras_clase = int(clases.min())
        max_muestras_clase = int(clases.max())
    else:
        min_muestras_clase = 0
        max_muestras_clase = 0

    registro_base["Num_Clases"] = n_clases
    registro_base["Min_Muestras_Clase"] = min_muestras_clase
    registro_base["Max_Muestras_Clase"] = max_muestras_clase

    nulos_X = int(X.isna().sum().sum())
    nulos_y = int(target.isna().sum())
    total_nulos = nulos_X + nulos_y

    registro_base["Nulos_NaN"] = total_nulos

    marcadores_X = detectar_marcadores_faltantes(X)
    marcadores_y = detectar_marcadores_faltantes(pd.DataFrame({target_name: target}))

    marcadores = {}
    for clave, valor in marcadores_X.items():
        marcadores[clave] = valor
    for clave, valor in marcadores_y.items():
        marcadores[f"target_{clave}"] = valor

    registro_base["Marcadores_Faltantes"] = str(marcadores) if marcadores else "Ninguno"

    columnas_sin_nombre = [
        col for col in X.columns
        if col is None or str(col).strip() == "" or str(col).lower().startswith("unnamed:")
    ]

    columnas_duplicadas = list(X.columns[X.columns.duplicated()])
    columnas_constantes = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]

    registro_base["Columnas_Duplicadas"] = str(columnas_duplicadas) if columnas_duplicadas else "Ninguna"
    registro_base["Columnas_Constantes"] = str(columnas_constantes) if columnas_constantes else "Ninguna"

    print(f"\nDataset UCI : {nombre_dataset}")
    print(f"Categoría   : {categoria_uci}")
    print(f"UCI ID      : {uci_id}")
    print(f"Instancias  : {n_instancias}")
    print(f"Features    : {n_features}")
    print(f"Target      : {target_name}")
    print(f"Clases      : {n_clases}")

    print("\nDistribución de clases:")
    print(clases.to_string())

    print("\n" + "-" * 75)
    print("VALIDACIONES")
    print("-" * 75)

    es_clasificacion_ok = es_clasificacion
    print(f"Clasificación       : {'OK' if es_clasificacion_ok else 'NO'}")
    registro_base["Clasificacion"] = "SI" if es_clasificacion_ok else "NO"

    clases_ok = n_clases >= MIN_CLASSES
    print(f"Cantidad de clases   : {'OK' if clases_ok else 'NO'} (mínimo {MIN_CLASSES})")

    clases_muestras_ok = min_muestras_clase >= MIN_SAMPLES_PER_CLASS
    print(f"Muestras por clase   : {'OK' if clases_muestras_ok else 'NO'} (mínimo {MIN_SAMPLES_PER_CLASS})")

    nulos_ok = total_nulos == 0
    print(f"Valores NaN          : {'OK' if nulos_ok else 'NO'} ({total_nulos})")

    if marcadores:
        print(f"Marcadores faltantes : REVISAR {marcadores}")
    else:
        print("Marcadores faltantes : OK")

    nombres_ok = len(columnas_sin_nombre) == 0 and len(columnas_duplicadas) == 0
    print(f"Nombres de columnas  : {'OK' if nombres_ok else 'REVISAR'}")

    motivos = []
    if not es_clasificacion_ok:
        motivos.append("UCI no identifica el dataset como clasificación.")
    if not clases_ok:
        motivos.append(f"Tiene menos de {MIN_CLASSES} clases.")
    if not clases_muestras_ok:
        motivos.append(f"Una o más clases tienen menos de {MIN_SAMPLES_PER_CLASS} muestras.")
    if not nulos_ok:
        motivos.append(f"Contiene {total_nulos} valores NaN.")
    if marcadores:
        motivos.append(f"Contiene posibles marcadores de valores faltantes: {marcadores}.")
    if not nombres_ok:
        motivos.append("Existen nombres de columnas faltantes o duplicados.")

    apto = len(motivos) == 0

    print("\n" + "=" * 75)
    if apto:
        print("RESULTADO: DATASET APTO")

        df_final = X.copy()
        target_name_final = target_name

        if target_name_final in df_final.columns:
            contador = 1
            nuevo_target = "target"
            while nuevo_target in df_final.columns:
                nuevo_target = f"target_{contador}"
                contador += 1
            target_name_final = nuevo_target

        df_final[target_name_final] = target.values

        os.makedirs(DATA_DIR, exist_ok=True)

        nombre_archivo = limpiar_nombre_archivo(nombre_dataset)
        ruta_csv = os.path.join(DATA_DIR, f"{nombre_archivo}.csv")

        df_final.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

        registro_base["Resultado"] = "APTO"
        registro_base["Motivo"] = "Cumple todos los criterios esenciales establecidos."
        registro_base["Nombre_CSV"] = f"{nombre_archivo}.csv"

        print("\nCSV guardado correctamente:")
        print(ruta_csv)

        guardar_registro(registro_base)
        print("\nRegistro actualizado:")
        print(REGISTRO_FILE)

    else:
        print("RESULTADO: DATASET NO APTO")
        print("\nMotivos:")
        for motivo in motivos:
            print(f"  - {motivo}")
        print("\nEl dataset NO será guardado ni registrado.")

    print("=" * 75)


# ============================================================
# MENÚ
# ============================================================

def menu():
    while True:
        print("\n")
        print("=" * 75)
        print("                    SELECTOR DE DATASETS UCI")
        print("=" * 75)
        print(f"Mínimo por clase: {MIN_SAMPLES_PER_CLASS}")
        print("-" * 75)
        print("1. Evaluar dataset UCI")
        print("2. Ver ubicación del registro")
        print("3. Salir")
        print("=" * 75)

        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "1":
            print("\nEjemplo:")
            print("Statlog German Credit Data → ID 144")

            entrada = input("\nIngresa el ID del dataset de UCI: ").strip()

            try:
                uci_id = int(entrada)
            except ValueError:
                print("\nERROR: el ID debe ser un número.")
                continue

            evaluar_dataset(uci_id)
            input("\nPresiona ENTER para regresar al menú...")

        elif opcion == "2":
            print("\nEl registro se encuentra en:")
            print(REGISTRO_FILE)
            input("\nPresiona ENTER para regresar al menú...")

        elif opcion == "3":
            print("\nSaliendo...")
            break

        else:
            print("\nOpción no válida.")


if __name__ == "__main__":
    menu()